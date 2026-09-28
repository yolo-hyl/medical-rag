"""多轮对话的记忆管理（MultiDialogueRag 与 MedicalAgent 共用）

记忆分两层：
    短期记忆  最近若干轮的原文消息（只有问答，不含文档）
    长期记忆  从短期记忆里移出的旧消息压缩成的摘要

本模块与存储无关：只接收消息、token 统计，返回要丢弃多少条旧消息、新增什么摘要，
读写 Redis 或 state 由调用方负责。

    TokenStats    会话级的 token 统计，用来估算「一个字符约多少 token」
    TokenBudget   按配置估算 token：判断历史是否超预算、按剩余预算装入参考资料
    BaseMemory    记忆策略接口，替换策略只需实现 compress()
        SummaryBufferMemory  超预算时把最旧的一部分消息压成摘要（默认策略）
        SlidingWindowMemory  只保留最近 N 条消息，不生成摘要
"""
from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import tiktoken
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field

from ..config.models import MemoryConfig
from ..prompts.templates import get_prompt_template

logger = logging.getLogger(__name__)

# 没有历史统计时的初始值：给一个很小的值，避免第一轮就误判为超长
INIT_TOKENS_PER_CHAR = 1e-5


# ===================== token 估算函数注册表 =====================

ESTIMATE_FUNCTION_REGISTRY: Dict[str, Callable[[str], float]] = {}


def register_estimate_function(name):
    """装饰器：注册函数到字典"""
    def decorator(func):
        ESTIMATE_FUNCTION_REGISTRY[name] = func
        return func
    return decorator


@register_estimate_function("tiktoken")
def estimate_tokens(text: str) -> int:
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = len(encoding.encode(text))
    return tokens


# ===================== token 统计 =====================

class TokenStats(BaseModel):
    """会话内每次 LLM 输出的字符数与 token 数，序列化格式与 Redis 中已有数据一致"""
    msg_len: List[int] = Field(default_factory=list)
    msg_token_len: List[int] = Field(default_factory=list)

    def add(self, msg_len: int, msg_token_len: int) -> None:
        self.msg_len.append(msg_len)
        self.msg_token_len.append(msg_token_len)

    def avg_tokens_per_char(self) -> float:
        """平均一个字符消耗多少 token"""
        if not sum(self.msg_len):
            return INIT_TOKENS_PER_CHAR
        return sum(self.msg_token_len) / sum(self.msg_len)


# ===================== token 预算 =====================

class TokenBudget:
    """按配置估算 token 用量"""

    def __init__(self, config: MemoryConfig) -> None:
        self.config = config

    @property
    def use_estimate_fun(self) -> bool:
        return self.config.estimate_token_fun != "avg"

    def estimate(self, text: str) -> float:
        """按配置的估计函数计算文本的 token 数；配置为 avg 时由调用方另行处理"""
        return ESTIMATE_FUNCTION_REGISTRY[self.config.estimate_token_fun](text)

    @property
    def limit(self) -> float:
        return self.config.llm_max_token * self.config.max_token_threshold

    def over_budget(self, history_text: str, stats: TokenStats) -> bool:
        """历史加上预测的本轮回答，是否会超出模型上限"""
        if self.use_estimate_fun:
            try:
                return self.estimate(history_text) >= self.limit
            except Exception:
                logger.exception("注册的估计函数错误，回退到默认avg实现...")
        return self._avg_over_budget(history_text, stats)

    def _avg_over_budget(self, history_text: str, stats: TokenStats) -> bool:
        """用平均 token 占比预测本轮总量是否会超出模型上限"""
        if not stats.msg_len:
            return False
        avg = stats.avg_tokens_per_char()
        # 用历史消息的平均长度，预测这次回答可能会生成多少 token
        predict_token = avg * sum(stats.msg_len) / max(1, len(stats.msg_len))
        curr_all_token = predict_token + len(history_text) * avg
        return curr_all_token > self.limit

    def fit_documents(
        self,
        documents: List[str],
        fixed_text: str,
        avg_tokens_per_char: float,
        title: str = "文档",
    ) -> str:
        """扣除 fixed_text（system、历史、问题等）占用后，按剩余预算尽可能多地拼入参考资料（纯计算）"""
        cfg = self.config
        remain_token = cfg.llm_max_token

        if self.use_estimate_fun:
            all_token = self.estimate(fixed_text)
        else:
            all_token = avg_tokens_per_char * len(fixed_text)
        remain_token -= all_token - cfg.llm_max_token * 0.01

        parts = []
        used = 0
        for idx, body in enumerate(documents):
            header = f"## {title}{idx+1}：\n"
            body = body or ""
            if self.use_estimate_fun:
                header_tokens = self.estimate(header)
                body_tokens = self.estimate(body)
            else:
                header_tokens = avg_tokens_per_char * len(header)
                body_tokens = avg_tokens_per_char * len(body)

            if used + header_tokens + body_tokens <= remain_token:
                parts.append(header + body + "\n")
                used += header_tokens + body_tokens
            else:
                # 放不下整篇
                if cfg.console_debug:
                    logger.warning(f"根据给定的token估计方法，预估无法完成全部{title}编码，{title}{idx+1}被截断，后续{title}将无法被放入上下文...")
                remain = remain_token - used - header_tokens
                if remain > 0:
                    # 依据平均token估算可保留字符数
                    keep_chars = max(0, int(remain / max(0.1, avg_tokens_per_char)))
                    if keep_chars > 0:
                        parts.append(header + body[:keep_chars] + "\n...[内容已截断]\n")
                break  # 无论是否部分放入，预算已到

        return "".join(parts)


# ===================== 记忆策略 =====================

@dataclass
class CompressResult:
    """一次压缩的结果：丢弃最旧的 drop 条消息，summary 为新增的摘要（为空表示不生成摘要）"""
    drop: int
    summary: str = ""


def messages_text(messages: List[BaseMessage]) -> str:
    return "\n".join(m.content for m in messages if hasattr(m, "content"))


class BaseMemory(ABC):
    """记忆策略：决定短期记忆何时、如何收缩"""

    @abstractmethod
    async def compress(self, messages: List[BaseMessage], stats: TokenStats) -> Optional[CompressResult]:
        """不需要压缩时返回 None"""


class SummaryBufferMemory(BaseMemory):
    """历史超出 token 预算时，把最旧的 1/cut_dialogue_scale 消息交给 LLM 压成摘要"""

    def __init__(self, config: MemoryConfig, llm: BaseChatModel) -> None:
        self.config = config
        self.budget = TokenBudget(config)
        self.llm = llm

    async def compress(self, messages: List[BaseMessage], stats: TokenStats) -> Optional[CompressResult]:
        if not messages or not self.budget.over_budget(messages_text(messages), stats):
            return None

        if self.config.console_debug:
            logger.warning("对话过长，需要生成摘要...")

        cutoff = max(2, len(messages) // self.config.cut_dialogue_scale)
        summarize_prompt = ChatPromptTemplate.from_messages([
            ("system", get_prompt_template("summary")["system"]),
            MessagesPlaceholder("history"),
            ("human", get_prompt_template("summary")["user"])
        ])
        summary_result: AIMessage = await (summarize_prompt | self.llm).ainvoke({"history": messages[:cutoff]})
        summary = re.sub(r"<think>.*?</think>\s*", "", summary_result.content, flags=re.DOTALL).strip()

        if self.config.console_debug:
            dur = summary_result.response_metadata.get("total_duration", 0) / 1e9
            tokens = (summary_result.usage_metadata or {}).get("total_tokens", 0)
            logger.warning(f"摘要生成完毕，耗时：{dur} s，使用tokens：{tokens}\n摘要文本：\n{summary}")

        return CompressResult(drop=cutoff, summary=summary)


class SlidingWindowMemory(BaseMemory):
    """只保留最近 max_messages 条消息，旧消息直接丢弃，不生成摘要"""

    def __init__(self, max_messages: int = 20) -> None:
        self.max_messages = max_messages

    async def compress(self, messages: List[BaseMessage], stats: TokenStats) -> Optional[CompressResult]:
        if len(messages) <= self.max_messages:
            return None
        return CompressResult(drop=len(messages) - self.max_messages)
