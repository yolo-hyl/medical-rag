"""OpenAI 兼容协议的 LLM / Embedding 客户端（D2）

核心层统一通过 `openai` SDK 访问模型，不依赖 LangChain。DashScope、vLLM、Ollama（/v1 端点）、
OpenAI 都按 base_url + 密钥接入；以后接入其他模型商，只需再实现满足同样调用方式的类。

    llm = OpenAICompatLLM(cfg.llm)
    result = await llm([{"role": "user", "content": "你好"}])
    result.text, result.usage.completion_tokens

    emb = OpenAICompatEmbedding(cfg.embedding.text_dense)
    vectors = await emb(["文本1", "文本2"])     # 内部按 max_batch 分批、按 concurrency 并发
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from ..config.models import DenseConfig, LLMConfig
from .utils import split_think

Message = Dict[str, Any]   # {"role": "system" | "user" | "assistant" | "tool", "content": "..."}


def _make_client(cfg: LLMConfig | DenseConfig) -> AsyncOpenAI:
    http_client = httpx.AsyncClient(proxy=cfg.proxy) if cfg.proxy else None
    return AsyncOpenAI(
        api_key=cfg.api_key(),
        base_url=cfg.base_url,
        timeout=cfg.timeout,
        max_retries=cfg.max_retries,
        http_client=http_client,
    )


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class LLMResult(BaseModel):
    text: str                       # 去掉思考内容后的正文
    reasoning: str = ""             # 思考内容（reasoning_content 字段或 <think> 块）
    usage: Usage = Field(default_factory=Usage)
    model: str = ""
    latency: float = 0.0            # 墙上时钟耗时，秒
    raw_text: str = ""              # 接口返回的原始 content

    def parse_json(self) -> Any:
        """结构化输出直接取用"""
        return json.loads(self.text)


class LLMDelta(BaseModel):
    """流式输出的一个增量；最后一个增量的 usage 才有值"""
    text: str = ""
    reasoning: str = ""
    usage: Optional[Usage] = None


class OpenAICompatLLM:
    """Chat Completions 客户端"""

    def __init__(self, config: LLMConfig, client: AsyncOpenAI | None = None):
        self.config = config
        self.model = config.model
        self.client = client or _make_client(config)
        self._sem = asyncio.Semaphore(config.concurrency)

    def _params(self, params: dict) -> dict:
        merged: Dict[str, Any] = {"model": self.model, "temperature": self.config.temperature}
        if self.config.max_tokens:
            merged["max_tokens"] = self.config.max_tokens
        if self.config.extra_body:
            merged["extra_body"] = dict(self.config.extra_body)
        merged.update(params)
        return merged

    async def __call__(
        self,
        messages: List[Message],
        *,
        json_schema: Optional[dict] = None,
        **params: Any,
    ) -> LLMResult:
        """json_schema 给定时要求结构化输出；不支持 json_schema 的服务可以改传 response_format={"type": "json_object"}"""
        kwargs = self._params(params)
        if json_schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": json_schema.get("title", "output"), "schema": json_schema},
            }
        start = time.perf_counter()
        async with self._sem:
            resp = await self.client.chat.completions.create(messages=messages, **kwargs)
        latency = time.perf_counter() - start

        msg = resp.choices[0].message
        raw = msg.content or ""
        think_inline, text = split_think(raw)
        reasoning = getattr(msg, "reasoning_content", None) or think_inline
        usage = Usage(**resp.usage.model_dump(include={"prompt_tokens", "completion_tokens", "total_tokens"})) \
            if resp.usage else Usage()
        return LLMResult(text=text, reasoning=reasoning, usage=usage, model=resp.model or self.model,
                         latency=latency, raw_text=raw)

    async def stream(self, messages: List[Message], **params: Any) -> AsyncIterator[LLMDelta]:
        """逐段返回正文增量。内嵌在 content 里的 <think> 块不在这里剥离，由调用方按需处理"""
        kwargs = self._params(params)
        kwargs["stream"] = True
        kwargs["stream_options"] = {"include_usage": True}  # 摘要压缩依赖 token 统计
        async with self._sem:
            stream = await self.client.chat.completions.create(messages=messages, **kwargs)
            async for chunk in stream:
                usage = Usage(**chunk.usage.model_dump(include={"prompt_tokens", "completion_tokens", "total_tokens"})) \
                    if chunk.usage else None
                if not chunk.choices:
                    if usage:
                        yield LLMDelta(usage=usage)
                    continue
                delta = chunk.choices[0].delta
                yield LLMDelta(
                    text=delta.content or "",
                    reasoning=getattr(delta, "reasoning_content", None) or "",
                    usage=usage,
                )

    async def aclose(self) -> None:
        await self.client.close()


class OpenAICompatEmbedding:
    """Embeddings 客户端：按 max_batch 分批、按 concurrency 并发，输出顺序与输入一致"""

    def __init__(self, config: DenseConfig, client: AsyncOpenAI | None = None):
        self.config = config
        self.model = config.model
        self.dim = config.dimension
        self.client = client or _make_client(config)
        self._sem = asyncio.Semaphore(config.concurrency)

    async def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        kwargs: Dict[str, Any] = {"model": self.model, "input": texts}
        if self.config.send_dimensions:
            kwargs["dimensions"] = self.dim
        async with self._sem:
            resp = await self.client.embeddings.create(**kwargs)
        # 接口按 index 返回，不保证顺序
        return [d.embedding for d in sorted(resp.data, key=lambda d: d.index)]

    async def __call__(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        size = self.config.max_batch
        batches = [texts[i:i + size] for i in range(0, len(texts), size)]
        results = await asyncio.gather(*(self._embed_batch(b) for b in batches))
        return [vec for batch in results for vec in batch]

    async def embed_query(self, text: str) -> List[float]:
        return (await self([text]))[0]

    async def aclose(self) -> None:
        await self.client.close()
