from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from langchain_core.messages import AIMessage, HumanMessage

from ...llm.utils import strip_think

if TYPE_CHECKING:
    from .medical_agent import MedicalAgentState

logger = logging.getLogger(__name__)


def get_last_human(state: MedicalAgentState, stage: str = "asking_messages") -> str:
    mess = "dialogue_messages" if stage == "dialogue" else "asking_messages"
    for m in reversed(state.get(mess, [])):
        if isinstance(m, HumanMessage) and getattr(m, "content", ""):
            return m.content
    return ""


def strip_think_get_tokens(msg: AIMessage) -> dict:
    """去掉思考内容，并取出用于估算「每个字符多少 token」的长度与输出 token 数"""
    text = msg.content
    msg_len = len(text)
    usage = msg.usage_metadata or {}
    msg_token_len = usage.get("output_tokens")
    if msg_token_len is None:
        msg_token_len = (msg.response_metadata.get("token_usage") or {}).get("completion_tokens", 0)
        if not msg_token_len:
            logger.debug("模型响应里没有输出 token 数，按 0 计入统计")
    dur = msg.response_metadata.get("total_duration", 0) / 1e9
    return {
        "msg": strip_think(text),
        "msg_len": msg_len,
        "msg_token_len": msg_token_len,
        "generate_time": dur,
    }
