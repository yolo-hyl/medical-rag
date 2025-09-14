from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .MedicalAgent import MedicalAgentState
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
import re

def get_last_human(state: MedicalAgentState, stage: str = "asking_messages") -> str:
    mess = "dialogue_messages" if stage == "dialogue" else "asking_messages"
    for m in reversed(state.get(mess, [])):
        if isinstance(m, HumanMessage) and getattr(m, "content", ""):
            return m.content
    return ""

@staticmethod
def strip_think_get_tokens(msg: AIMessage):
    text = msg.content
    # 用于衡量大概每一个字消耗多少token
    msg_len = len(msg.content)
    try:
        msg_token_len = msg.usage_metadata["output_tokens"]
    except Exception as e1:
        print("尝试第一次解析token输出错误...")
        try:
            msg_token_len = msg.response_metadata["token_usage"]["output_tokens"]
        except Exception as e2:
            print("尝试第二次解析token输出错误，初始化为0")
            msg_token_len = 0
    dur = msg.response_metadata.get("total_duration", 0) / 1e9
    return {
        "msg" : re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip(),
        "msg_len": msg_len,
        "msg_token_len": msg_token_len,
        "generate_time": dur
    }