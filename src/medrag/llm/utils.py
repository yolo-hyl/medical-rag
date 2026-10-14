"""模型输出的通用处理"""
import re
from typing import Tuple

_THINK_RE = re.compile(r"<think>(.*?)</think>\s*", flags=re.DOTALL)


def split_think(text: str) -> Tuple[str, str]:
    """把 `<think>…</think>` 从正文里拆出来，返回 (思考内容, 正文)。

    Qwen3 等模型在 OpenAI 兼容接口上有两种返回方式：思考内容放在 reasoning_content 字段，
    或者直接内嵌在 content 里；后一种需要在这里剥掉，否则会混进回答和对话历史。
    """
    text = text or ""
    reasoning = "\n".join(m.strip() for m in _THINK_RE.findall(text))
    return reasoning, _THINK_RE.sub("", text).strip()


def strip_think(text: str) -> str:
    return split_think(text)[1]
