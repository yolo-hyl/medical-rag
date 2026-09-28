import time
from typing import Any, Dict, Tuple
from uuid import UUID

from langchain_core.callbacks import AsyncCallbackHandler

# token 估算函数注册表已移到 core/memory.py，这里保留导出以兼容旧的导入路径
from ..core.memory import ESTIMATE_FUNCTION_REGISTRY, estimate_tokens, register_estimate_function  # noqa: F401


class StageTimer(AsyncCallbackHandler):
    """按 run_name 记录链中每次 LLM 调用的墙上时钟耗时。

    OpenAI 兼容接口不返回生成耗时（只有 Ollama 有 total_duration），所以在链外按阶段计时；
    这样链里就不需要用 RunnableLambda 包住 LLM —— 包住会切断 token 级流式。
    每个请求新建一个实例，通过 config 的 callbacks 传入，因此并发请求之间互不干扰。
    """

    def __init__(self) -> None:
        self._started: Dict[UUID, Tuple[str, float]] = {}
        self.durations: Dict[str, float] = {}

    async def on_chat_model_start(self, serialized: dict, messages: Any, *, run_id: UUID, **kwargs) -> None:
        self._started[run_id] = (kwargs.get("name") or "", time.time())

    async def on_llm_end(self, response: Any, *, run_id: UUID, **kwargs) -> None:
        name, start = self._started.pop(run_id, ("", None))
        if name and start is not None:
            self.durations[name] = time.time() - start

    def duration(self, run_name: str, fallback: float = 0.0) -> float:
        """取某个阶段的耗时；模型自带的耗时（Ollama）优先"""
        return fallback or self.durations.get(run_name, 0.0)
