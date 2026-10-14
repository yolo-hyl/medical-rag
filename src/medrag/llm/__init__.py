"""模型层。langchain_adapter 不在这里导出：它只给 apps/ 和评测用，核心层不应依赖 LangChain"""
from .openai_compat import LLMDelta, LLMResult, Message, OpenAICompatEmbedding, OpenAICompatLLM, Usage
from .utils import split_think, strip_think

__all__ = [
    "LLMDelta",
    "LLMResult",
    "Message",
    "OpenAICompatEmbedding",
    "OpenAICompatLLM",
    "Usage",
    "split_think",
    "strip_think",
]
