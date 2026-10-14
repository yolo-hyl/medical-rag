"""只供 apps/ 与 eval/scorers 使用：用同一份配置构造 LangChain 的模型对象

apps/ 里的 LangGraph 图和 RunnableWithMessageHistory 需要 LangChain 的 ChatModel。
这里用与核心层 OpenAICompatLLM 相同的 LLMConfig 构造 ChatOpenAI，保证两边连的是同一个模型。
核心层（apps/ 以外）不要 import 本模块。
"""
import httpx
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from ..config.models import DenseConfig, LLMConfig


def create_chat_model(config: LLMConfig) -> BaseChatModel:
    kwargs = {
        "model": config.model,
        "temperature": config.temperature,
        "api_key": config.api_key(),
        "base_url": config.base_url,
        "timeout": config.timeout,
        "max_retries": config.max_retries,
        # 流式调用时也要求返回 usage，否则摘要压缩依赖的 token 统计会缺失
        "stream_usage": True,
    }
    if config.max_tokens:
        kwargs["max_tokens"] = config.max_tokens
    if config.extra_body:
        kwargs["extra_body"] = dict(config.extra_body)
    if config.proxy:
        # 同步和异步请求各自需要一个客户端，异步调用走 http_async_client
        kwargs["http_client"] = httpx.Client(proxy=config.proxy)
        kwargs["http_async_client"] = httpx.AsyncClient(proxy=config.proxy)
    return ChatOpenAI(**kwargs)


def create_embeddings(config: DenseConfig) -> Embeddings:
    """评测（ragas）需要 LangChain 的 Embeddings；核心层的向量化用 OpenAICompatEmbedding"""
    kwargs = {
        "model": config.model,
        "api_key": config.api_key(),
        "base_url": config.base_url,
        "chunk_size": config.max_batch,
        # OpenAI 兼容网关（如 DashScope）只接受字符串输入，不接受 token id 列表
        "check_embedding_ctx_length": False,
    }
    if config.send_dimensions:
        kwargs["dimensions"] = config.dimension
    if config.proxy:
        kwargs["http_client"] = httpx.Client(proxy=config.proxy)
        kwargs["http_async_client"] = httpx.AsyncClient(proxy=config.proxy)
    return OpenAIEmbeddings(**kwargs)
