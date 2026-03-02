"""
工具类，创建合适的llm和embedding客户端
"""
import logging
from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings  
from langchain_ollama import ChatOllama, OllamaEmbeddings
from ..config.models import LLMConfig, DenseConfig
import os

logger = logging.getLogger(__name__)

def create_llm_client(config: LLMConfig) -> BaseChatModel:
    """创建LLM客户端"""
    if config.provider == "openai":
        kwargs = {
            "model": config.model,
            "temperature": config.temperature,
        }
        
        if config.env_key_name:
            kwargs["api_key"] = os.environ[config.env_key_name]
        if config.base_url:
            kwargs["base_url"] = config.base_url
        if config.max_tokens:
            kwargs["max_tokens"] = config.max_tokens
        if config.proxy:
            kwargs["http_client"] = {"proxies": {"http": config.proxy, "https": config.proxy}}
        
        return ChatOpenAI(**kwargs)
        
    elif config.provider == "ollama":
        kwargs = {
            "model": config.model,
            "temperature": config.temperature,
        }
        
        if config.base_url:
            kwargs["base_url"] = config.base_url
        if config.max_tokens:
            kwargs["num_predict"] = config.max_tokens
        
        return ChatOllama(**kwargs)
    
    else:
        raise ValueError(f"不支持的LLM提供商: {config.provider}")

def create_embedding_client(config: DenseConfig) -> Embeddings:
    """创建嵌入客户端"""
    if config.provider == "openai":
        kwargs = {
            "model": config.model,
            "dimensions": config.dimension
        }
        
        if config.env_key_name:
            kwargs["api_key"] = os.environ[config.env_key_name]
        if config.base_url:
            kwargs["base_url"] = config.base_url
            # 对 OpenAI 兼容网关（如 DashScope）避免发送 token id 列表，直接发送字符串文本。
            if "api.openai.com" not in config.base_url:
                kwargs["check_embedding_ctx_length"] = False
        if config.proxy:
            kwargs["http_client"] = {"proxies": {"http": config.proxy, "https": config.proxy}}
        
        return OpenAIEmbeddings(**kwargs)
        
    elif config.provider == "ollama":
        kwargs = {"model": config.model}
        if config.base_url:
            kwargs["base_url"] = config.base_url
        
        return OllamaEmbeddings(**kwargs)
    
    else:
        raise ValueError(f"不支持的嵌入提供商: {config.provider}")
