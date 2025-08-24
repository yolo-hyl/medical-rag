# src/medical_rag/core/components.py
"""
核心组件封装，使用langchain-milvus
"""
import logging
from typing import List, Dict, Any, Optional, Union
from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings  
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_milvus import Milvus, BM25BuiltInFunction

from ..config.models import AppConfig, LLMConfig, DenseConfig
from ..knowledge.bm25 import SimpleBM25Manager, BM25SparseEmbedding

logger = logging.getLogger(__name__)

def create_llm_client(config: LLMConfig) -> BaseChatModel:
    """创建LLM客户端"""
    if config.provider == "openai":
        kwargs = {
            "model": config.model,
            "temperature": config.temperature,
        }
        
        if config.api_key:
            kwargs["api_key"] = config.api_key
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
        
        if config.api_key:
            kwargs["api_key"] = config.api_key
        if config.base_url:
            kwargs["base_url"] = config.base_url
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