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

from ..config.models import AppConfig, LLMConfig, DenseConfig, SparseConfig
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

def create_vector_store(config: AppConfig) -> VectorStore:
    """创建向量存储"""
    # 准备连接参数
    connection_args = {"uri": config.milvus.uri}
    if config.milvus.token:
        connection_args["token"] = config.milvus.token
    
    # 创建稠密嵌入
    dense_embedding = create_embedding_client(config.embedding.dense)
    
    # 根据稀疏向量配置选择方案
    if config.embedding.sparse.manager == "milvus":
        # 使用Milvus内置BM25
        logger.info("使用Milvus内置BM25")
        
        vector_store = Milvus(
            embedding_function=dense_embedding,
            builtin_function=BM25BuiltInFunction(),  # Milvus内置BM25
            collection_name=config.milvus.collection_name,
            connection_args=connection_args,
            drop_old=config.milvus.drop_old,
            auto_id=config.milvus.auto_id,
            # 字段配置
            vector_field="dense_vector",  # 稠密向量字段
            # Milvus会自动为BM25创建sparse字段
        )
        
    else:
        # 使用自管理BM25
        logger.info("使用自管理BM25")
        
        # 创建BM25管理器
        bm25_manager = SimpleBM25Manager(
            vocab_path=config.embedding.sparse.vocab_path,
            domain_model=config.embedding.sparse.domain_model
        )
        
        # 创建稀疏嵌入适配器
        sparse_embedding = BM25SparseEmbedding(bm25_manager)
        
        # 创建支持混合检索的向量存储
        # 注意：这里需要配置支持两个向量字段的schema
        vector_store = Milvus(
            embedding_function=dense_embedding,
            collection_name=config.milvus.collection_name,
            connection_args=connection_args,
            drop_old=config.milvus.drop_old,
            auto_id=config.milvus.auto_id,
            # 字段配置
            vector_field="dense_vector",
            # 需要手动处理稀疏向量字段
        )
        
        # 为向量存储添加BM25管理器引用
        vector_store._bm25_manager = bm25_manager
    
    return vector_store

class KnowledgeBase:
    """知识库管理器"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.vector_store = create_vector_store(config)
        self._llm = None
        
        # 检查是否使用自管理BM25
        self.use_self_bm25 = config.embedding.sparse.manager == "self"
        if self.use_self_bm25:
            self.bm25_manager = getattr(self.vector_store, '_bm25_manager', None)
    
    @property
    def llm(self) -> BaseChatModel:
        """获取LLM客户端"""
        if self._llm is None:
            self._llm = create_llm_client(self.config.llm)
        return self._llm
    
    def build_vocab_if_needed(self, texts: List[str]) -> None:
        """如果使用自管理BM25，构建词表"""
        if self.use_self_bm25 and self.bm25_manager:
            # 检查词表是否已存在且非空
            if not hasattr(self.bm25_manager.vocab, 'token2id') or len(self.bm25_manager.vocab.token2id) == 0:
                logger.info("构建BM25词表...")
                self.bm25_manager.build_vocab_from_texts(texts)
                logger.info("BM25词表构建完成")
            else:
                logger.info("BM25词表已存在，跳过构建")
    
    def add_texts(
        self, 
        texts: List[str], 
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """添加文本到知识库"""
        # 如果使用自管理BM25，需要先构建词表
        if self.use_self_bm25:
            self.build_vocab_if_needed(texts)
        
        # 使用langchain-milvus的标准接口
        return self.vector_store.add_texts(
            texts=texts,
            metadatas=metadatas,
            ids=ids
        )
    
    def add_documents(self, documents) -> List[str]:
        """添加文档到知识库"""
        texts = [doc.page_content for doc in documents]
        
        # 构建词表
        if self.use_self_bm25:
            self.build_vocab_if_needed(texts)
        
        return self.vector_store.add_documents(documents)
    
    def search(
        self,
        query: str,
        k: Optional[int] = None,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Any]:
        """搜索知识库"""
        k = k or self.config.search.top_k
        
        # 使用langchain-milvus的标准搜索
        if self.use_self_bm25:
            # 混合检索：需要同时进行dense和sparse搜索
            # 这里可以使用langchain-milvus的混合检索功能
            return self.vector_store.similarity_search(
                query=query,
                k=k,
                filter=filter,
                **kwargs
            )
        else:
            # Milvus内置BM25，支持自动混合检索
            return self.vector_store.similarity_search(
                query=query,
                k=k,
                filter=filter,
                **kwargs
            )
    
    def as_retriever(self, **kwargs):
        """转换为检索器"""
        search_kwargs = {
            "k": self.config.search.top_k,
        }
        
        if self.config.search.filters:
            search_kwargs["filter"] = self.config.search.filters
        if self.config.search.score_threshold:
            search_kwargs["score_threshold"] = self.config.search.score_threshold
        
        # 合并传入的参数
        search_kwargs.update(kwargs.get("search_kwargs", {}))
        
        return self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs=search_kwargs
        )
    
    def delete(self, ids: Optional[List[str]] = None, **kwargs) -> bool:
        """删除文档"""
        return self.vector_store.delete(ids=ids, **kwargs)