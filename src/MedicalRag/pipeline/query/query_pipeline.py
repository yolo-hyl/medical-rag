"""
查询编排（过滤+混合检索+重排+裁剪）
"""

import logging
from typing import List, Dict, Any, Optional
from MedicalRag.config.milvus_cfg import AppCfg, load_cfg
from MedicalRag.core.vectorstore.milvus_client import MilvusConn
from MedicalRag.core.vectorstore.milvus_hybrid import HybridRetriever
from MedicalRag.data.processor.sparse import Vocabulary, BM25Vectorizer
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)


class QueryPipeline:
    """
    查询 Pipeline
    根据配置文件的 search 配置进行混合检索
    """
    
    def __init__(self, config_path: Optional[str] = None, cfg: Optional[AppCfg] = None):
        """
        初始化查询 Pipeline
        
        Args:
            config_path: 配置文件路径
            cfg: 直接传入的配置对象（优先级高于 config_path）
        """
        if cfg is not None:
            self.cfg = cfg
        elif config_path is not None:
            self.cfg = load_cfg(config_path)
        else:
            raise ValueError("必须提供 config_path 或 cfg 参数")
        
        self.conn: Optional[MilvusConn] = None
        self.retriever: Optional[HybridRetriever] = None
        self.embedder = None
        self.vectorizer = None
        self.collection_name = self.cfg.milvus.collection.name
        
    def connect(self) -> bool:
        """
        连接到 Milvus
        
        Returns:
            bool: 连接是否成功
        """
        try:
            self.conn = MilvusConn(self.cfg)
            is_healthy = self.conn.healthy(timeout_sec=10)
            if is_healthy:
                logger.info(f"成功连接到 Milvus: {self.cfg.milvus.client.uri}")
                return True
            else:
                logger.error(f"Milvus 健康检查失败: {self.cfg.milvus.client.uri}")
                return False
        except Exception as e:
            logger.error(f"连接 Milvus 失败: {e}")
            return False
    
    def check_collection_loaded(self) -> bool:
        """
        检查集合是否已加载
        
        Returns:
            bool: 集合是否已加载
        """
        if not self.conn:
            raise RuntimeError("尚未连接到 Milvus，请先调用 connect()")
        
        try:
            # 尝试加载集合
            if self.conn.has_collection():
                self.conn.load_collection()
                logger.info(f"集合 '{self.collection_name}' 已加载")
                return True
            else:
                logger.error(f"集合 '{self.collection_name}' 不存在")
                return False
        except Exception as e:
            logger.error(f"加载集合失败: {e}")
            return False
    
    def init_embedder(self):
        """
        初始化嵌入模型
        """
        try:
            dense_cfg = self.cfg.embedding.dense
            if dense_cfg.provider == "ollama":
                self.embedder = OllamaEmbeddings(
                    model=dense_cfg.model,
                    base_url=dense_cfg.base_url
                )
            elif dense_cfg.provider == "openai":
                self.embedder = OpenAIEmbeddings(model=dense_cfg.model)
            elif dense_cfg.provider == "hf":
                self.embedder = HuggingFaceEmbeddings(model_name=dense_cfg.model)
            else:
                raise ValueError(f"不支持的嵌入提供商: {dense_cfg.provider}")
            
            logger.info(f"成功初始化嵌入模型: {dense_cfg.provider}/{dense_cfg.model}")
        except Exception as e:
            logger.error(f"初始化嵌入模型失败: {e}")
            raise
    
    def init_vectorizer(self):
        """
        初始化稀疏向量化器
        """
        try:
            sparse_cfg = self.cfg.embedding.sparse_bm25
            vocab = Vocabulary.load(sparse_cfg.vocab_path)
            self.vectorizer = BM25Vectorizer(vocab, domain_model=sparse_cfg.domain_model)
            logger.info(f"成功初始化稀疏向量化器: {sparse_cfg.vocab_path}")
        except Exception as e:
            logger.error(f"初始化稀疏向量化器失败: {e}")
            raise
    
    def init_retriever(self):
        """
        初始化混合检索器
        """
        if not self.conn or not self.embedder or not self.vectorizer:
            raise RuntimeError("请先完成连接和模型初始化")
        
        try:
            client = self.conn.get_client()
            self.retriever = HybridRetriever(client, self.cfg, self.embedder, self.vectorizer)
            logger.info("成功初始化混合检索器")
        except Exception as e:
            logger.error(f"初始化混合检索器失败: {e}")
            raise
    
    def setup(self) -> bool:
        """
        设置查询 Pipeline（连接、加载集合、初始化模型）
        
        Returns:
            bool: 设置是否成功
        """
        logger.info("=== 开始设置查询 Pipeline ===")
        
        # 1. 连接 Milvus
        if not self.connect():
            return False
        
        # 2. 检查并加载集合
        if not self.check_collection_loaded():
            return False
        
        # 3. 初始化嵌入模型
        try:
            self.init_embedder()
        except Exception:
            return False
        
        # 4. 初始化稀疏向量化器
        try:
            self.init_vectorizer()
        except Exception:
            return False
        
        # 5. 初始化检索器
        try:
            self.init_retriever()
        except Exception:
            return False
        
        logger.info("=== 查询 Pipeline 设置完成 ===")
        return True
    
    def search(
        self,
        queries: List[str],
        expr_vars: Optional[Dict[str, Any]] = None,
        page: int = 1,
        page_size: Optional[int] = None,
        limit_override: Optional[int] = None
    ) -> List[List[Dict[str, Any]]]:
        """
        执行混合检索
        
        Args:
            queries: 查询文本列表
            expr_vars: 表达式变量（用于过滤条件）
            page: 页码
            page_size: 每页大小
            limit_override: 覆盖默认限制数量
            
        Returns:
            List[List[Dict[str, Any]]]: 每个查询的结果列表
        """
        if not self.retriever:
            raise RuntimeError("检索器未初始化，请先调用 setup()")
        
        try:
            logger.info(f"开始执行检索，查询数量: {len(queries)}")
            
            # 调用混合检索
            results = self.retriever.search(
                queries=queries,
                expr_vars=expr_vars,
                page=page,
                page_size=page_size
            )
            
            # 转换结果格式
            formatted_results = []
            for i, hits in enumerate(results):
                query_results = []
                for hit in hits:
                    result = {
                        "id": hit.id,
                        "distance": hit.distance,
                        "score": 1.0 / (1.0 + hit.distance) if hit.distance >= 0 else hit.distance,
                    }
                    # 添加输出字段
                    for field in self.cfg.milvus.search.output_fields:
                        result[field] = hit.get(field)
                    query_results.append(result)
                formatted_results.append(query_results)
            
            logger.info(f"检索完成，返回结果数量: {[len(r) for r in formatted_results]}")
            return formatted_results
            
        except Exception as e:
            logger.error(f"执行检索失败: {e}")
            raise
    
    def search_single(
        self,
        query: str,
        expr_vars: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        执行单个查询的检索
        
        Args:
            query: 单个查询文本
            expr_vars: 表达式变量
            limit: 限制返回数量
            
        Returns:
            List[Dict[str, Any]]: 查询结果列表
        """
        results = self.search(
            queries=[query],
            expr_vars=expr_vars,
            limit_override=limit
        )
        return results[0] if results else []
    
    def get_search_config(self) -> Dict[str, Any]:
        """
        获取当前搜索配置
        
        Returns:
            Dict[str, Any]: 搜索配置信息
        """
        search_cfg = self.cfg.milvus.search
        return {
            "default_limit": search_cfg.default_limit,
            "output_fields": search_cfg.output_fields,
            "expr_template": search_cfg.expr_template,
            "rrf_enabled": search_cfg.rrf.enabled,
            "rrf_k": search_cfg.rrf.k if search_cfg.rrf.enabled else None,
            "channels": [
                {
                    "name": ch.name,
                    "field": ch.field,
                    "enabled": ch.enabled,
                    "kind": ch.kind,
                    "metric_type": ch.metric_type,
                    "limit": ch.limit,
                    "weight": ch.weight,
                    "expr_template": ch.expr_template
                }
                for ch in search_cfg.channels
            ],
            "pagination": {
                "page_size": search_cfg.pagination.page_size,
                "max_pages": search_cfg.pagination.max_pages
            }
        }
    
    def update_search_channels(self, channel_updates: List[Dict[str, Any]]) -> bool:
        """
        动态更新搜索通道配置
        
        Args:
            channel_updates: 通道更新配置列表
            [{"name": "dense_q", "enabled": True, "weight": 0.5}, ...]
            
        Returns:
            bool: 更新是否成功
        """
        try:
            for update in channel_updates:
                channel_name = update.get("name")
                if not channel_name:
                    continue
                
                # 找到对应的通道
                for channel in self.cfg.milvus.search.channels:
                    if channel.name == channel_name:
                        # 更新通道配置
                        for key, value in update.items():
                            if key != "name" and hasattr(channel, key):
                                setattr(channel, key, value)
                        logger.info(f"更新通道配置: {channel_name}")
                        break
            
            logger.info("搜索通道配置更新完成")
            return True
        except Exception as e:
            logger.error(f"更新搜索通道配置失败: {e}")
            return False