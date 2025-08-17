"""
初始化全流程编排（Loader->Processor->Annotator->Validator->Embed->Insert）
"""

import logging
from typing import Optional
from pathlib import Path
from MedicalRag.config.default_cfg import AppCfg, load_cfg
from MedicalRag.core.vectorstore.milvus_client import MilvusConn
from MedicalRag.core.vectorstore.milvus_schema import ensure_collection
from MedicalRag.core.vectorstore.milvus_index import build_index_params

logger = logging.getLogger(__name__)


class CollectionCreationPipeline:
    """
    集合创建 Pipeline
    根据配置文件中的 client、collection、schema、index 配置创建 Milvus 集合
    """
    
    def __init__(self, config_path: Optional[str] = None, cfg: Optional[AppCfg] = None):
        """
        初始化集合创建 Pipeline
        
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
    
    def check_collection_exists(self) -> bool:
        """
        检查集合是否已存在
        
        Returns:
            bool: 集合是否存在
        """
        if not self.conn:
            raise RuntimeError("尚未连接到 Milvus，请先调用 connect()")
        
        exists = self.conn.has_collection()
        if exists:
            logger.info(f"集合 '{self.collection_name}' 已存在")
        else:
            logger.info(f"集合 '{self.collection_name}' 不存在")
        return exists
    
    def drop_collection_if_exists(self) -> bool:
        """
        如果配置要求，删除已存在的集合
        
        Returns:
            bool: 操作是否成功
        """
        if not self.conn:
            raise RuntimeError("尚未连接到 Milvus，请先调用 connect()")
        
        try:
            if self.cfg.milvus.collection.recreate_if_exists and self.conn.has_collection():
                logger.warning(f"配置要求重建集合，正在删除现有集合 '{self.collection_name}'")
                self.conn.drop_collection()
                logger.info(f"成功删除集合 '{self.collection_name}'")
            return True
        except Exception as e:
            logger.error(f"删除集合失败: {e}")
            return False
    
    def create_collection(self) -> bool:
        """
        创建集合和索引
        
        Returns:
            bool: 创建是否成功
        """
        if not self.conn:
            raise RuntimeError("尚未连接到 Milvus，请先调用 connect()")
        
        try:
            client = self.conn.get_client()
            
            # 构建索引参数
            index_params = build_index_params(client, self.cfg)
            logger.info("索引参数构建完成")
            
            # 确保集合存在（包含 schema 和 index 创建）
            ensure_collection(client, self.cfg, index_params)
            logger.info(f"成功创建集合 '{self.collection_name}'")
            
            return True
        except Exception as e:
            logger.error(f"创建集合失败: {e}")
            return False
    
    def load_collection(self) -> bool:
        """
        加载集合到内存
        
        Returns:
            bool: 加载是否成功
        """
        if not self.conn:
            raise RuntimeError("尚未连接到 Milvus，请先调用 connect()")
        
        try:
            if self.cfg.milvus.collection.load_on_start:
                self.conn.load_collection()
                logger.info(f"成功加载集合 '{self.collection_name}' 到内存")
            return True
        except Exception as e:
            logger.error(f"加载集合失败: {e}")
            return False
    
    def run(self) -> bool:
        """
        运行完整的集合创建流程
        
        Returns:
            bool: 整个流程是否成功
        """
        logger.info("=== 开始集合创建 Pipeline ===")
        
        # 1. 连接 Milvus
        if not self.connect():
            return False
        
        # 2. 检查并处理现有集合
        if not self.drop_collection_if_exists():
            return False
        
        # 3. 创建集合（如果不存在）
        if not self.check_collection_exists():
            if not self.create_collection():
                return False
        else:
            logger.info("集合已存在，跳过创建")
        
        # 4. 加载集合
        if not self.load_collection():
            return False
        
        logger.info("=== 集合创建 Pipeline 完成 ===")
        return True
    
    def get_collection_info(self) -> dict:
        """
        获取集合信息
        
        Returns:
            dict: 集合信息
        """
        if not self.conn:
            raise RuntimeError("尚未连接到 Milvus，请先调用 connect()")
        
        try:
            client = self.conn.get_client()
            exists = self.conn.has_collection()
            
            info = {
                "collection_name": self.collection_name,
                "exists": exists,
                "milvus_uri": self.cfg.milvus.client.uri,
            }
            
            if exists:
                # 可以添加更多集合统计信息
                stats = client.get_collection_stats(collection_name=self.collection_name)
                info["stats"] = stats
            
            return info
        except Exception as e:
            logger.error(f"获取集合信息失败: {e}")
            return {"error": str(e)}