"""
RAG 搜索工具
提供简单易用的接口，支持配置文件或配置字典
"""

import logging
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from MedicalRag.pipeline.query.query_pipeline import QueryPipeline
from MedicalRag.config.search.search_cfg import SearchAppCfg

logger = logging.getLogger(__name__)


class RAGSearchTool:
    """
    RAG 搜索工具
    提供简化的搜索接口，支持多种配置方式
    """
    
    def __init__(
        self,
        config_source: Union[str, Path, Dict[str, Any], SearchAppCfg],
        auto_setup: bool = True
    ):
        """
        初始化 RAG 搜索工具
        
        Args:
            config_source: 配置源，可以是：
                - 搜索配置文件路径（str/Path）
                - 配置字典（Dict）
                - 配置对象（SearchAppCfg）
            auto_setup: 是否自动设置Pipeline
        """
        self.pipeline: Optional[QueryPipeline] = None
        self._setup_complete = False
        
        # 根据配置源类型创建Pipeline
        if isinstance(config_source, (str, Path)):
            self.pipeline = QueryPipeline.create_from_search_config(str(config_source))
            logger.info(f"从配置文件创建Pipeline: {config_source}")
        elif isinstance(config_source, dict):
            self.pipeline = QueryPipeline.create_from_config_dict(config_source)
            logger.info("从配置字典创建Pipeline")
        elif isinstance(config_source, SearchAppCfg):
            self.pipeline = QueryPipeline(cfg=config_source)
            logger.info("从配置对象创建Pipeline")
        else:
            raise ValueError(f"不支持的配置源类型: {type(config_source)}")
        
        # 自动设置
        if auto_setup:
            self.setup()
    
    def setup(self) -> bool:
        """
        设置Pipeline
        
        Returns:
            bool: 设置是否成功
        """
        if not self.pipeline:
            logger.error("Pipeline未初始化")
            return False
        
        try:
            success = self.pipeline.setup()
            self._setup_complete = success
            if success:
                logger.info("RAG搜索工具设置完成")
            else:
                logger.error("RAG搜索工具设置失败")
            return success
        except Exception as e:
            logger.error(f"设置RAG搜索工具失败: {e}")
            return False
    
    def search(
        self,
        query: Union[str, List[str]],
        filters: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
        page: int = 1,
        page_size: Optional[int] = None
    ) -> Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]]:
        """
        执行搜索
        
        Args:
            query: 查询文本或查询列表
            filters: 过滤条件字典
            limit: 限制返回数量
            page: 页码
            page_size: 每页大小
            
        Returns:
            搜索结果，单个查询返回List，多个查询返回List[List]
        """
        if not self._setup_complete:
            raise RuntimeError("Pipeline未设置完成，请先调用setup()")
        
        # 处理单个查询
        if isinstance(query, str):
            return self.pipeline.search_single(
                query=query,
                expr_vars=filters,
                limit=limit
            )
        
        # 处理批量查询
        elif isinstance(query, list):
            return self.pipeline.search(
                queries=query,
                expr_vars=filters,
                page=page,
                page_size=page_size,
                limit_override=limit
            )
        
        else:
            raise ValueError(f"不支持的查询类型: {type(query)}")
    
    def get_config_info(self) -> Dict[str, Any]:
        """
        获取配置信息
        
        Returns:
            Dict[str, Any]: 配置信息
        """
        if not self.pipeline:
            return {"error": "Pipeline未初始化"}
        
        try:
            config_info = {
                "collection_name": self.pipeline.collection_name,
                "config_type": self.pipeline._config_type,
                "milvus_uri": self.pipeline.cfg.milvus.client.uri,
                "embedding_provider": self.pipeline.cfg.embedding.dense.provider,
                "embedding_model": self.pipeline.cfg.embedding.dense.model,
                "sparse_vocab": self.pipeline.cfg.embedding.sparse_bm25.vocab_path,
                "setup_complete": self._setup_complete
            }
            
            # 添加搜索配置
            if self._setup_complete:
                config_info["search_config"] = self.pipeline.get_search_config()
            
            return config_info
        except Exception as e:
            logger.error(f"获取配置信息失败: {e}")
            return {"error": str(e)}
    
    def update_search_config(self, updates: Dict[str, Any]) -> bool:
        """
        更新搜索配置
        
        Args:
            updates: 更新字典，支持以下键：
                - channels: 通道更新列表
                - limit: 默认限制数量
                - output_fields: 输出字段列表
                
        Returns:
            bool: 更新是否成功
        """
        if not self._setup_complete:
            logger.error("Pipeline未设置完成")
            return False
        
        try:
            success = True
            
            # 更新通道配置
            if "channels" in updates:
                channel_success = self.pipeline.update_search_channels(updates["channels"])
                success = success and channel_success
            
            # 更新其他配置
            if self.pipeline._config_type == "search":
                search_cfg = self.pipeline.cfg.search
            else:
                search_cfg = self.pipeline.cfg.milvus.search
            
            if "limit" in updates:
                search_cfg.default_limit = updates["limit"]
                logger.info(f"更新默认限制数量: {updates['limit']}")
            
            if "output_fields" in updates:
                search_cfg.output_fields = updates["output_fields"]
                logger.info(f"更新输出字段: {updates['output_fields']}")
            
            if success:
                logger.info("搜索配置更新成功")
            else:
                logger.error("搜索配置更新失败")
            
            return success
        except Exception as e:
            logger.error(f"更新搜索配置失败: {e}")
            return False
    
    def is_ready(self) -> bool:
        """
        检查工具是否准备就绪
        
        Returns:
            bool: 是否准备就绪
        """
        return self._setup_complete and self.pipeline is not None
    
    def close(self):
        """
        关闭工具，清理资源
        """
        # 可以在这里添加资源清理逻辑
        logger.info("RAG搜索工具已关闭")
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.close()


# 便捷创建函数
def create_rag_tool_from_file(config_path: Union[str, Path]) -> RAGSearchTool:
    """
    从配置文件创建RAG搜索工具
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        RAGSearchTool: 配置好的工具实例
    """
    return RAGSearchTool(config_path)


def create_rag_tool_from_dict(config_dict: Dict[str, Any]) -> RAGSearchTool:
    """
    从配置字典创建RAG搜索工具
    
    Args:
        config_dict: 配置字典
        
    Returns:
        RAGSearchTool: 配置好的工具实例
    """
    return RAGSearchTool(config_dict)


# 使用示例
if __name__ == "__main__":
    # 示例1：从配置文件创建
    tool = create_rag_tool_from_file("src/MedicalRag/config/search/search_answer.yaml")
    
    if tool.is_ready():
        # 单个查询
        results = tool.search("梅毒")
        print(f"单个查询结果数量: {len(results)}")
        
        # 批量查询
        batch_results = tool.search(["梅毒", "高血压"])
        print(f"批量查询结果: {[len(r) for r in batch_results]}")
        
        # 带过滤条件的查询
        filtered_results = tool.search("梅毒", filters={"src": "huatuo_qa"})
        print(f"过滤查询结果数量: {len(filtered_results)}")
        
        # 获取配置信息
        config_info = tool.get_config_info()
        print(f"配置信息: {config_info}")
    
    # 示例2：使用上下文管理器
    with create_rag_tool_from_file("config.yaml") as tool:
        if tool.is_ready():
            results = tool.search("查询内容")
            print(results)