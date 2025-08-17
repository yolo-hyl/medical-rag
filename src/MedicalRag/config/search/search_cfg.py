"""
搜索专用配置类
从完整的milvus配置中提取搜索需要的部分
"""
from __future__ import annotations
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
import yaml
from pathlib import Path
import logging
from ..default_cfg import (
    ClientCfg, CollectionCfg, SearchCfg, EmbeddingCfg,
    ChannelCfg, RrfCfg, PaginationCfg, DenseEmbedCfg, SparseBM25Cfg
)

logger = logging.getLogger(__name__)


class SearchAppCfg(BaseModel):
    """搜索专用的应用配置"""
    milvus: SearchMilvusCfg
    embedding: EmbeddingCfg
    search: SearchCfg

class SearchMilvusCfg(BaseModel):
    """搜索专用的Milvus配置"""
    client: ClientCfg
    collection: SearchCollectionCfg

class SearchCollectionCfg(BaseModel):
    """搜索专用的Collection配置"""
    name: str
    description: str = ""
    load_on_start: bool = True


def load_search_cfg(path: str) -> SearchAppCfg:
    """
    加载搜索配置文件
    
    Args:
        path: 配置文件路径
        
    Returns:
        SearchAppCfg: 搜索配置对象
    """
    curr_dir = Path(__file__).resolve().parent
    
    # 检查文件是否存在
    if not (Path(path).exists() and Path(path).is_file()):
        logger.warning(f"搜索配置文件不存在: {path}，使用默认配置")
        path = str(curr_dir / "search" / "search_answer.yaml")
    
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        
        # 构造SearchAppCfg需要的结构
        search_config = {
            "milvus": {
                "client": raw["milvus"]["client"],
                "collection": {
                    "name": raw["milvus"]["collection"]["name"],
                    "description": raw["milvus"]["collection"].get("description", ""),
                    "load_on_start": raw["milvus"]["collection"].get("load_on_start", True)
                }
            },
            "embedding": raw["embedding"],
            "search": raw["search"]
        }
        
        return SearchAppCfg(**search_config)
        
    except Exception as e:
        logger.error(f"加载搜索配置失败: {e}")
        raise


def create_search_config_from_dict(config_dict: Dict[str, Any]) -> SearchAppCfg:
    """
    从字典创建搜索配置对象
    
    Args:
        config_dict: 配置字典
        
    Returns:
        SearchAppCfg: 搜索配置对象
    """
    try:
        return SearchAppCfg(**config_dict)
    except Exception as e:
        logger.error(f"从字典创建搜索配置失败: {e}")
        raise


def merge_search_config(base_config_path: str, search_config_path: str) -> SearchAppCfg:
    """
    合并基础配置和搜索配置
    从基础配置中提取milvus连接信息，从搜索配置中提取搜索相关配置
    
    Args:
        base_config_path: 基础配置文件路径（如milvus.yaml）
        search_config_path: 搜索配置文件路径
        
    Returns:
        SearchAppCfg: 合并后的搜索配置对象
    """
    try:
        # 加载基础配置
        with open(base_config_path, "r", encoding="utf-8") as f:
            base_config = yaml.safe_load(f)
        
        # 加载搜索配置
        with open(search_config_path, "r", encoding="utf-8") as f:
            search_config = yaml.safe_load(f)
        
        # 合并配置
        merged_config = {
            "milvus": {
                "client": base_config["milvus"]["client"],
                "collection": {
                    "name": base_config["milvus"]["collection"]["name"],
                    "description": base_config["milvus"]["collection"].get("description", ""),
                    "load_on_start": base_config["milvus"]["collection"].get("load_on_start", True)
                }
            },
            "embedding": base_config.get("embedding", search_config.get("embedding", {})),
            "search": search_config.get("search", base_config.get("milvus", {}).get("search", {}))
        }
        
        return SearchAppCfg(**merged_config)
        
    except Exception as e:
        logger.error(f"合并配置失败: {e}")
        raise