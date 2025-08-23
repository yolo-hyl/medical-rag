"""
配置加载器
"""
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging
from .models import AppConfig, MilvusConfig, EmbeddingConfig, LLMConfig, SearchConfig, IngestionConfig, AnnotationConfig, RAGConfig
from typing import Dict, List, Optional, Literal, Any, Union
from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)

class ConfigLoader:
    """配置加载器"""
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        初始化配置加载器
        
        Args:
            config_dir: 配置文件目录，默认为项目根目录下的config文件夹
        """
        if config_dir is None:
            # 默认配置目录
            project_root = Path(__file__).parent.parent.parent.parent
            self.config_dir = project_root / "config"
        else:
            self.config_dir = Path(config_dir)
            
        if not self.config_dir.exists():
            raise FileNotFoundError(f"配置目录不存在: {self.config_dir}")
    
    def _load_yaml(self, filename: str) -> Dict[str, Any]:
        """加载YAML文件"""
        file_path = self.config_dir / filename
        if not file_path.exists():
            logger.warning(f"配置文件不存在: {file_path}")
            return {}
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.error(f"加载配置文件失败 {file_path}: {e}")
            return {}
    
    def load_milvus_config(self) -> MilvusConfig:
        """加载Milvus配置"""
        data = self._load_yaml("milvus.yaml")
        return MilvusConfig(**data)
    
    def load_embedding_config(self) -> EmbeddingConfig:
        """加载嵌入配置"""
        data = self._load_yaml("embedding.yaml")
        return EmbeddingConfig(**data)
    
    def load_llm_config(self) -> LLMConfig:
        """加载LLM配置"""
        data = self._load_yaml("llm.yaml")
        return LLMConfig(**data)
    
    def load_search_config(self) -> SearchConfig:
        """加载检索配置"""
        data = self._load_yaml("search.yaml")
        return SearchConfig(**data)
    
    def load_ingestion_config(self) -> IngestionConfig:
        """加载数据入库配置"""
        data = self._load_yaml("ingestion.yaml")
        return IngestionConfig(**data)
    
    def load_annotation_config(self) -> AnnotationConfig:
        """加载标注配置"""
        data = self._load_yaml("annotation.yaml")
        return AnnotationConfig(**data)
    
    def load_rag_config(self) -> RAGConfig:
        """加载RAG配置"""
        data = self._load_yaml("rag.yaml")
        return RAGConfig(**data)
    
    def load_full_config(self) -> AppConfig:
        """加载完整应用配置"""
        return AppConfig(
            milvus=self.load_milvus_config(),
            embedding=self.load_embedding_config(),
            llm=self.load_llm_config(),
            search=self.load_search_config(),
            ingestion=self.load_ingestion_config(),
            annotation=self.load_annotation_config(),
            rag=self.load_rag_config()
        )
    
    def load_config_from_single_file(self, filename: str) -> AppConfig:
        """从单个文件加载完整配置"""
        data = self._load_yaml(filename)
        return AppConfig(**data)

# 便捷函数
def load_config(config_dir: Optional[str] = None) -> AppConfig:
    """加载配置的便捷函数"""
    loader = ConfigLoader(config_dir)
    return loader.load_full_config()

def load_config_from_file(config_file: str) -> AppConfig:
    """从单个文件加载配置"""
    config_path = Path(config_file)
    loader = ConfigLoader(config_path.parent)
    return loader.load_config_from_single_file(config_path.name)