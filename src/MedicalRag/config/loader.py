"""
配置加载器
"""
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging
from .models import AgentConfig, AppConfig, MilvusConfig, EmbeddingConfig, DataConfig, LLMConfig, SearchConfig, AnnotationConfig, RAGConfig
from typing import Dict, List, Optional, Literal, Any, Union
from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)


def _load_yaml(file_path: str) -> Dict[str, Any]:
    """加载YAML文件"""
    if not Path(file_path).exists():
        logger.warning(f"配置文件不存在: {file_path}")
        return {}
        
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        logger.error(f"加载配置文件失败 {file_path}: {e}")
        return {}
        
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
        
    @staticmethod
    def _load_yaml(self, file_path: str) -> Dict[str, Any]:
        """加载YAML文件"""
        if not Path(file_path).exists():
            logger.warning(f"配置文件不存在: {file_path}")
            return {}
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.error(f"加载配置文件失败 {file_path}: {e}")
            return {}
    
    @staticmethod
    def load_milvus_config(file_path: str) -> MilvusConfig:
        """加载Milvus配置"""
        data = _load_yaml(file_path)
        return MilvusConfig(**data)
    
    @staticmethod
    def load_embedding_config(file_path: str) -> EmbeddingConfig:
        """加载嵌入配置"""
        data = _load_yaml(file_path)
        return EmbeddingConfig(
            summary_dense=data["embedding"]["summary_dense"],
            text_dense=data["embedding"]["text_dense"]
        )
    
    @staticmethod
    def load_llm_config(file_path: str) -> LLMConfig:
        """加载LLM配置"""
        data = _load_yaml(file_path)
        return LLMConfig(**data)
    
    @staticmethod
    def load_data_config(file_path: str) -> DataConfig:
        data = _load_yaml(file_path)
        return DataConfig(**data)
    
    @staticmethod
    def load_search_config(file_path: str) -> SearchConfig:
        """加载检索配置"""
        data = _load_yaml(file_path)
        return SearchConfig(**data)
    
    @staticmethod
    def load_annotation_config(file_path: str) -> AnnotationConfig:
        """加载标注配置"""
        data = _load_yaml(file_path)
        return AnnotationConfig(**data)
    
    @staticmethod
    def load_rag_config(file_path: str) -> RAGConfig:
        """加载RAG配置"""
        data = _load_yaml(file_path)
        return RAGConfig(**data)
    
    @staticmethod
    def load_rag_config(file_path: str) -> AgentConfig:
        data = _load_yaml(file_path)
        return AgentConfig(**data)
    
    @staticmethod
    def load_config_from_single_file(file_path: str) -> AppConfig:
        """从单个文件加载完整配置"""
        data = _load_yaml(file_path)
        return AppConfig(**data)


def load_config_from_file(config_file: str) -> AppConfig:
    """从单个文件加载配置"""
    config_path = Path(config_file)
    loader = ConfigLoader(config_path.parent)
    return loader.load_config_from_single_file(config_path.name)