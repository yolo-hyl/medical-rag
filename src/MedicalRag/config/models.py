# src/medical_rag/config/models.py
"""
简化的配置数据模型
"""
from typing import Dict, List, Optional, Literal, Any, Union
from pydantic import BaseModel, Field

# =============================================================================
# Milvus 配置
# =============================================================================
class MilvusConfig(BaseModel):
    """Milvus配置"""
    uri: str = "http://localhost:19530"
    token: Optional[str] = None
    collection_name: str = "medical_knowledge"
    drop_old: bool = False
    auto_id: bool = True

# =============================================================================
# 嵌入配置
# =============================================================================
class SparseConfig(BaseModel):
    """稀疏向量配置"""
    # 'self' 自管理BM25, 'milvus' 使用Milvus内置BM25
    manager: Literal['self', 'milvus'] = 'self'
    # 自管理BM25配置
    vocab_path: Optional[str] = "vocab.pkl.gz"
    domain_model: str = "medicine"
    k1: float = 1.5
    b: float = 0.75

class DenseConfig(BaseModel):
    """稠密向量配置"""
    provider: Literal['openai', 'ollama'] = 'ollama'
    model: str
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    proxy: Optional[str] = None
    dimension: int = 1024

class EmbeddingConfig(BaseModel):
    """嵌入配置"""
    dense: DenseConfig
    sparse: SparseConfig

# =============================================================================
# LLM 配置
# =============================================================================
class LLMConfig(BaseModel):
    """LLM配置"""
    provider: Literal['openai', 'ollama'] = 'ollama'
    model: str
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    proxy: Optional[str] = None
    temperature: float = 0.1
    max_tokens: Optional[int] = None

# =============================================================================
# 数据配置
# =============================================================================
class DataConfig(BaseModel):
    """数据配置"""
    path: Union[str, List[str]]
    format: Literal['json', 'jsonl', 'parquet'] = 'jsonl'
    # 字段映射
    question_field: str = "question"
    answer_field: str = "answer"
    id_field: Optional[str] = None
    source_field: Optional[str] = None
    default_source: str = "unknown"
    batch_size: int = 100

# =============================================================================
# 检索配置
# =============================================================================
class SearchConfig(BaseModel):
    """检索配置"""
    top_k: int = 10
    score_threshold: Optional[float] = None
    # 混合检索权重 (RRF参数)
    rrf_k: int = 100
    # 输出字段
    output_fields: List[str] = Field(default_factory=lambda: ["question", "answer", "source"])
    # 默认过滤器
    filters: Optional[Dict[str, Any]] = None

# =============================================================================
# 主配置
# =============================================================================
class AppConfig(BaseModel):
    """应用主配置"""
    milvus: MilvusConfig
    embedding: EmbeddingConfig
    llm: LLMConfig
    data: DataConfig
    search: SearchConfig