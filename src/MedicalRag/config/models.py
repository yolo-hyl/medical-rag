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
class DenseConfig(BaseModel):
    """稠密向量配置"""
    provider: Literal['openai', 'ollama'] = 'ollama'
    model: str
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    proxy: Optional[str] = None
    dimension: int = 1024

class SparseConfig(BaseModel):
    provider: Literal['self', 'Milvus'] = 'self'
    vocab_path_or_name: str = "vocab.pkl.gz"
    algorithm: str = "BM25"
    domain_model: str = "medicine"
    k1: float = 1.5
    b: float = 0.75
    build: dict = {"workers": 8, "chunksize": 64}

# 更新嵌入配置，支持多向量
class EmbeddingConfig(BaseModel):
    summary_dense: DenseConfig
    text_dense: DenseConfig
    text_sparse: SparseConfig

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
    # 字段映射
    summary_field: str = "question"
    document_field: str = "answer"
    source_field: Optional[str] = None
    source_name_field: Optional[str] = None
    ### 文档独有 ###
    lt_doc_id_field: Optional[str] = None
    chunk_id_field: Optional[int] = None
    ### 文档独有 ###
    default_source: Optional[str] = "qa"  # 只支持QA和文献literature
    default_source_name: Optional[str] = "huatuo"  # QA数据源名称
    default_lt_doc_id: Optional[str] = ""
    default_chunk_id: Optional[int] = -1

# =============================================================================
# 更新主配置类
# =============================================================================
class AppConfig(BaseModel):
    """应用主配置 - 更新版"""
    milvus: MilvusConfig
    embedding: EmbeddingConfig  # 包含multi_vector配置
    llm: LLMConfig
    data: DataConfig
    
# =============================================================================
# 检索时需要传入的数据模型
# =============================================================================

AnnsField = Literal[
    "summary_dense", "text_dense", "text_sparse"
]

OutputFields = Literal[
    "pk", "text", 
    "summary", "document", 
    "source", "source_name", 
    "lt_doc_id", "chunk_id", 
    "summary_dense", "text_dense", "text_sparse"
]

class FusionSpec(BaseModel):
    method: Literal["rrf","weighted"] = "rrf"
    k: Optional[int] = Field(default=60, gt=0, le=200)  # RRF常用k=60
    weights: Optional[List] = [0.3, 0.4, 0.3]

class SingleSearchRequest(BaseModel):
    anns_field: AnnsField = "summary_dense"
    metric_type: Literal["COSINE","IP"] = "COSINE"
    search_params: dict = {"ef": 64}
    limit: int = Field(default=50, gt=0, le=500)
    expr: Optional[str] = "" 

class SearchRequest(BaseModel):
    query: str
    collection_name: str
    requests: List[SingleSearchRequest] = Field(default_factory=lambda: [SingleSearchRequest])
    output_fields: List[OutputFields] = Field(default_factory=lambda: ["text","summary","document"])
    fuse: Optional[FusionSpec] = Field(default_factory=FusionSpec)
    limit: int = Field(default=10, gt=0, le=500)