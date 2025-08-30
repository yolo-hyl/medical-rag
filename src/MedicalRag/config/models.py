from typing import Dict, List, Optional, Literal, Any, Union
from pydantic import BaseModel, Field
from pydantic import BaseModel, Field, field_validator

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
# 多轮RAG对话配置
# =============================================================================
class MultiDialogueRagConfig(BaseModel):
    """ 多轮对话关键配置 """
    estimate_token_fun: str = "avg"
    llm_max_token: int = 1024
    max_token_threshold: float = 1.1   # 宽松阈值
    cut_dialogue_scale: int = Field(default=2, ge=2, description="裁切一次砍一半，必须>=2")
    smith_debug: bool = False
    console_debug: bool = False
    thinking_in_context: bool = False

# =============================================================================
# Agent对话配置
# =============================================================================
class AgentConfig(BaseModel):
    """ 多轮对话关键配置 """
    # analysis 模式会拆解子目标分开多次检索，并验证是否符合事实,不符合事实需要重写检索
    # normal 模式下不会拆分子目标,只会重写查询后进行检索
    # fast 模式下重写查询检索后即返回,不进行验证事实
    mode: Literal["analysis", "fast", "normal"] = "analysis"
    max_attempts: int = 3  # 重复验证事实最大次数
    network_search_enabled: bool = True  # 是否启用联网搜索
    network_search_cnt: int = 10  # 开启联网搜索时，返回的数量
    auto_search_param: bool = True  # 是否开启确定搜索参数
    


# =============================================================================
# 更新主配置类
# =============================================================================
class AppConfig(BaseModel):
    """应用主配置 - 更新版"""
    milvus: MilvusConfig
    embedding: EmbeddingConfig  # 包含multi_vector配置
    llm: LLMConfig
    data: DataConfig
    multi_dialogue_rag: MultiDialogueRagConfig
    agent: AgentConfig
    
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
    method: Literal["rrf","weighted"] = Field("rrf", description="向量融合策略")
    k: Optional[int] = Field(default=60, gt=0, le=200, description="如果使用rrf融合策略,那么这个k值会影响结果")  # RRF常用k=60
    weights: Optional[List] = Field([0.3, 0.4, 0.3], description="如果使用weighted融合策略,那么这个weights会影响结果")

class SingleSearchRequest(BaseModel):
    anns_field: AnnsField = Field("summary_dense", description="向量检索字段")
    metric_type: Literal["COSINE","IP"] = Field("COSINE", description="向量距离计算指标,除了稀疏向量,其余都用'COSINE'")
    search_params: dict = Field({"ef": 64}, description="如果是稀疏向量检索,那么应该指定drop_ratio_search,值为float,例如0.0,否则指定参数ef,值为int")
    limit: int = Field(default=50, gt=0, le=500, description="限制这个向量检索字段返回的多少条数据")
    expr: Optional[str] = Field("", description="过滤不符合这个表达式的数据,例如当需要筛选数据源时,填入:'source == qa',一般不需要更改,除非用户指定")

class SearchRequest(BaseModel):
    query: str = Field("", description="查询文本")
    collection_name: str = Field(default="medical_knowledge", description="查询的collection,默认为'medical_knowledge'")
    requests: List[SingleSearchRequest] = Field(default_factory=lambda: [SingleSearchRequest()], description="多路向量查询的检索配置")
    output_fields: List[OutputFields] = Field(default_factory=lambda: ["text","summary","document"], description="最后输出的参考文档字段;text是由summary和document组合而来")
    fuse: Optional[FusionSpec] = Field(default_factory=FusionSpec, description="向量融合策略")
    limit: int = Field(default=5, gt=0, le=10, description="经过融合排序之后,最终返回的数据量大小,请不要大于10篇")