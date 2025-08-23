# src/medical_rag/config/models.py
"""
简化的配置数据模型
"""
from typing import Dict, List, Optional, Literal, Any, Union
from pydantic import BaseModel, Field

# =============================================================================
# 多向量字段配置 (新增)
# =============================================================================
class VectorFieldConfig(BaseModel):
    """单个向量字段配置"""
    name: str
    embedding_config: 'DenseConfig'
    index_params: Optional[Dict[str, Any]] = None
    search_params: Optional[Dict[str, Any]] = None

class MultiVectorConfig(BaseModel):
    """多向量字段配置"""
    enabled: bool = True
    
    # 向量字段定义
    question_vector: VectorFieldConfig = Field(
        default_factory=lambda: VectorFieldConfig(
            name="vec_question",
            embedding_config=DenseConfig(
                provider="ollama",
                model="bge-m3:latest",
                dimension=1024
            )
        )
    )
    
    text_vector: VectorFieldConfig = Field(
        default_factory=lambda: VectorFieldConfig(
            name="vec_text", 
            embedding_config=DenseConfig(
                provider="ollama",
                model="bge-m3:latest",
                dimension=1024
            )
        )
    )
    
    # BM25稀疏向量配置
    sparse_vector: Dict[str, Any] = Field(default_factory=lambda: {
        "name": "sparse",
        "use_builtin_bm25": True,  # 使用Milvus内置BM25
        "analyzer_params": {}  # BM25分析器参数
    })
    
    # 重排配置
    reranker: Dict[str, Any] = Field(default_factory=lambda: {
        "type": "weighted",  # weighted 或 rrf
        "params": {
            "weights": [0.4, 0.3, 0.3]  # [question_vec, text_vec, sparse_vec]
        }
    })


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
    workers: int = 8
    chunksize: int = 64

class DenseConfig(BaseModel):
    """稠密向量配置"""
    provider: Literal['openai', 'ollama'] = 'ollama'
    model: str
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    proxy: Optional[str] = None
    dimension: int = 1024

# 更新嵌入配置，支持多向量
class EmbeddingConfig(BaseModel):
    """嵌入配置 - 更新版"""
    # 原有的单一嵌入配置（向后兼容）
    dense: DenseConfig
    sparse: SparseConfig
    
    # 新增的多向量配置
    multi_vector: MultiVectorConfig = Field(default_factory=MultiVectorConfig)

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
    default_source: str = "qa"
    batch_size: int = 100

# =============================================================================
# 检索配置
# =============================================================================
class HybridSearchConfig(BaseModel):
    """混合搜索配置"""
    enabled: bool = True
    
    # 多向量字段权重
    field_weights: Dict[str, float] = Field(default_factory=lambda: {
        "vec_question": 0.4,
        "vec_text": 0.3, 
        "sparse": 0.3
    })
    
    # 重排策略
    ranker_type: Literal["weighted", "rrf"] = "weighted"
    ranker_params: Dict[str, Any] = Field(default_factory=lambda: {
        "weights": [0.4, 0.3, 0.3]
    })
    
    # 检索参数
    top_k_per_field: Optional[List[int]] = None  # 每个字段的top-k
    final_top_k: int = 10  # 最终返回的top-k

class SearchConfig(BaseModel):
    """检索配置 - 更新版"""
    # 原有配置（向后兼容）
    top_k: int = 10
    score_threshold: Optional[float] = None
    rrf_k: int = 100
    output_fields: List[str] = Field(default_factory=lambda: ["question", "answer", "source"])
    filters: Optional[Dict[str, Any]] = None
    
    # 新增混合搜索配置
    hybrid_search: HybridSearchConfig = Field(default_factory=HybridSearchConfig)

# =============================================================================
# 数据入库配置
# =============================================================================
class IngestionConfig(BaseModel):
    """数据入库配置"""
    # 数据处理
    batch_size: int = 100
    max_workers: int = 4
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None
    
    # 重试机制
    max_retries: int = 3
    retry_delay: float = 1.0
    
    # 向量化选项
    build_vocab_if_needed: bool = True
    update_existing: bool = False
    
    # 预处理
    clean_text: bool = True
    remove_duplicates: bool = True
    min_text_length: int = 10
    max_text_length: int = 10000
    
    # 进度保存
    save_progress: bool = True
    progress_file: str = "ingestion_progress.json"

# =============================================================================
# 标注配置  
# =============================================================================
class AnnotationConfig(BaseModel):
    """标注配置"""
    # 并发控制
    max_concurrent: int = 3
    batch_size: int = 10
    
    # 重试机制
    max_retries: int = 3
    retry_delay: float = 2.0
    
    # 模型配置
    model_backend: Literal['ollama', 'openai', 'vllm'] = 'ollama'
    model_base_url: Optional[str] = None
    model_name: str = "qwen3:32b"
    temperature: float = 0.1
    max_tokens: Optional[int] = 2000
    timeout: int = 120
    
    # 标注选项
    departments_enabled: bool = True  # 科室分类
    categories_enabled: bool = True   # 问题类别分类
    confidence_threshold: float = 0.8
    
    # 科室分类配置 (6大科室)
    department_labels: List[str] = Field(default_factory=lambda: [
        "内科", "外科", "妇产儿科", "五官科", "皮肤性病科", "其他科室"
    ])
    
    # 问题类别配置 (8大类别)
    category_labels: List[str] = Field(default_factory=lambda: [
        "疾病诊断", "症状咨询", "治疗方案", "药物咨询", 
        "检查化验", "预防保健", "康复指导", "其他咨询"
    ])
    
    # 结果保存
    save_intermediate: bool = True
    intermediate_save_interval: int = 100
    validate_results: bool = True
    
    # 提示词配置
    use_custom_prompts: bool = False
    custom_prompts_path: Optional[str] = None

# =============================================================================
# RAG配置
# =============================================================================
class RetrievalConfig(BaseModel):
    """检索配置 - 更新版"""
    # 基础检索参数
    top_k: int = 10
    score_threshold: Optional[float] = None
    rerank_top_k: Optional[int] = None
    
    # 混合检索模式
    hybrid_mode: bool = True  # 是否启用混合检索
    
    # 多向量字段配置
    use_question_vector: bool = True   # 是否使用问题向量
    use_text_vector: bool = True       # 是否使用文本向量
    use_sparse_vector: bool = True     # 是否使用稀疏向量
    
    # 动态权重调整
    adaptive_weights: bool = False     # 是否根据查询类型动态调整权重
    query_type_weights: Dict[str, List[float]] = Field(default_factory=lambda: {
        "symptom": [0.5, 0.3, 0.2],      # 症状类查询：更重视问题匹配
        "treatment": [0.3, 0.4, 0.3],    # 治疗类查询：更重视内容匹配
        "prevention": [0.3, 0.4, 0.3],   # 预防类查询：更重视内容匹配
        "default": [0.4, 0.3, 0.3]       # 默认权重
    })
    
    # 过滤器
    enable_filters: bool = True
    default_filters: Optional[Dict[str, Any]] = None
    
    # 重排序
    enable_rerank: bool = True
    rerank_model: Optional[str] = None

class GenerationConfig(BaseModel):
    """生成配置"""  
    # 生成参数
    temperature: float = 0.1
    max_tokens: Optional[int] = 2000
    top_p: float = 0.9
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    
    # 回答格式
    include_sources: bool = True
    max_sources: int = 3
    source_format: Literal['simple', 'detailed'] = 'simple'
    
    # 安全检查
    enable_safety_check: bool = True
    max_response_length: int = 5000

class RAGConfig(BaseModel):
    """RAG配置"""
    # 模式选择
    mode: Literal['basic', 'agent'] = 'basic'
    
    # 检索配置
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    
    # 生成配置
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    
    # 智能体RAG配置 (当mode='agent'时生效)
    agent_config: Optional['AgentConfig'] = None
    
    # 提示词配置
    system_prompt_template: str = "medical_qa_system"
    user_prompt_template: str = "medical_qa_user"
    
    # 对话历史
    enable_chat_history: bool = False
    max_history_length: int = 10
    
    # 缓存配置
    enable_cache: bool = True
    cache_ttl: int = 3600  # 缓存时间(秒)

class AgentConfig(BaseModel):
    """智能体RAG配置"""
    # 工具使用
    enable_web_search: bool = False
    web_search_engine: Literal['brave', 'google', 'bing'] = 'brave'
    web_search_max_results: int = 5
    
    # 计算工具
    enable_calculator: bool = True
    
    # 决策参数
    max_iterations: int = 5
    planning_enabled: bool = True
    
    # 自主检索
    auto_retrieval: bool = True
    adaptive_k: bool = True  # 动态调整检索数量
    
    # 多源信息融合
    multi_source_fusion: bool = True
    source_priority: List[str] = Field(default_factory=lambda: [
        "knowledge_base", "web_search", "calculation"
    ])

# 更新前向引用
RAGConfig.model_rebuild()

# =============================================================================
# 更新主配置类
# =============================================================================
class AppConfig(BaseModel):
    """应用主配置 - 更新版"""
    milvus: MilvusConfig
    embedding: EmbeddingConfig  # 包含multi_vector配置
    llm: LLMConfig
    data: DataConfig
    search: SearchConfig        # 包含hybrid_search配置
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    annotation: AnnotationConfig = Field(default_factory=AnnotationConfig)
    rag: RAGConfig = Field(default_factory=RAGConfig)
    
    
    
def create_multi_vector_config(
    question_model: str = "bge-m3:latest",
    text_model: str = "text-embedding-3-large",
    provider: str = "ollama",
    weights: List[float] = None
) -> MultiVectorConfig:
    """创建多向量配置的便捷函数"""
    if weights is None:
        weights = [0.4, 0.3, 0.3]
    
    return MultiVectorConfig(
        question_vector=VectorFieldConfig(
            name="vec_question",
            embedding_config=DenseConfig(
                provider=provider,
                model=question_model,
                dimension=1024
            )
        ),
        text_vector=VectorFieldConfig(
            name="vec_text",
            embedding_config=DenseConfig(
                provider=provider if provider != "openai" else "openai",
                model=text_model,
                dimension=1024 if provider != "openai" else 3072
            )
        ),
        reranker={
            "type": "weighted",
            "params": {"weights": weights}
        }
    )

def create_hybrid_search_config(
    ranker_type: str = "weighted",
    weights: List[float] = None
) -> HybridSearchConfig:
    """创建混合搜索配置的便捷函数"""
    if weights is None:
        weights = [0.4, 0.3, 0.3]
    
    return HybridSearchConfig(
        ranker_type=ranker_type,
        ranker_params={"weights": weights} if ranker_type == "weighted" else {"k": 60}
    )