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
    default_source: str = "qa"
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
    """检索配置"""
    # 检索参数
    top_k: int = 10
    score_threshold: Optional[float] = None
    rerank_top_k: Optional[int] = None
    
    # 混合检索权重
    dense_weight: float = 0.5
    sparse_weight: float = 0.5
    
    # 过滤器
    enable_filters: bool = True
    default_filters: Optional[Dict[str, Any]] = None
    
    # 重排序
    enable_rerank: bool = False
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
    """应用主配置"""
    milvus: MilvusConfig
    embedding: EmbeddingConfig
    llm: LLMConfig
    data: DataConfig
    search: SearchConfig
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    annotation: AnnotationConfig = Field(default_factory=AnnotationConfig)  
    rag: RAGConfig = Field(default_factory=RAGConfig)