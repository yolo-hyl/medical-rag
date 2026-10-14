"""配置模型

各段配置都是可选的：只用到检索的调用方不需要配置 LLM，只用到入库的调用方不需要配置 Agent。
组件只接收自己用到的那一段（例如 `MedicalHybridKnowledgeBase(cfg.milvus, cfg.embedding)`），
不接收整个 AppConfig。
"""
import os
from typing import Any, Dict, List, Literal, Optional
from urllib.parse import quote

from pydantic import BaseModel, Field


def read_secret(env_name: str) -> str:
    """从环境变量读取密码；deploy/.env 会在 ConfigLoader 初始化时自动加载"""
    value = os.environ.get(env_name)
    if not value:
        raise RuntimeError(f"未找到环境变量 {env_name}，请先执行 deploy/start.sh（会生成 deploy/.env）")
    return value


# =============================================================================
# 存储与中间件（由 deploy/start.sh 启动，密码统一从环境变量读取）
#
# 服务地址都可以不配：没有配置地址时自动降级为嵌入式实现，不需要启动任何服务（D18）。
#   Milvus  uri 为空    → Milvus Lite（本地文件 lite_path）
#   Redis   host 为空   → fakeredis（进程内存）
#   Postgres host 为空  → SQLite（本地文件 sqlite_path）
#   Neo4j   uri 为空    → FalkorDBLite（本地文件 falkordb_path，需要 Python >= 3.12）
# 降级实现只适合单进程的开发、测试和小规模使用，具体的选择逻辑在 storage/backends.py。
# 本地文件路径是相对当前工作目录的。
# =============================================================================
class MilvusConfig(BaseModel):
    uri: Optional[str] = Field(default=None, description="Milvus 服务地址，如 http://localhost:19530；为空时使用 Milvus Lite")
    token: Optional[str] = None
    lite_path: str = Field(default=".medrag/milvus_lite.db", description="未配置 uri 时 Milvus Lite 的数据文件")
    collection_name: str = "medical_knowledge"
    drop_old: bool = False
    auto_id: bool = True

    @property
    def is_lite(self) -> bool:
        return not self.uri


class PostgresConfig(BaseModel):
    host: Optional[str] = Field(default=None, description="PostgreSQL 地址；为空时使用 SQLite")
    port: int = 5432
    user: str = "medrag"
    database: str = "medrag"
    password_env: str = "POSTGRES_PASSWORD"
    sqlite_path: str = Field(default=".medrag/medrag.sqlite3", description="未配置 host 时 SQLite 的数据文件")

    @property
    def is_sqlite(self) -> bool:
        return not self.host

    def dsn(self) -> str:
        if self.is_sqlite:
            raise RuntimeError("未配置 postgres.host，当前使用的是 SQLite，没有 PostgreSQL DSN")
        password = quote(read_secret(self.password_env), safe="")
        return f"postgresql://{self.user}:{password}@{self.host}:{self.port}/{self.database}"


class RedisConfig(BaseModel):
    host: Optional[str] = Field(default=None, description="Redis 地址；为空时使用进程内的 fakeredis")
    port: int = 6379
    db: int = 0
    password_env: str = "REDIS_PASSWORD"
    session_ttl: int = 7 * 24 * 3600  # 会话数据（历史 / 摘要 / Agent 状态）的存活时间，单位秒
    # 会话锁：lock_timeout 要大于一轮对话的最长耗时（Agent 拆子查询时可能几分钟），
    # 设上限只是为了进程崩溃后锁能自动过期；lock_wait 是请求排队等待的上限
    lock_timeout: int = 600
    lock_wait: int = 600

    @property
    def is_fake(self) -> bool:
        return not self.host

    def url(self) -> str:
        if self.is_fake:
            raise RuntimeError("未配置 redis.host，当前使用的是 fakeredis，没有 Redis URL")
        password = quote(read_secret(self.password_env), safe="")
        return f"redis://:{password}@{self.host}:{self.port}/{self.db}"


class Neo4jConfig(BaseModel):
    uri: Optional[str] = Field(default=None, description="Neo4j 地址，如 bolt://127.0.0.1:7687；为空时使用 FalkorDBLite")
    user: str = "neo4j"
    password_env: str = "NEO4J_PASSWORD"
    database: Optional[str] = None
    falkordb_path: str = Field(default=".medrag/falkordb.db", description="未配置 uri 时 FalkorDBLite 的数据文件")
    graph_name: str = Field(default="medrag", description="FalkorDB 的图名（Neo4j 用 database 区分）")

    @property
    def is_lite(self) -> bool:
        return not self.uri

    def password(self) -> str:
        return read_secret(self.password_env)


# =============================================================================
# 模型（D2：统一走 OpenAI 兼容协议，DashScope / vLLM / Ollama 的 /v1 端点都按这个接入）
# =============================================================================
class _OpenAICompatBase(BaseModel):
    provider: Literal["openai"] = "openai"  # 以后接其他模型商时在这里扩展
    model: str
    base_url: Optional[str] = None
    # 本地 vLLM / Ollama 不校验密钥，可以不配；openai SDK 要求非空，届时用占位值
    api_key_env: Optional[str] = None
    proxy: Optional[str] = None
    timeout: float = 120.0
    max_retries: int = 3
    concurrency: int = Field(default=8, ge=1, description="同一客户端同时在途的请求数上限")

    def api_key(self) -> str:
        if not self.api_key_env:
            return "EMPTY"
        value = os.environ.get(self.api_key_env)
        if not value:
            raise RuntimeError(f"未找到环境变量 {self.api_key_env}（{self.model} 的 API 密钥）")
        return value


class LLMConfig(_OpenAICompatBase):
    temperature: float = 0.1
    max_tokens: Optional[int] = None
    # 透传给接口的额外参数，例如 DashScope 的 {"enable_thinking": false}
    extra_body: Dict[str, Any] = Field(default_factory=dict)


class DenseConfig(_OpenAICompatBase):
    """稠密向量模型"""
    dimension: int = 1024
    # 是否在请求里带 dimensions。不支持 Matryoshka 的模型（如 Octen-Embedding-8B）收到这个参数会报错
    send_dimensions: bool = True
    # 单次请求最多几条文本；DashScope text-embedding-v4 上限是 10
    max_batch: int = Field(default=10, ge=1)
    concurrency: int = Field(default=4, ge=1)


class SparseConfig(BaseModel):
    provider: Literal["self", "Milvus"] = "self"
    vocab_path_or_name: str = "vocab.pkl.gz"
    algorithm: str = "BM25"
    domain_model: str = "medicine"
    k1: float = 1.5
    b: float = 0.75
    build: dict = {"workers": 8, "chunksize": 64}


class EmbeddingConfig(BaseModel):
    """v2 的单行多向量布局；M1 改为由 Chunk 生成的 content / summary 两路"""
    summary_dense: DenseConfig
    text_dense: DenseConfig
    text_sparse: SparseConfig = Field(default_factory=SparseConfig)


# =============================================================================
# 数据字段映射（M1 起由各数据集的 loader 取代）
# =============================================================================
class DataConfig(BaseModel):
    summary_field: str = "question"
    document_field: str = "answer"
    source_field: Optional[str] = None
    source_name_field: Optional[str] = None
    lt_doc_id_field: Optional[str] = None
    chunk_id_field: Optional[str] = None
    default_source: Optional[str] = "qa"  # 只支持 qa 和文献 literature
    default_source_name: Optional[str] = "huatuo"
    default_lt_doc_id: Optional[str] = ""
    default_chunk_id: Optional[int] = -1
    # 入库批量：每批的文档一起做 embedding、一次写入 Milvus；concurrency 为同时进行的批次数
    batch_size: int = Field(default=10, ge=1, description="每批入库的文档数")
    concurrency: int = Field(default=1, ge=1, description="同时进行的批次数")


# =============================================================================
# 提示词（版本化管理，见 docs/design/prompt-management.md）
# =============================================================================
class PromptsConfig(BaseModel):
    lang: Literal["zh", "en"] = "zh"
    dirs: List[str] = Field(default_factory=list, description="自定义模板目录，同名提示词优先使用这里的")
    versions: Dict[str, str] = Field(default_factory=dict, description="提示词名 -> 版本或别名；未配置的用 stable")


# =============================================================================
# 多轮对话与 Agent
# =============================================================================
class MemoryConfig(BaseModel):
    """多轮记忆管理配置（MultiDialogueRag 与 Agent 共用，见 apps/memory.py）"""
    estimate_token_fun: str = "avg"  # avg 表示按会话统计的平均值估算，其余为注册的估算函数名
    llm_max_token: int = 1024
    max_token_threshold: float = 1.1
    cut_dialogue_scale: int = Field(default=2, ge=2, description="裁切一次砍一半，必须>=2")
    console_debug: bool = False


class MultiDialogueRagConfig(MemoryConfig):
    smith_debug: bool = False
    thinking_in_context: bool = False


class AgentConfig(BaseModel):
    # analysis：拆解子目标分别检索，并校验回答是否忠于事实，不符合就重写
    # normal：不拆分子目标，只改写查询后检索
    # fast：改写查询、检索后直接返回，不校验事实
    mode: Literal["analysis", "fast", "normal"] = "analysis"
    max_attempts: int = 3            # 事实校验不通过时的最大重试次数
    network_search_enabled: bool = True
    network_search_cnt: int = 10     # 联网搜索返回的条数
    db_search_top_k: int = Field(default=5, ge=1, le=10, description="本地检索工具单次返回的条数上限")
    console_debug: bool = False
    max_ask_num: int = 3             # 一个问题最多主动追问的次数
    memory: MemoryConfig = Field(default_factory=MemoryConfig)


# =============================================================================
# 主配置：每一段都可以缺省
# =============================================================================
class AppConfig(BaseModel):
    milvus: MilvusConfig = Field(default_factory=MilvusConfig)
    postgres: PostgresConfig = Field(default_factory=PostgresConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    neo4j: Neo4jConfig = Field(default_factory=Neo4jConfig)
    embedding: Optional[EmbeddingConfig] = None
    llm: Optional[LLMConfig] = None
    data: DataConfig = Field(default_factory=DataConfig)
    prompts: PromptsConfig = Field(default_factory=PromptsConfig)
    multi_dialogue_rag: MultiDialogueRagConfig = Field(default_factory=MultiDialogueRagConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)


# =============================================================================
# 检索请求（M0 沿用 v2 的 Milvus 检索计划；M1 起对外只暴露 QueryParam，这些类型收进 storage/milvus.py）
# =============================================================================
AnnsField = Literal["summary_dense", "text_dense", "text_sparse"]

OutputFields = Literal[
    "pk", "text",
    "summary", "document",
    "source", "source_name",
    "lt_doc_id", "chunk_id",
    "summary_dense", "text_dense", "text_sparse"
]


class FusionSpec(BaseModel):
    method: Literal["rrf", "weighted"] = Field("rrf", description="向量融合策略")
    k: Optional[int] = Field(default=60, gt=0, le=200, description="rrf 的 k 值")
    weights: Optional[List] = Field([0.3, 0.4, 0.3], description="weighted 融合时各路的权重")


class SingleSearchRequest(BaseModel):
    anns_field: AnnsField = Field("summary_dense", description="向量检索字段")
    # BM25：稀疏向量交给 Milvus 托管（text_sparse.provider=Milvus）时必须用它
    metric_type: Literal["COSINE", "IP", "BM25"] = Field("COSINE", description="距离指标")
    search_params: dict = Field({"ef": 64}, description="稀疏检索用 drop_ratio_search，稠密检索用 ef")
    limit: int = Field(default=50, gt=0, le=500, description="这一路返回的条数")
    expr: Optional[str] = Field("", description="Milvus 过滤表达式")


class SearchRequest(BaseModel):
    query: str = Field("", description="查询文本")
    collection_name: str = Field(default="medical_knowledge", description="查询的 collection")
    requests: List[SingleSearchRequest] = Field(default_factory=lambda: [SingleSearchRequest()], description="多路检索配置")
    output_fields: List[OutputFields] = Field(default_factory=lambda: ["text", "summary", "document"], description="输出字段")
    fuse: Optional[FusionSpec] = Field(default_factory=FusionSpec, description="融合策略")
    limit: int = Field(default=5, gt=0, le=10, description="融合排序后最终返回的条数")
