from __future__ import annotations
from typing import Dict, List, Optional, Literal, Any
from pydantic import BaseModel, Field, field_validator
import yaml
from pathlib import Path
import logging
import os
from pydantic import ConfigDict

logger = logging.getLogger(__name__)


# -------------------------
# 基础枚举（用 Literal 保守约束）
# -------------------------
DType = Literal[
    "BOOL","INT8","INT16","INT32","INT64","FLOAT","DOUBLE",
    "VARCHAR","JSON","ARRAY","FLOAT_VECTOR","BINARY_VECTOR","SPARSE_FLOAT_VECTOR"
]
IndexType = Literal["HNSW","IVF_FLAT","IVF_PQ","IVF_SQ8","SPARSE_INVERTED_INDEX"]
MetricType = Literal["L2","IP","COSINE"]
Consistency = Literal["Strong","Bounded","Eventually"]
ChannelKind = Literal["dense_document","dense_query","sparse_document","sparse_query"]
EmbedProvider = Literal["ollama","openai","local"]

# -------------------------
# Schema / Index / Search
# -------------------------
class FieldCfg(BaseModel):
    name: str
    dtype: DType
    # 可选属性（不同类型下是否生效由构建器控制）
    is_primary: bool = False
    is_partition_key: bool = False
    enable_analyzer: bool = False
    max_length: Optional[int] = None
    dim: Optional[int] = None
    element_type: Optional[DType] = None
    max_capacity: Optional[int] = None
    description: str = ""

class IndexPerFieldCfg(BaseModel):
    index_type: IndexType
    metric_type: MetricType
    params: Dict[str, Any] = Field(default_factory=dict)

class IndexCfg(BaseModel):
    build_on_create: bool = True
    by_field: Dict[str, IndexPerFieldCfg] = Field(default_factory=dict)

class SchemaCfg(BaseModel):
    auto_id: bool = False
    enable_dynamic_field: bool = True
    fields: List[FieldCfg]

class CollectionCfg(BaseModel):
    name: str
    description: str = ""
    recreate_if_exists: bool = False
    num_partitions: int = None
    load_on_start: bool = True
    properties: Optional[Dict[str, Any]] = Field(default_factory=dict)

class ClientCfg(BaseModel):
    uri: str
    token: Optional[str] = None
    db_name: str = ""
    timeout_ms: int = 30000
    tls: bool = False
    tls_verify: bool = False

class RrfCfg(BaseModel):
    enabled: bool = True
    k: int = 100

class ChannelCfg(BaseModel):
    name: str
    field: str
    enabled: bool = True
    kind: ChannelKind
    metric_type: MetricType
    limit: int = 10
    params: Dict[str, Any] = Field(default_factory=dict)
    expr: Optional[str] = None
    weight: float = -1.0

class PaginationCfg(BaseModel):
    page_size: int = 10
    max_pages: int = 100

class SearchCfg(BaseModel):
    default_limit: int = 10
    output_fields: List[str] = Field(default_factory=list)
    rrf: RrfCfg = Field(default_factory=RrfCfg)
    channels: List[ChannelCfg]
    pagination: PaginationCfg = Field(default_factory=PaginationCfg)

class CompactionCfg(BaseModel):
    enabled: bool = True
    interval_sec: int = 300
    sealed_segments_only: bool = True
    threshold_deleted_ratio: float = 0.5

class WriteCfg(BaseModel):
    consistency_level: Consistency = "Bounded"
    batch_size: int = 512
    id_field: str = "id"
    partition_key_field: Optional[str] = None
    upsert: bool = True
    auto_load_after_write: bool = True
    flush_interval_ms: int = 0
    compaction: CompactionCfg = Field(default_factory=CompactionCfg)

class DeleteCfg(BaseModel):
    allow_hard_delete: bool = True
    soft_delete_field: str = ""

# -------------------------
# Embedding / Sparse
# -------------------------
class DenseEmbedCfg(BaseModel):
    provider: EmbedProvider = "ollama"
    model: str
    base_url: Optional[str] = None
    dim: int = 1024
    normalize: bool = False
    max_concurrent: int = 8
    proxy: Optional[str] = None
    verify: bool = True
    timeout: int = 60
    max_retries: int = 5
    backoff_base: float = 0.5
    backoff_cap: float = 4.0
    prefixes: Dict[str, str] = Field(default_factory=lambda: {"query": "", "document": ""})
    

class SparseBM25Cfg(BaseModel):
    vocab_path: str
    domain_model: str = "medicine"
    prune_empty_sparse: bool = True
    empty_sparse_fallback: Dict[int, float] = Field(default_factory=lambda: {0: 0.0})
    k1: float = 1.5
    b: float = 0.75

    @field_validator("empty_sparse_fallback", mode="before")
    @classmethod
    def _to_int_keys(cls, v):
        # 允许 YAML 里用字符串 key
        if isinstance(v, dict):
            return {int(k): float(v[k]) for k in v}
        return v

class EmbeddingCfg(BaseModel):
    dense: DenseEmbedCfg
    sparse_bm25: SparseBM25Cfg

# -------------------------
# Ingest / Runtime / Logging
# -------------------------

class RuntimeCfg(BaseModel):
    concurrency: int = 4
    timeout_ms: int = 120000
    retries: int = 2

class LoggingCfg(BaseModel):
    level: Literal["DEBUG","INFO","WARN","ERROR"] = "INFO"

# -------------------------
# 顶层配置
# -------------------------
class MilvusAllCfg(BaseModel):
    client: ClientCfg
    collection: CollectionCfg
    schema_: SchemaCfg = Field(alias="schema")   # ← 字段内部叫 schema_
    index: IndexCfg
    search: SearchCfg
    write: WriteCfg
    delete: DeleteCfg

    # 允许用字段名或别名来构造；序列化时也能用 alias
    model_config = ConfigDict(populate_by_name=True)

class AppCfg(BaseModel):
    app: Dict[str, Any] = Field(default_factory=dict)
    milvus: MilvusAllCfg
    embedding: EmbeddingCfg
    runtime: RuntimeCfg = Field(default_factory=RuntimeCfg)
    logging: LoggingCfg = Field(default_factory=LoggingCfg)

# -------------------------
# 加载器
# -------------------------
import yaml

def is_yaml_file(path):
    # 1. 扩展名检查
    if os.path.splitext(path)[1].lower() not in {".yaml", ".yml"}:
        return False
    
    # 2. 内容解析
    try:
        with open(path, "r", encoding="utf-8") as f:
            yaml.safe_load(f)
        return True
    except yaml.YAMLError:
        return False
    except FileNotFoundError:
        return False

def load_cfg(path: str) -> AppCfg:
    curr_dir = Path(__file__).resolve().parent
    if not (Path(path).exists() and Path(path).is_file() and is_yaml_file(path)):
        path = str(curr_dir) + "/" + "default.yaml"
        logger.warning(f"当前配置文件不存在或者为非法yaml文件，读取默认配置: {path}")
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return AppCfg(**raw)
