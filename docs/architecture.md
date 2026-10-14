# 架构设计（v2.0.0-rebuild）

> 本文定义重构后的目标架构：分层、模块边界、核心接口、数据流和扩展点。
> 现状说明：仓库里的代码仍然是 v2 旧结构（`src/MedicalRag/`），按 [第一阶段规划](phase1/plan.md) 的 M0 逐步迁移到本文描述的 `src/medrag/`。
> 方向性的取舍记录在 [decisions.md](decisions.md)。本文中标注"待确认"的地方，以 decisions.md 的状态为准。

---

## 1. 定位

**专攻医疗领域的通用 RAG 库。** 在抽象方式上对齐 LightRAG：存储可替换、LLM 和 Embedding 以函数形式注入、门面类统一 `ainsert / aquery`。差异化的投入集中在医疗层：

| 阶段 | 内容 |
| --- | --- |
| 第一阶段（本次） | 统一 Schema、存储抽象、增量入库、OpenAI 兼容的模型层、版本化提示词、可复现评测 |
| 第二阶段 | 医疗知识图谱：实体关系抽取，存入 Neo4j；查询模式 local / global / mix |
| 第三阶段 | 本体对齐（以 ICD / ATC 等概念 ID 合并实体）、断言检测（否定 / 不确定 / 家族史） |
| 第四阶段 | 安全层：引用校验、剂量计算工具、PHI 脱敏、急症分诊 |

包本身只提供 RAG 能力，不包含 HTTP 服务。`software/` 是基于这个包写的示例应用。

---

## 2. 分层

```
┌──────────────────────────────────────────────────────────────┐
│ apps/        SimpleRAG · DialogueRAG · MedicalAgent（LangGraph） │  应用层：可以依赖 LangChain / LangGraph
├──────────────────────────────────────────────────────────────┤
│ medrag.py    MedRAG 门面：ainsert / adelete / aretrieve / aquery │  门面层
├───────────────┬───────────────┬──────────────┬───────────────┤
│ ingest/       │ retrieval/    │ prompts/     │ eval/         │  能力层
│ 切块 · 入库流程 │ 稀疏 · 融合 · 重排│ 版本化提示词   │ 指标 · 运行记录 │
├───────────────┴───────────────┴──────────────┴───────────────┤
│ storage/      向量库 · 文档状态 · 会话                          │  基础层
│ llm/          OpenAI 兼容的 LLM / Embedding 客户端             │
│ types.py · base.py · config/                                  │
└──────────────────────────────────────────────────────────────┘
datasets/   数据集下载、清洗、子集、loader（评测和示例使用，核心流程不依赖它）
```

**依赖规则**（M0 之后由测试检查 import 关系）：

1. 依赖方向只能是上层 → 下层，同层之间不能循环依赖。
2. `apps/` 之外的模块**不 import LangChain / LangGraph**。LLM 统一通过 `llm/` 的 `openai` SDK 访问。
3. 组件不自己创建依赖，存储、模型客户端、提示词注册表都由调用方注入；组件只接收自己用到的那一段配置。
4. 所有 IO 方法都是 `async`，不保留同步版本。CPU 密集的操作（如 pkuseg 分词）用 `asyncio.to_thread` 放到线程里执行。

---

## 3. 目录结构

```
src/medrag/                          # 包名：D1（待确认）
├── __init__.py                      # 导出 MedRAG、QueryParam、Document 等公共 API
├── medrag.py                        # 门面类
├── types.py                         # 数据模型（Pydantic）
├── base.py                          # 抽象接口
├── config/
│   ├── models.py                    # 各段配置模型，全部可选
│   └── loader.py                    # 多 YAML 合并、overrides、自动加载 deploy/.env（从旧版迁移）
├── llm/
│   ├── openai_compat.py             # OpenAICompatLLM / OpenAICompatEmbedding
│   ├── langchain_adapter.py         # 只供 apps/ 使用：用同一份配置构造 ChatOpenAI
│   └── utils.py                     # 去掉 <think> 内容、token 统计、重试与限流
├── prompts/
│   ├── registry.py                  # PromptRegistry：加载、解析版本、渲染、锁定校验
│   └── templates/<name>/            # 每个提示词一个目录，每个版本一个文件，见 design/prompt-management.md
├── storage/
│   ├── milvus.py                    # MilvusVectorStorage（由 KnowledgeBase.py + insert.py 改造）
│   ├── local.py                     # LocalVectorStorage（numpy + JSON，零依赖，用于测试和快速上手）
│   ├── doc_status_pg.py             # PostgreSQL 文档状态（D4，待确认）
│   ├── doc_status_json.py           # 本地 JSON 文档状态
│   └── redis_session.py             # 会话存储（原 session_store.py）
├── retrieval/
│   ├── sparse/                      # BaseSparseEncoder 及其实现：pkuseg 词表 BM25 / Milvus 原生 BM25（D3）
│   ├── fusion.py                    # RRF / 加权融合
│   ├── rerank.py                    # BaseReranker（第一阶段只定义接口，提供一个 OpenAI 兼容的 rerank 实现）
│   └── context.py                   # 按 token 预算组装上下文、给证据编号
├── ingest/
│   ├── pipeline.py                  # 增量入库：状态检查 → 切块 → 向量化 → 写入 → 更新状态
│   └── chunkers.py                  # QAPairChunker / SectionParagraphChunker / TokenChunker
├── datasets/
│   ├── download.py                  # 下载与 sha256 校验
│   ├── cleaning/                    # 清洗规则（每条规则带名称和版本）+ 清洗报告
│   ├── subsets.py                   # 子集定义（固定种子）+ manifest
│   └── loaders/                     # huatuo_qa / graphrag_bench / medqa / mirage / cmb / huatuo_encyclopedia
├── eval/
│   ├── runner.py                    # 采样、收集答案、落盘（从旧版迁移）
│   ├── metrics.py                   # Hit@k / MRR / 选择题准确率 / 置信区间
│   ├── scorers/                     # ragas、GraphRAG-Bench 官方指标
│   └── manifest.py                  # RunManifest：保证运行可复现、可对比
├── apps/
│   ├── simple_rag.py · dialogue_rag.py · memory.py
│   └── agent/                       # SearchGraph / MedicalAgent / tools
├── tools/annotation.py              # 数据标注（非核心）
└── cli.py                           # medrag data|ingest|eval|prompts 子命令
```

---

## 4. 核心数据模型（`types.py`）

```python
class Document(BaseModel):
    """入库的最小单位：一个 QA 对、一本书、一篇指南都是一个 Document"""
    doc_id: str                         # 由 loader 决定：QA 用 question_hash，教材用 "medqa_zh/内科学"；缺省用 md5(content)
    content: str
    doc_type: Literal["qa", "textbook", "guideline", "drug_label", "literature", "other"]
    lang: Literal["zh", "en"]
    title: str = ""
    source: str = ""                    # 数据集名或机构名
    metadata: dict[str, Any] = {}       # QA 的 question / answer、教材的书名、指南的发布年份等

    @property
    def content_hash(self) -> str: ...  # 增量入库时用它判断内容是否变化


class Chunk(BaseModel):
    """检索的最小单位"""
    chunk_id: str                       # f"{doc_id}#{order}"
    doc_id: str
    order: int
    content: str                        # 参与 embedding 和 BM25 的文本
    summary: str = ""                   # 可选的第二向量字段：QA 的问题、教材的章节标题
    section: str = ""                   # 章节路径，如 "内科学/第二章 急性上呼吸道感染/实验室检查"
    doc_type: str
    lang: str
    source: str = ""
    metadata: dict[str, Any] = {}


class QueryParam(BaseModel):
    """面向领域语义的检索参数；不暴露 anns_field / metric_type / expr 这类存储细节"""
    mode: Literal["naive", "hybrid"] = "hybrid"   # naive = 只用稠密向量；local / global / mix 留给图谱阶段
    top_k: int = 10                     # 最终返回的 chunk 数
    candidate_k: int = 50               # 每一路召回的候选数
    filters: dict[str, Any] = {}        # {"doc_type": "textbook", "source": ["medqa_zh"]}，由存储实现翻译
    fusion: Literal["rrf", "weighted"] = "rrf"
    weights: dict[str, float] = {}      # fusion=weighted 时生效，如 {"content_dense": 0.6, "content_sparse": 0.4}
    rerank: bool = False
    rewrite: bool = False               # 是否先改写查询


class ScoredChunk(BaseModel):
    chunk: Chunk
    score: float
    channel_scores: dict[str, float] = {}         # 各路召回的原始分数，方便调参和排查


class QueryResult(BaseModel):
    answer: str
    chunks: list[ScoredChunk]           # 进入上下文的证据，顺序即引用编号 [1]..[n]
    trace: "Trace"                      # 用到的提示词版本、模型、token 用量、各阶段耗时
```

---

## 5. 核心接口（`base.py`）

```python
class LLMFunc(Protocol):
    model: str
    async def __call__(self, messages: list[Message], *, json_schema: dict | None = None,
                       **params) -> LLMResult: ...        # LLMResult: text / reasoning / usage / latency
    def stream(self, messages: list[Message], **params) -> AsyncIterator[LLMDelta]: ...

class EmbeddingFunc(Protocol):
    model: str
    dim: int
    async def __call__(self, texts: list[str]) -> list[list[float]]: ...   # 内部负责分批、并发和重试

class BaseSparseEncoder(ABC):
    managed_by_store: bool              # True 时由向量库自己算（Milvus 原生 BM25），encode 不会被调用
    async def encode_documents(self, texts: list[str]) -> list[dict[int, float]]: ...
    async def encode_query(self, text: str) -> dict[int, float]: ...

class BaseVectorStorage(ABC):
    async def initialize(self) -> None: ...                  # 建表、建索引（幂等）
    async def upsert(self, chunks: list[Chunk], vectors: ChunkVectors) -> None: ...
    async def delete_by_doc_ids(self, doc_ids: list[str]) -> int: ...
    async def search(self, query: QueryVectors, param: QueryParam) -> list[ScoredChunk]: ...
    async def get(self, chunk_ids: list[str]) -> list[Chunk]: ...
    async def count(self) -> int: ...
    async def drop(self) -> None: ...
    async def close(self) -> None: ...

class DocStatusStorage(ABC):
    async def get(self, doc_ids: list[str]) -> dict[str, DocStatus]: ...
    async def upsert(self, records: list[DocStatus]) -> None: ...
    async def list_by_status(self, status: DocState, limit: int = 1000) -> list[DocStatus]: ...
    async def delete(self, doc_ids: list[str]) -> None: ...
# DocStatus: doc_id / content_hash / state(pending|processing|processed|failed) / chunk_ids / error / updated_at

class BaseChunker(ABC):
    name: str; version: str             # 写进 chunk 元数据和运行记录；切块策略一变，评测结果就不可比
    def chunk(self, doc: Document) -> list[Chunk]: ...

class BaseReranker(ABC):
    async def rerank(self, query: str, chunks: list[ScoredChunk], top_k: int) -> list[ScoredChunk]: ...
```

---

## 6. 门面类

```python
rag = MedRAG(
    workspace="medqa_zh",                       # 映射为 collection 名、文档状态的命名空间；一个 workspace 只有一种语言（D16）
    llm=OpenAICompatLLM.from_config(cfg.llm),
    embedding=OpenAICompatEmbedding.from_config(cfg.embedding),
    vector_storage=MilvusVectorStorage(cfg.storage.milvus, dense_fields=["content", "summary"]),
    doc_status=JsonDocStatusStorage("runs/medqa_zh/doc_status.json"),
    sparse=MilvusBM25Encoder(analyzer="chinese"),
    prompts=PromptRegistry.from_config(cfg.prompts),
)
await rag.initialize()
report = await rag.ainsert(docs, chunker=SectionParagraphChunker(max_chars=600))
#   → InsertReport(new=…, updated=…, skipped=…, failed=…)
hits   = await rag.aretrieve("急性咽扁桃体炎的病原体", QueryParam(top_k=8))
result = await rag.aquery("急性咽扁桃体炎的病原体", QueryParam(top_k=8, rerank=True))
await rag.adelete(["medqa_zh/内科学"])
await rag.aclose()
```

---

## 7. 数据流

**入库（增量、幂等）**

```
Document[]
  → DocStatus.get(doc_ids)
      content_hash 不变且状态为 processed → 跳过
      content_hash 变了                   → 先 delete_by_doc_ids，再重新处理
  → 标记 processing
  → chunker.chunk
  → embedding（分批 + 并发 + 重试）  ┐
  → sparse.encode_documents（可选）  ┘ 两者并行
  → vector_storage.upsert
  → 标记 processed，记录 chunk_ids；出错标记 failed，并记录错误信息
```

进程中断后重跑：状态为 `processing` 和 `failed` 的文档会被重新处理。由于 upsert 以 chunk_id 为主键，重复写入是安全的。

**查询（第一阶段）**

```
query → [可选改写，LLM] → 查询向量化（稠密 + 稀疏）
      → vector_storage.search（多路召回 + 融合）→ [可选 rerank]
      → context.build（token 预算，给证据编号）→ prompts.render("rag_answer")
      → llm → QueryResult（answer + 证据 + trace）
```

---

## 8. 模型层（已定：D2）

- 统一使用 **`openai` 官方 SDK**（`AsyncOpenAI`）走网络请求。`base_url`、`api_key_env`、`model`、超时、重试和并发都可以配置。DashScope、vLLM、Ollama（`/v1` 端点）、OpenAI 都按 OpenAI 兼容协议接入。
- 后续接入其他模型商，只需要再实现一个满足 `LLMFunc` / `EmbeddingFunc` 协议的类，调用方不用改。
- 以下逻辑统一收拢到 `llm/`，不再散落在 RAG 和 Agent 代码里：
  - 去掉思考内容（`<think>…</think>` 和 `reasoning_content`）；
  - 统计 token 用量（以接口返回的 `usage` 为准）；
  - Embedding 分批（DashScope 单次最多 10 条，由 `max_batch` 配置）；
  - 用信号量控制并发。
- `apps/` 里的 LangGraph 需要 LangChain 的 ChatModel。由 `llm/langchain_adapter.py` 用**同一份配置**构造 `ChatOpenAI`，保证两边连的是同一个模型。

```yaml
# config/models.yaml（示例）
llm:
  base_url: https://dashscope.aliyuncs.com/compatible-mode/v1
  api_key_env: DASHSCOPE_API_KEY
  model: qwen3.5-plus
  params: {temperature: 0.1}
  timeout: 120
  max_retries: 3
  concurrency: 8
embedding:
  base_url: https://dashscope.aliyuncs.com/compatible-mode/v1
  api_key_env: DASHSCOPE_API_KEY
  model: text-embedding-v4
  dim: 1024
  max_batch: 10
  concurrency: 4
```

---

## 9. 存储层

### Milvus collection（由 `Chunk` 生成，一个 workspace 对应一个 collection）

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `chunk_id` | VARCHAR，主键 | `doc_id#order` |
| `doc_id` | VARCHAR | 建标量索引，按文档删除和过滤时使用 |
| `content` | VARCHAR | 开启 analyzer，供 Milvus 原生 BM25 使用 |
| `summary` | VARCHAR | 可以为空 |
| `doc_type` / `lang` / `source` / `section` / `title` | VARCHAR | 过滤字段 |
| `order` | INT64 | |
| `metadata` | JSON | 其他元数据 |
| `content_dense` / `summary_dense` | FLOAT_VECTOR | `summary_dense` 可以通过配置关闭 |
| `content_sparse` | SPARSE_FLOAT_VECTOR | 自管理 BM25 或 Milvus Function 输出 |

- VARCHAR 的上限是 65,535 **字节**，中文每个字占 3 字节，所以单个字段大约只能放 2.1 万字。切块器必须保证 `content` 不超过这个长度，这也是旧版超长 answer 的一个隐患。
- **中英文隔离（D16，已定）**：一个 workspace 只能有一种语言，`lang` 写在 workspace 配置里，建表时据此选择 analyzer（中文 jieba / 英文 standard）。入库时，如果文档的 `lang` 和 workspace 不一致，直接报错，不允许静默混入。`Chunk.lang` 字段保留，供以后做多语言混合检索。

### 文档状态（D4，待确认）
生产环境用 PostgreSQL（`deploy/` 里已经部署）；单测和快速上手用本地 JSON。

### 会话（Redis）
原样沿用 v2 的 `RedisSessionStore`：key 规划、分布式锁和 TTL 都不变。

---

## 10. 稀疏检索（D3，待确认）

两种实现都放在 `BaseSparseEncoder` 接口之后：

| 实现 | 优点 | 缺点 |
| --- | --- | --- |
| `VocabBM25Encoder`（旧版 pkuseg medicine） | 医学分词质量好 | IDF 和 avgdl 在建词表时就冻结了，增量入库后需要定期重建；英文需要另接一个分词器 |
| `MilvusBM25Encoder` | 统计量由 Milvus 自动维护，天然支持增量；英文有现成的 analyzer | 中文分词用 jieba，医学术语需要加载自定义词典 |

默认值的建议：用 `MilvusBM25Encoder`，在 A、C 两层上各做一次 A/B 实验后再最终确定。

---

## 11. 提示词

提示词全部改为**带版本的文件**，由 `PromptRegistry` 统一管理，代码里不再内联提示词字符串。每次 LLM 调用都会在 trace 里记录 `name@version#sha`，评测结果因此可以按提示词版本对比。详见 [design/prompt-management.md](design/prompt-management.md)。

---

## 12. 可复现评测

每次评测都会写出一份 `RunManifest`，和结果文件放在一起（`runs/<dataset>/<run_id>/`）：

```yaml
run_id: 20261015-1432-a1b2
git: {commit: 60b7966, dirty: false}
dataset: {name: medqa_zh, subset: c_mini, version: v1, manifest_sha: …}   # 数据版本
prompts: {rag_answer: v1#3fa2…, query_rewrite: v1#9c01…}                  # 提示词版本
models: {llm: qwen3.5-plus, embedding: text-embedding-v4, judge: …}
config_sha: …                                                            # 检索、切块等全部配置的哈希
chunker: {name: section_paragraph, version: v1}
seed: 42
```

`medrag eval compare <run_a> <run_b>` 会输出：

- 各指标的差值和 95% 置信区间；
- 按题型或科目分组的对比；
- 逐题差异：前对后错、前错后对的题目清单。

只要两次运行的 manifest 里有字段不同，报告就会在开头列出这些不同项，避免把数据或提示词的变化误当成算法本身的提升。

---

## 13. 测试策略

| 类型 | 依赖 | 说明 |
| --- | --- | --- |
| 单元测试 | 无 | 使用 `LocalVectorStorage`、`JsonDocStatusStorage`，以及确定性的假 LLM / 假 Embedding（按文本哈希生成向量），不需要网络和 Docker |
| 集成测试 | Milvus / Redis / PostgreSQL | 标记为 `@pytest.mark.services`，需要先执行 `deploy/start.sh` |
| 回归测试 | 真实 Embedding 服务 | A 层的 198 题 Hit@k，作为每个里程碑的合并门槛 |
