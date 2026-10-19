# 架构设计

> 本文定义 medrag 的结构、技术方案和用法。项目要做什么、为什么这样做、分几步走，见 [solution.md](solution.md)。
> 现状：M0 已完成。本文描述目标形态，并标注各部分是已经存在，还是计划在哪个里程碑出现；当前代码与目标之间的差距见第 15 节。

---

## 1. 设计原则

第 1～6 条是 M0 已经落地的公共基础（D31），新代码必须遵守；第 7～9 条是知识层新增的约定。

1. **只用异步**：IO 方法一律写成 `async def`，不保留同步版本。CPU 密集的操作（pkuseg 分词、PDF 解析），以及只有同步驱动的库（SQLite），都用 `asyncio.to_thread`。
2. **依赖注入**：存储、模型客户端、提示词注册表都由调用方创建后传入，组件内部不自己创建；组件只接收自己用到的那一段配置，不接收整个 `AppConfig`。按配置组装默认组件，只在组装根（`build.py`）里做。
3. **模型层统一走 OpenAI 兼容协议**（D2）：LLM 和 Embedding 都用 `openai` SDK（`AsyncOpenAI`）访问。核心层不 import LangChain / LangGraph，例外只有三处：`apps/`、`llm/langchain_adapter.py`、`eval/scorers/`。由 `tests/test_layering.py` 检查。
4. **存储细节不外泄**：对外只暴露 `QueryParam` 这类领域参数。具体约束：
   - Milvus 的 `anns_field`、`metric_type`、`expr` 只出现在 `storage/milvus.py`；
   - Cypher 只出现在 `storage/graph*.py`；
   - SQL 只出现在 `storage/` 下。

   这些细节都不交给 LLM 或上层应用。
5. **提示词放在文件里**：代码里不写提示词字符串，通过注入的 `PromptRegistry` 按名称获取（第 11 节）。
6. **服务可以不配地址**（D18）：Milvus、Redis、PostgreSQL、Neo4j 没配地址时，分别降级为 Milvus Lite、fakeredis、SQLite、FalkorDBLite（第 6.5 节）。新写的存储代码必须同时支持两种后端，并在两种后端上各测一遍。
7. **关系库是唯一真相源**（D25）：文档、Claim、概念、治理事件都以关系库为准。Milvus 和图数据库是派生索引，随时可以从关系库重建。
8. **回答路径只读**（D26）：查询不写知识库。写入只能经由入库流程或后台治理，每次写入都伴随一条治理事件。
9. **可复现**：每次回答的 trace 都记录提示词版本、模型、命中的 Claim 和当时的治理事件序号；每次评测都写出 RunManifest（第 13 节）。

---

## 2. 分层

```
┌──────────────────────────────────────────────────────────────────────┐
│ apps/        SimpleRAG · DialogueRAG · MedicalAgent（LangGraph）        │  应用层：可以依赖 LangChain / LangGraph；
│              医生助手 · 患者助手 · 安全策略（阶段三）                      │  会话状态在 Redis，组件本身无状态
├──────────────────────────────────────────────────────────────────────┤
│ medrag.py    MedRAG 门面：ainsert / aquery / aretrieve / aclaims /         │  门面层
│              averify / aconflicts / agovern / areplay / adelete          │
├─────────┬─────────┬──────────┬────────────┬──────────┬────────┬───────┤
│ ingest/ │ extract/│ concepts/│ governance/│retrieval/│ answer/│ guard/│  能力层
│ 解析     │ Claim    │ 规范化    │ 时效 · 冲突 │ 双通道    │ 作答    │ 确定性 │  （另有 prompts/、eval/、
│ 切块     │ 抽取     │ 概念层级  │ 合并 · 任务 │ 条件匹配  │ 核查    │ 工具   │    datasets/）
│ 入库编排 │ 证据对齐 │          │ 队列        │          │        │ 风险信号│
├─────────┴─────────┴──────────┴────────────┴──────────┴────────┴───────┤
│ storage/     关系库（真相源）· 向量索引 · 图投影 · 会话；未配地址时降级        │  基础层
│ llm/         OpenAI 兼容的 LLM / Embedding 客户端                         │
│ types.py · base.py · config/                                          │
└──────────────────────────────────────────────────────────────────────┘
```

**依赖规则**：

1. 依赖方向只能是上层 → 下层。
2. 能力层内部只允许以下几种依赖，同层之间不能成环：
   - `extract/`、`concepts/`、`retrieval/`、`guard/` 只依赖基础层；
   - `governance/` 可以依赖 `extract/`、`concepts/`；
   - `ingest/` 是入库的编排者，可以依赖 `extract/`、`concepts/`、`governance/`；
   - `answer/` 可以依赖 `retrieval/`、`guard/`。
3. 安全**策略**（拦截、改写、转人工、按受众调整）只在 `apps/`；core 的 `guard/` 只产出事实和风险信号（D30）。

---

## 3. 目录结构

标注说明：「已有」表示 M0 已存在；「Mx」表示计划在哪个里程碑出现或改造。

```
src/medrag/
├── __init__.py                 # 已有；M1 起导出公共 API：MedRAG、QueryParam、Document、Claim……
├── medrag.py                   # M1：门面类
├── build.py                    # M1：组装根，按配置创建默认组件（未配地址时自动降级）
├── types.py                    # 已有（只有过渡类型 LegacyDocument）；M1 起放第 4 节的数据模型
├── base.py                     # M1：抽象接口（第 5 节）
├── config/                     # 已有：配置模型与多 YAML 加载器；M1 起新增 knowledge 段
├── llm/                        # 已有：openai_compat.py、langchain_adapter.py、utils.py
├── prompts/                    # 已有：registry.py、templates/<名称>/<版本>.<语言>.yaml、prompts.lock
├── storage/
│   ├── backends.py             # 已有：未配地址时的降级选择（D18），选择逻辑只在这里
│   ├── milvus.py               # 已有（v2 的 MedicalHybridKnowledgeBase）；M1 改造为 MilvusVectorStorage
│   ├── milvus_io.py            # 已有：批量写入辅助；M1 并入 milvus.py
│   ├── sql.py                  # 已有：SqlDatabase（PostgreSQL / SQLite）；M1 增加事务
│   ├── graph.py                # 已有：GraphDatabase（Neo4j / FalkorDBLite），执行 Cypher
│   ├── redis_session.py        # 已有：会话存储（Redis / fakeredis）
│   ├── knowledge_store.py      # M1：关系库上的文档、Claim、概念、冲突集、事件、任务（真相源）
│   └── graph_projection.py     # M1：消费治理事件，维护图投影（两种后端共用的 openCypher 子集）
├── ingest/
│   ├── pipeline.py             # 已有（v2 形态）；M1 重写为带版本语义的增量入库编排
│   ├── chunkers.py             # M1：QAPairChunker / SectionParagraphChunker / TokenChunker
│   └── parsers/                # M2：PDF / HTML → 带章节的文本；说明书按【禁忌】等章节切分
├── extract/                    # M3：schema.py（实体类型、谓词、条件）、extractor.py、align.py（证据对齐）
├── concepts/                   # M4：linker.py（别名表 → 向量 → LLM）、vocab/（医保药品编码、ICD-10、ATC 导入器）
├── governance/                 # M5：validity.py、conflicts.py、merge.py、tasks.py、worker.py
├── retrieval/
│   ├── sparse/                 # 已有：pkuseg 自管理词表的 BM25；M1 增加 Milvus 托管 BM25（D3）
│   ├── fusion.py               # M1：RRF / 加权融合
│   ├── rerank.py               # M1：BaseReranker 与一个 OpenAI 兼容的实现
│   ├── claims.py               # M6：按概念取 Claim、层级扩展、条件匹配
│   └── context.py              # M6：按 token 预算组装上下文、给证据编号
├── answer/                     # M6：understand.py（概念 + 条件）、compose.py、verify.py（逐句核查）
├── guard/                      # M6 起：确定性工具与风险信号；阶段三扩充
├── datasets/                   # M2：下载与校验、清洗规则、子集与 manifest、loaders/
├── eval/                       # 已有：runner.py、adapters.py、scorers/；M2 增加 manifest.py，M7 增加 timesliced.py
├── apps/                       # 已有：应用层（SimpleRAG、DialogueRAG、memory、agent/）
├── tools/                      # 已有：数据标注等非核心工具
├── testing/                    # 已有：fake_openai.py；M3 起支持按提示词名返回预置 JSON
└── cli.py                      # M2：medrag data | ingest | govern | eval | prompts
```

包外的目录：

| 目录 | 内容 |
| --- | --- |
| `scripts/` | 使用示例和里程碑回归脚本 |
| `tests/` | 单测与集成测试 |
| `deploy/` | 四个服务的 docker compose；`deploy/storage.yaml` 是连接它们的配置 |
| `software/` | 前后端示例应用，冻结到阶段三（D11） |
| `.medrag/` | 降级实现的本地数据，不进 git |

---

## 4. 数据模型（`types.py`，M1 起）

以下字段是首版设计。Claim 的谓词和条件清单在 M3 用金标准定稿（D36）。

```python
class Document(BaseModel):
    """入库的最小单位：一个 QA 对、一本教材、一份指南的一个版本、一份说明书的一个版本"""
    doc_id: str                         # 一个具体版本的唯一 ID，如 "guideline/中国高血压防治指南@2024"
    work_id: str = ""                   # 同一作品的各个版本共用；缺省等于 doc_id（没有版本概念的文档）
    version: str = ""                   # "2024年修订版"、"2017-03 修订"
    content: str
    doc_type: Literal["guideline", "drug_label", "textbook", "gov_doc", "literature", "qa", "other"]
    lang: Literal["zh", "en"]           # 必须与 workspace 的语言一致（D16）
    title: str = ""
    publisher: str = ""                 # 发布机构：学会、卫健委、药企、出版社
    published_at: date | None = None    # 发布 / 修订日期
    effective_from: date | None = None  # 开始生效的日期，缺省取 published_at
    supersedes: list[str] = []          # 被本版本取代的旧版本 doc_id；不同机构的指南之间不是取代关系
    region: str = ""                    # 适用地区，如 "CN"
    evidence_level: str = ""            # 文献类型或证据等级：指南、系统综述、RCT、教材、问答……
    source: str = ""                    # 数据集名或获取渠道
    metadata: dict[str, Any] = {}

    @property
    def content_hash(self) -> str: ...  # 增量入库时用来判断内容是否变化


class Chunk(BaseModel):
    """文本块：检索的最小单位，同时是 Claim 证据的容器"""
    chunk_id: str                       # f"{doc_id}#{order}"
    doc_id: str
    order: int
    content: str                        # 参与 embedding 和 BM25 的文本
    summary: str = ""                   # 可选的第二向量字段：QA 的问题、教材的章节标题
    section: str = ""                   # 章节路径，如 "二甲双胍片/【禁忌】"
    start: int = 0                      # 在 Document.content 中的字符偏移，证据对齐用
    end: int = 0
    doc_type: str
    lang: str
    metadata: dict[str, Any] = {}


class Evidence(BaseModel):
    """Claim 的出处：精确到原句"""
    chunk_id: str
    start: int                          # 在 Chunk.content 中的字符区间
    end: int
    quote: str                          # 原句；对齐失败的抽取结果直接丢弃，不进 Claim 库


class Concept(BaseModel):
    concept_id: str                     # 锚定时如 "ATC:A10BA02"、"ICD10:I10"；未锚定时为 "local:<hash>"
    type: Literal["disease", "symptom", "ingredient", "product", "drug_class",
                  "lab_test", "procedure", "population", "organism", "anatomy"]
    name: str                           # 规范名
    aliases: list[str] = []
    codes: dict[str, str] = {}          # {"ATC": "A10BA02", "NHSA": "...", "ICD10": "...", "LOINC": "..."}
    anchored: bool                      # 允许未锚定的概念存在，但必须标记（D24）
    merged_into: str | None = None      # 被合并后指向保留的概念；原记录不物理删除，方便拆回


class ConceptRelation(BaseModel):
    """粒度不同的概念之间用关系连接，不合并"""
    src: str
    rel: Literal["is_a", "has_ingredient", "brand_of", "same_as_candidate"]
    dst: str
    score: float | None = None          # same_as_candidate 的置信度


class Mention(BaseModel):
    """原文里的一次提及；合并与拆分都以 Mention 为单位重新归属"""
    text: str
    evidence: Evidence
    concept_id: str
    link_score: float
    link_method: Literal["exact", "alias", "vector", "llm", "human"]


class Condition(BaseModel):
    kind: Literal["measure", "population", "comorbidity", "co_medication", "phase", "other"]
    concept_id: str | None = None       # eGFR → LOINC 概念；妊娠 → population 概念；合并用药 → ingredient 概念
    op: Literal["<", "<=", ">", ">=", "between", "=", "present", "absent"] | None = None
    value: float | list[float] | None = None
    unit: str = ""
    text: str                           # 原文措辞；结构化失败时至少保留它，此时不参与自动匹配


class Claim(BaseModel):
    claim_id: str
    subject: str                        # concept_id
    predicate: Literal["indication", "contraindication", "caution", "adverse_effect", "interaction",
                       "dosage", "dose_adjustment", "monitoring", "diagnostic_criterion",
                       "treatment_target", "recommendation"]
    object: str | None = None           # concept_id；剂量、控制目标这类陈述可以没有客体
    value: dict[str, Any] = {}          # 数值内容，如 {"dose": 500, "unit": "mg", "freq": "bid"}
    conditions: list[Condition] = []
    polarity: Literal["affirm", "negate"] = "affirm"
    strength: Literal["strong_for", "weak_for", "weak_against", "strong_against"] | None = None
    certainty: Literal["certain", "possible", "uncertain"] = "certain"
    frequency: str = ""                 # 不良反应的发生频率：十分常见 / 常见 / 偶见 / 罕见 / 十分罕见
    evidence_grade: str = ""            # 原文给出的推荐级别或证据等级
    evidence: list[Evidence]            # 至少一条
    source_doc_id: str
    valid_from: date | None = None      # 现实有效期
    valid_to: date | None = None
    recorded_at: datetime               # 系统时间：写入与撤回，用于审计回放
    retracted_at: datetime | None = None
    rank: Literal["preferred", "normal", "deprecated"] = "normal"
    extractor: str                      # "claim_extract@v1#3fa2… / qwen3.5-plus"
    review: Literal["auto", "reviewed", "rejected"] = "auto"


class ConflictSet(BaseModel):
    conflict_id: str
    claim_ids: list[str]
    type: Literal["complementary", "condition_mismatch", "temporal", "dispute",
                  "evidence_level", "misinformation", "extraction_error"]
    resolution: str = ""                # 处理结论，例如"新版本 preferred，旧版本 deprecated"
    status: Literal["open", "resolved"] = "open"
    reviewed_by: str = ""


class GovernanceEvent(BaseModel):
    """只追加。图投影、审计回放、回滚都基于它"""
    seq: int                            # 单调递增
    type: str                           # doc_added / doc_superseded / claim_added / claim_retracted / claim_ranked /
                                        # concept_merged / concept_split / conflict_detected / conflict_resolved ……
    payload: dict[str, Any]
    actor: Literal["ingest", "governance", "query_driven", "human"]
    created_at: datetime
```

**文档版本的语义**：

| 情况 | 例子 | 处理 |
| --- | --- | --- |
| 同一 `doc_id` 的内容变了 | OCR 纠错、修正解析错误 | 更正：替换旧文本块；旧 Claim 写上 `retracted_at`（软撤回，留痕） |
| 新的 `doc_id`，`supersedes` 指向旧版本 | 新版说明书、新版指南 | 修订：旧版本标记为 superseded，**不删除**；冲突检测把旧版本的 Claim 标为 deprecated，并写上 `valid_to` |
| `adelete` | 撤下一份文档 | 从检索中移除；Claim 软撤回；事件留痕 |

**查询相关**：

```python
class QueryParam(BaseModel):
    """面向领域语义的查询参数，不暴露 anns_field / metric_type / expr / Cypher 这类存储细节"""
    mode: Literal["naive", "hybrid", "claims", "mix"] = "mix"
    # naive = 只用稠密向量查文本块；hybrid = 文本块稠密 + 稀疏；claims = 只用 Claim；mix = Claim + 文本块
    top_k: int = 10
    candidate_k: int = 50               # 每一路召回的候选数
    filters: dict[str, Any] = {}        # {"doc_type": "guideline", "region": "CN"}，由存储实现翻译
    fusion: Literal["rrf", "weighted"] = "rrf"
    weights: dict[str, float] = {}
    rerank: bool = False
    rewrite: bool = False
    as_of: date | None = None           # 按某个时间点作答，缺省为今天
    context: dict[str, Any] = {}        # 用户给出的条件，如 {"eGFR": 40, "pregnant": False, "medications": ["华法林"]}
    expand_hierarchy: bool = True       # 沿概念层级扩展（类别继承）
    max_hops: int = 2
    include_deprecated: bool = False
    verify: bool = True                 # 回答后逐句核查


class QueryResult(BaseModel):
    status: Literal["answered", "need_info", "no_evidence"]
    answer: str                         # 逐句带 [n] 引用
    missing: list[MissingCondition]     # status=need_info 时：缺哪些关键条件，由上层决定是否追问
    claims: list[ScoredClaim]           # 进入上下文的 Claim，以及每条的条件匹配结果（满足 / 不满足 / 缺失）
    chunks: list[ScoredChunk]           # 进入上下文的文本块
    conflicts: list[ConflictSet]        # 命中的 Claim 所在的冲突集，带类型，供上层决定如何呈现
    signals: list[RiskSignal]           # 风险信号：未被支持的句子、过时引用、高风险内容……
    trace: Trace                        # 提示词版本、模型、token 用量、各阶段耗时、命中的 claim_id、事件序号
```

---

## 5. 核心接口（`base.py`，M1 起）

```python
class LLMFunc(Protocol):                         # 已有实现：llm/openai_compat.py 的 OpenAICompatLLM
    model: str
    async def __call__(self, messages: list[Message], *, json_schema: dict | None = None, **params) -> LLMResult: ...
    def stream(self, messages: list[Message], **params) -> AsyncIterator[LLMDelta]: ...

class EmbeddingFunc(Protocol):                   # 已有实现：OpenAICompatEmbedding（分批、并发、保持顺序）
    model: str
    dim: int
    async def __call__(self, texts: list[str]) -> list[list[float]]: ...

class BaseSparseEncoder(ABC):                    # D3：pkuseg 自管理词表 / Milvus 托管 BM25
    managed_by_store: bool                       # True 时由向量库自己计算，encode 不会被调用
    async def encode_documents(self, texts: list[str]) -> list[dict[int, float]]: ...
    async def encode_query(self, text: str) -> dict[int, float]: ...

class BaseVectorStorage(ABC):                    # Milvus / Milvus Lite；一个 workspace 三个 collection（第 6.2 节）
    async def initialize(self) -> None: ...      # 建表、建索引（幂等）
    async def upsert(self, kind: IndexKind, records: list[IndexRecord]) -> None: ...   # kind: chunk / claim / alias
    async def delete(self, kind: IndexKind, ids: list[str]) -> int: ...
    async def search(self, kind: IndexKind, query: QueryVectors, param: QueryParam) -> list[Hit]: ...
    async def drop(self) -> None: ...
    async def close(self) -> None: ...

class KnowledgeStore(ABC):                       # 关系库上的真相源（第 6.1 节）
    documents: DocumentRepo                      # 文档、版本链、入库状态
    chunks: ChunkRepo
    concepts: ConceptRepo                        # 概念、别名、概念关系、提及
    claims: ClaimRepo                            # Claim、条件、证据；按概念 / 谓词 / 有效期 / rank 查询
    conflicts: ConflictRepo
    events: EventLog                             # 只追加；按 seq 读取
    tasks: TaskQueue                             # 后台任务
    def transaction(self) -> AsyncContextManager: ...   # 状态变更和对应的事件在同一事务里写入

class GraphProjection(ABC):                      # Neo4j / FalkorDBLite（第 6.3 节）
    async def apply(self, events: list[GovernanceEvent]) -> int: ...       # 幂等；返回投影到的最新 seq
    async def rebuild(self, store: KnowledgeStore) -> None: ...            # 清空后按事件重放
    async def expand(self, concept_ids: list[str], rels: list[str], max_hops: int) -> list[str]: ...
    async def paths(self, src: str, dst: str, max_hops: int) -> list[GraphPath]: ...
    async def last_seq(self) -> int: ...

class BaseParser(ABC):                           # M2：PDF / HTML → 带章节的文本
    async def parse(self, raw: bytes, *, mime: str) -> ParsedDocument: ...

class BaseChunker(ABC):
    name: str; version: str                      # 写进 chunk 元数据和运行记录；切块策略一变，评测结果就不可比
    def chunk(self, doc: Document) -> list[Chunk]: ...

class BaseClaimExtractor(ABC):                   # M3
    async def extract(self, doc: Document, chunks: list[Chunk]) -> list[ClaimDraft]: ...   # 草稿：尚未链接概念

class BaseConceptLinker(ABC):                    # M4
    async def link(self, mentions: list[str], *, type_hint: str | None = None) -> list[LinkResult]: ...

class ConflictDetector(ABC):                     # M5
    async def detect(self, new: list[Claim], store: KnowledgeStore) -> list[ConflictSet]: ...

class BaseReranker(ABC):
    async def rerank(self, query: str, items: list[Hit], top_k: int) -> list[Hit]: ...
```

---

## 6. 存储

### 6.1 关系库：唯一真相源（PostgreSQL / SQLite，D4）

基于已有的 `storage/sql.py` 中的 `SqlDatabase`。表名都带 workspace 前缀：

| 表 | 内容 |
| --- | --- |
| `documents`、`doc_status` | 文档正文与元数据、版本链（`work_id`、`supersedes`）、入库状态（pending / processing / processed / failed） |
| `chunks` | 文本块正文与字符偏移（向量在 Milvus） |
| `concepts`、`concept_aliases`、`concept_relations`、`mentions` | 概念、别名、粒度关系、原文提及 |
| `claims`、`claim_conditions`、`claim_evidence` | Claim 本体、结构化条件（便于做区间查询）、证据区间 |
| `conflict_sets`、`conflict_members` | 冲突集与成员 |
| `events` | 治理事件，只追加，`seq` 单调递增 |
| `tasks` | 后台任务队列：`project_graph` / `extract_uncovered` / `judge_merge` / `recheck_conflict` / `human_review` |

约定：
- 沿用 `SqlDatabase` 的规则：
  - 统一用 `?` 占位符；
  - 只用两种数据库都支持的语法（`CREATE TABLE IF NOT EXISTS`、`INSERT ... ON CONFLICT ... DO UPDATE`）；
  - SQLite 走 `asyncio.to_thread`。
- `SqlDatabase` 目前每条语句都自动提交。M1 给它增加事务（`async with db.transaction()`），状态变更和对应的事件必须在同一个事务里写入。
- 有效期查询（`as_of`）、按条件区间筛选、冲突分组，都在这一层用 SQL 完成。

### 6.2 向量索引：派生（Milvus / Milvus Lite）

一个 workspace 有三个 collection：

| collection | 主键 | 向量 | 用于过滤的标量字段 | 用途 |
| --- | --- | --- | --- | --- |
| `{ws}_chunks` | `chunk_id` | `content_dense`、可选的 `summary_dense`、`content_sparse` | `doc_id`、`doc_type`、`section`、`status`（是否已被取代） | 文本块通道，延续已有能力 |
| `{ws}_claims` | `claim_id` | Claim 文本化表述的稠密向量与稀疏向量 | `subject`、`predicate`、`rank`、`valid_from`、`valid_to` | Claim 的语义召回兜底 |
| `{ws}_aliases` | `alias_id` | 别名的稠密向量 | `concept_id`、`type` | 概念链接 |

- Claim 的文本化表述形如"二甲双胍｜不推荐起始用药｜eGFR 30–45 mL/min/1.73m²｜某说明书 第 3 版"。它只负责"找到"；条件是否满足，由关系库里的结构化字段判断。
- 沿用 M0 实测得出的规则：
  - 批量写入后必须先 flush，再建索引、再检索；
  - VARCHAR 上限是 65,535 **字节**（中文约 2.1 万字），切块器必须保证不超过；
  - analyzer 按 workspace 的语言选择：中文 jieba，英文 standard（D16）。
- 修改 collection 的 schema 需要维护者确认。

### 6.3 图投影：派生（Neo4j / FalkorDBLite）

基于已有的 `storage/graph.py` 中的 `GraphDatabase`，图名或标签带 workspace 前缀。

```
(:Document {id, work_id, version, status})  -[:SUPERSEDES]->  (:Document)                 M1
(:Concept  {id, type, name, anchored})       -[:IS_A | HAS_INGREDIENT | BRAND_OF]->  (:Concept)   M4
(:Claim    {id, predicate, rank, valid_from, valid_to})
     -[:SUBJECT]-> (:Concept)   -[:OBJECT]-> (:Concept)   -[:CONDITION]-> (:Concept)       M4
     -[:FROM]-> (:Document)     -[:CONFLICTS_WITH {type}]-> (:Claim)                       M5
```

- **怎么维护**：投影器按 `seq` 顺序消费治理事件，用幂等的 `MERGE` 写入，并记下已投影到的 `seq`。入库流程结束时会运行一次投影器，后台 worker 里也会持续运行它。任何时候都可以清空后按事件重放（`rebuild`）。
- **用来做什么**：
  - 类别继承：查"依那普利"时，沿 `IS_A` 上溯到"ACEI"，带出类别级的 Claim；
  - 多跳查询：多药合用时的相互作用、合并症；
  - 解释路径：给出"为什么得出这个结论"的链路；
  - 阶段三人工审核时的可视化。
- **语法限制**：FalkorDB 只支持 openCypher 的子集，所以投影和查询只用两边都支持的语法，并在两种后端上各跑一遍测试。
- **一致性**：判断以关系库为准，图只负责扩展和找路径。图的投影滞后时，trace 会同时记下关系库的 `seq` 和图已投影到的 `seq`。

### 6.4 会话（Redis / fakeredis）

原样沿用已有的 `RedisSessionStore`：key 规划、分布式锁和 TTL 都不变。多轮对话的历史和摘要、Agent 的跨轮状态都存在这里，所以应用层组件本身无状态，可以多进程部署。fakeredis 的多个客户端要共用一个 `FakeServer` 才能共享数据。

### 6.5 未配置服务地址时的降级（D18，已实现）

| 服务 | 判断依据 | 降级实现 | 数据位置 |
| --- | --- | --- | --- |
| Milvus | `milvus.uri` 为空 | Milvus Lite（pymilvus 在本进程内拉起） | `milvus.lite_path`，默认 `.medrag/milvus_lite.db` |
| Redis | `redis.host` 为空 | fakeredis（含 Lua，会话锁可用） | 进程内存，不跨进程，退出即丢失 |
| PostgreSQL | `postgres.host` 为空 | SQLite（标准库，经 `asyncio.to_thread` 调用） | `postgres.sqlite_path`，默认 `.medrag/medrag.sqlite3` |
| Neo4j | `neo4j.uri` 为空 | FalkorDBLite（嵌入式 FalkorDB，兼容 openCypher 子集） | `neo4j.falkordb_path`，默认 `.medrag/falkordb.db` |

- 选择逻辑只在 `storage/backends.py`。包内默认的 `storage.yaml` 不配地址，所以开箱就是降级模式。连接 `deploy/` 启动的服务时，用 `deploy/storage.yaml` 整段覆盖（`MEDRAG_CONFIG_OVERRIDES=deploy/storage.yaml`）。
- 每种服务第一次降级时打一条 warning，避免生产环境漏配地址后悄悄跑在嵌入式实现上。降级实现只适合单进程的开发、测试和小规模使用。
- 实测过的边界：
  - Milvus Lite 需要 pymilvus 2.6 + milvus-lite 3.x（D20），分词器只有 `standard` 和 `jieba`，没有 Windows 版本；
  - redislite（FalkorDBLite）的异步 `close` 不会落盘，所以 `GraphDatabase.close` 里会先显式执行 `SAVE`。

### 6.6 写入顺序与失败恢复

一次写入（入库或治理）按以下顺序进行：

1. 在关系库事务中写入状态变更和事件；
2. 写向量索引；
3. 投影到图。

第 2、3 步失败时，不回滚第 1 步，而是在 `tasks` 里登记重试。向量和图都是派生的，可以补写或重建，所以真相源始终保持一致。

---

## 7. 数据流

### 7.1 入库（增量、幂等、带版本语义）

```
Document[]
  → 文档状态与版本判断
      content_hash 不变，且状态为 processed   → 跳过
      同一 doc_id 内容变化（更正）            → 撤回旧的文本块与 Claim（Claim 软撤回）后重新处理
      新 doc_id，且 supersedes 指向旧版本     → 旧版本标记为 superseded，不删除
  → 标记 processing
  → parser（PDF / HTML → 章节）→ chunker → 文本块写入关系库
  → embedding（分批 + 并发 + 重试） ┐ 两者并行
  → sparse.encode_documents（可选） ┘ → 写入 {ws}_chunks
  → 需要抽取 Claim 的文档（按 doc_type 和章节规则判断，如说明书的【禁忌】【用法用量】、指南的推荐条目）：
      extractor（JSON Schema 约束输出）→ 证据对齐（原句 + 字符区间；对不上的丢弃并计数）
      → concept linker（别名表 → 向量 → LLM）
      → 在一个事务内写入 Claim 和事件
      → 冲突检测：与已有的同主体 + 同谓词（+ 同客体）的 Claim 比较，分型后写入冲突集；
        时效型冲突自动设置 valid_to 和 rank
      → Claim 的文本化表述写入 {ws}_claims
  → 投影器消费新事件，更新图
  → 标记 processed；出错标记 failed 并记录错误信息，可以续跑
```

进程中断后重跑：状态为 `processing` 和 `failed` 的文档会被重新处理。各处的写入都以 ID 为主键做 upsert，所以重复写入是安全的。

### 7.2 查询（只读）

```
query + QueryParam
  → 理解：概念提及 → 链接（别名表 → {ws}_aliases 向量 → LLM）；条件抽取（数值、人群、合并用药）；问题类型（对应的谓词）
  → Claim 通道：图扩展概念集合（IS_A 上溯类别；成分—制剂—品牌）
      → 关系库按 主体 / 客体 ∈ 概念集合、谓词、as_of 时有效、rank ≠ deprecated 取候选；用 {ws}_claims 做语义召回兜底
      → 条件匹配：context 与每条 Claim 的条件逐一比较 → 满足 / 不满足 / 缺失
  → 文本块通道：稠密 + 稀疏融合（可选 rerank）
  → 冲突：命中的 Claim 在冲突集中时，带上冲突类型
  → 缺少关键条件 → status=need_info，返回 missing（是否追问由上层决定）
     否则         → 组装上下文（Claim 原句 + 文本块）→ prompts.render → llm → 逐句引用
  → 核查：逐句对照引用的证据，标出未被支持的句子，写入 signals
  → trace：提示词版本、模型、命中的 claim_id、关系库 seq、图投影 seq
  → 投递后台任务：未被 Claim 覆盖的文本块、合并候选（阶段二启用）
```

### 7.3 后台治理

`governance/worker.py` 从 `tasks` 里取任务执行。每个任务都在事务里写入事件，再由投影器同步到图。

| 任务 | 什么时候触发 | 阶段 |
| --- | --- | --- |
| `project_graph` | 入库或治理产生了新事件 | 阶段一 |
| `recheck_conflict` | 有新的 Claim 入库，或者人工修改了 rank | 阶段一 |
| `extract_uncovered` | 查询命中了未被 Claim 覆盖的高风险文本块 | 阶段二 |
| `judge_merge` | 查询或入库时发现候选碎片（依次用规则 → 本体 ID → LLM 判定） | 阶段二 |
| `human_review` | 高风险类别的自动结论低于置信门槛 | 阶段二 |

回滚：每种事件都有对应的逆操作（合并 ↔ 拆分、改 rank ↔ 恢复原 rank），回滚本身也写成事件。

---

## 8. 门面类用法

使用体验沿用 LightRAG 式的门面：组件由调用方注入，统一提供 `ainsert` / `aquery` 等方法。

```python
import asyncio
from datetime import date

from medrag import Document, MedRAG, QueryParam
from medrag.build import build_medrag
from medrag.config import ConfigLoader

cfg = ConfigLoader().config


async def main():
    # 组装根：按配置创建默认组件，未配地址时自动降级（Milvus Lite / SQLite / FalkorDBLite）
    rag = build_medrag(cfg, workspace="cardiometabolic_zh")
    await rag.initialize()

    report = await rag.ainsert([
        Document(
            doc_id="guideline/中国高血压防治指南@2024", work_id="guideline/中国高血压防治指南",
            version="2024年修订版", doc_type="guideline", lang="zh",
            title="中国高血压防治指南（2024年修订版）", publisher="中国高血压防治指南修订委员会",
            published_at=date(2024, 8, 11), supersedes=["guideline/中国高血压防治指南@2018"],
            region="CN", content=text,
        ),
    ])
    # InsertReport(new=1, updated=0, superseded=1, skipped=0, failed=0, claims_added=…, conflicts_found=…)

    result = await rag.aquery("我 eGFR 40，能吃二甲双胍吗？", QueryParam(context={"eGFR": 40}))
    if result.status == "need_info":
        print(result.missing)        # 例如：是否已经在服用二甲双胍（起始用药和继续用药的建议不同）
    else:
        print(result.answer)         # 逐句带 [n] 引用；另有 result.claims / chunks / conflicts / signals / trace

    hits = await rag.aretrieve("二甲双胍 肾功能", QueryParam(mode="hybrid", top_k=8))       # 只检索文本块
    claims = await rag.aclaims("二甲双胍", predicate="contraindication", as_of=date(2015, 6, 1))
    verdict = await rag.averify("二甲双胍在 eGFR < 45 时禁用")   # supported / contradicted / outdated / not_found + 证据
    sets = await rag.aconflicts("高血压", predicate="diagnostic_criterion")
    await rag.agovern(max_tasks=100)               # 处理一批后台任务；也可以用 `medrag govern` 常驻运行
    replay = await rag.areplay(result.trace.trace_id)   # 按当时的事件序号还原检索与判断
    await rag.adelete(["guideline/中国高血压防治指南@2018"])
    await rag.aclose()

asyncio.run(main())
```

需要替换组件时，可以跳过组装根，直接注入：

```python
from medrag.llm import OpenAICompatEmbedding, OpenAICompatLLM
from medrag.prompts import PromptRegistry
from medrag.storage.graph import GraphDatabase
from medrag.storage.graph_projection import CypherGraphProjection
from medrag.storage.knowledge_store import SqlKnowledgeStore
from medrag.storage.milvus import MilvusVectorStorage
from medrag.storage.sql import SqlDatabase

rag = MedRAG(
    workspace="cardiometabolic_zh",              # 映射为 collection 名、表名前缀、图名；一个 workspace 一种语言（D16）
    llm=OpenAICompatLLM(cfg.llm),
    embedding=OpenAICompatEmbedding(cfg.embedding.text_dense),
    vectors=MilvusVectorStorage(cfg.milvus),     # 三个 collection：chunks / claims / aliases
    store=SqlKnowledgeStore(SqlDatabase(cfg.postgres)),       # 真相源；未配 postgres.host 时是 SQLite
    graph=CypherGraphProjection(GraphDatabase(cfg.neo4j)),    # 图投影；未配 neo4j.uri 时是 FalkorDBLite
    sparse=MilvusBM25Encoder(analyzer="jieba"),
    prompts=PromptRegistry.from_config(cfg.prompts),
    config=cfg.knowledge,                        # 只传自己用到的那一段配置
)
```

| 方法 | 作用 | 出现的里程碑 |
| --- | --- | --- |
| `initialize` / `aclose` | 建表、建索引、建图约束（幂等）；释放连接 | M1 |
| `ainsert` / `adelete` | 增量入库（带版本语义）/ 撤下文档 | M1（Claim 部分在 M3–M5） |
| `aretrieve` | 只检索文本块 | M1 |
| `aclaims` | 按概念、谓词、时间点取 Claim | M3（层级扩展 M4，有效期 M5） |
| `aconflicts` | 列出冲突集 | M5 |
| `aquery` | 按条件作答 | M6 |
| `averify` | 核查一句话是否有依据 | M6 |
| `areplay` | 审计回放 | M6 |
| `agovern` | 处理后台任务 | M1 起处理 `project_graph`；阶段二起处理查询驱动的任务 |

---

## 9. 应用层（`apps/`）

- **已有组件继续保留**：`SimpleRAG`、`MultiDialogueRag`、`MedicalAgent` / `SearchGraph`（LangGraph）。会话状态存在 Redis，组件本身无状态；流式接口原样透传 LangChain 的 `astream_events(v2)` 事件，由外部转换成 SSE / WebSocket。阶段二改为调用 core 的 `aquery`。
- **core 作为 Agent 的工具**：`claims_lookup`、`verify`、`conflicts`、`dose_check` 等。工具只接收领域参数，检索计划由调用方固定（D10）。
- **安全策略只在这一层**（D30）。面对同一个 `QueryResult`，医生助手和患者助手的处理方式不同：

| 结果 | 医生助手 | 患者助手 |
| --- | --- | --- |
| `need_info` | 列出缺少的条件 | 用通俗的话追问；问不到就建议就医 |
| 学术分歧型冲突 | 呈现双方，附机构和证据等级 | 说明存在不同标准，建议遵医嘱 |
| 剂量、处方类内容 | 给出原文和版本 | 不给剂量，提示咨询医生或药师 |
| 急症红旗 | 提示紧急处理要点 | 立即给出就医指引，不再继续问答 |

- 阶段三新增医生助手和患者助手，并适配 `software/` 的前后端（D11）。

---

## 10. 模型层（`llm/`，已实现，D2）

- 统一使用 `openai` 官方 SDK（`AsyncOpenAI`）走网络请求。`base_url`、`api_key_env`、`model`、超时、重试和并发都可以配置。DashScope、vLLM、Ollama（`/v1` 端点）、OpenAI 都按 OpenAI 兼容协议接入。以后接入其他模型商，只需再实现一个满足 `LLMFunc` / `EmbeddingFunc` 协议的类。
- 以下逻辑统一收拢在 `llm/`：
  - 去掉思考内容（`<think>…</think>` 和 `reasoning_content`）；
  - 以接口返回的 `usage` 统计 token；
  - Embedding 按 `max_batch` 分批（DashScope 单次最多 10 条）；
  - 用信号量控制并发。
- 抽取和判定类的调用用 `json_schema` 约束输出，便于校验。
- `apps/` 里的 LangGraph 需要 LangChain 的 ChatModel，由 `llm/langchain_adapter.py` 用**同一份配置**构造 `ChatOpenAI`，保证两边连的是同一个模型。

---

## 11. 提示词（`prompts/`，已实现，D9、D15）

**原则**：
- 提示词是数据：每个版本单独存一个文件，可以在 PR 里看 diff。
- 已发布（released）的版本不可修改，要改就新建一个版本。改动已发布版本的文件，测试会失败。
- 代码只按名称取提示词，用哪个版本由配置决定。
- 每次调用都记录 `name@version#sha256`，写进 trace 和 RunManifest。
- 中英文分开维护，找不到对应语言时直接报错，不会退回另一种语言。

**文件布局**：

```
src/medrag/prompts/templates/<名称>/
├── _meta.yaml        # 说明、变量定义（缺少必填变量或传了未声明的变量都会报错）、别名（stable / latest）
├── v1.zh.yaml        # version / parent / status（draft | released | deprecated）/ changelog / params / messages
└── v2.zh.yaml
```

- **模板语法**：新版本默认用 Jinja2 + StrictUndefined，变量缺失时直接报错，JSON 示例不用转义花括号。从 v2 迁移的 15 个提示词是 `v1`，标注 `format: fstring`，与 v2 原文逐字一致（`tests/test_prompts.py` 按 sha256 比对）。
- **知识层新增的提示词**：`claim_extract`、`concept_link`、`conflict_type`、`query_understand`、`answer_with_claims`、`verify_sentence`。一律用 Jinja2，从 draft 开始迭代。
- **用法**：

```python
prompt = registry.get("claim_extract", lang="zh")      # 版本由配置决定
messages = prompt.render(doc=doc, chunks=chunks, schema=schema)
result = await llm(messages, json_schema=CLAIM_SCHEMA, **prompt.params)
trace.prompts.append(prompt.ref)                        # PromptRef(name, version, lang, sha256)
```

- **配置**：`prompts.versions: {claim_extract: v2}`；自定义模板目录放在 `prompts.dirs`，同名提示词优先使用自定义的。
- **命令行**：已有 `python -m medrag.prompts list | show | check | lock`；`diff | new | release` 随 `cli.py` 补齐，届时统一为 `medrag prompts ...`。
- **迭代流程**：
  1. 新建 draft 版本，在 **dev 集**上反复修改、评测；
  2. 用 `eval compare` 与 stable 版本对比，主要指标的提升超出置信区间、逐题检查也没有系统性退化，才 release；
  3. release 之后把 `stable` 指过去，在 test 集上跑一次作为正式结果，并执行 `python -m medrag.prompts lock`。

  不要在 test 集上调提示词。

---

## 12. 配置（`config/`，已实现）

配置按领域拆成多个 YAML，每一段都可以缺省：

| 文件 | 内容 |
| --- | --- |
| `storage.yaml` | `milvus`、`redis`、`postgres`、`neo4j`（默认都不配地址，即降级模式） |
| `models.yaml` | `llm`、`embedding` |
| `prompts.yaml` | `prompts` |
| `data.yaml` | v2 的入库字段映射（M1 起由各数据集的 loader 取代） |
| `dialogue.yaml`、`agent.yaml` | 多轮对话、Agent 策略 |
| `knowledge.yaml`（M1 起） | `workspace`、`lang`；`extract`（哪些 doc_type 和章节要抽 Claim、用哪个模型）；`concepts`（链接阈值）；`governance`（自动合并门槛、高风险类别）；`retrieval`（默认的 `QueryParam`） |

```python
ConfigLoader()                                    # 读取包内 config/，或环境变量 MEDRAG_CONFIG_DIR 指定的目录
ConfigLoader("my_conf/")                          # 读取自己的配置目录
ConfigLoader(overrides=["deploy/storage.yaml"])   # 默认配置 + 整段替换某几段
# 环境变量 MEDRAG_CONFIG_OVERRIDES 可以追加覆盖文件（多个用 : 分隔）
```

- 密钥只写环境变量名（`api_key_env`，默认 `DASHSCOPE_API_KEY`），不写明文。
- 中间件的密码由 `deploy/start.sh` 生成到 `deploy/.env`，`ConfigLoader` 会自动加载。
- `scripts/` 读取 `scripts/conf/` 下自己的一套配置，经由 `scripts/_common.py` 加载。

---

## 13. 评测与可复现

**RunManifest**：每次评测都写出一份，和结果文件放在一起（`runs/<dataset>/<run_id>/`）。

```yaml
run_id: 20261101-1432-a1b2
git: {commit: 9a404c6, dirty: false}
dataset: {name: cardiometabolic_zh, subset: timesliced_v1, split: dev, manifest_sha: …}
prompts: {claim_extract: v1#3fa2…, answer_with_claims: v1#9c01…}
models: {llm: qwen3.5-plus, embedding: text-embedding-v4, judge: …}
knowledge: {schema: v1, governance_seq: 18234, graph_seq: 18234}
config_sha: …          # 检索、切块、抽取等全部配置的哈希
chunker: {name: section_paragraph, version: v1}
seed: 42
```

`eval compare <run_a> <run_b>` 会输出：
- 各指标的差值和 95% 置信区间；
- 分组对比；
- 逐题差异：前对后错、前错后对的题目清单。

只要两次运行的 manifest 有字段不同，报告开头就会列出这些不同项，避免把数据或提示词的变化误当成算法本身的提升。

**评测集与指标**：

| 评测 | 数据 | 指标 | 作用 |
| --- | --- | --- | --- |
| 时间切片评测（主） | 切片语料按发布时间分批入库；300～500 题，每题有"截至某日"的答案和出处版本 | 按时间点作答的正确率；条件类问题的追问率、按条件作答的正确率；分歧类是否呈现双方；错误信息类是否排除 | 论文主结果 |
| Claim 抽取 | 金标准约 300 条 | 严格 / 宽松匹配下的准确率、召回率、F1；条件的完整度；证据对齐率 | M3 门槛 |
| 概念链接、冲突分型 | 标注集；CHIP-CDN 作为外部参照 | 准确率 | M4、M5 门槛 |
| claim 级忠实度 | 回答逐句拆分 | 有依据的比例、无依据率、矛盾率、安全关键错误率 | M6、M7 |
| A 层回归 | huatuo QA：198 题，加上 9,802 篇干扰文档组成 1 万篇语料（seed=42）；用 `new_question` 检索，以 `question_hash` 判定命中 | Hit@k、MRR | 改动存储或文本块检索时的门槛（D17） |
| 静态知识对照 | MedQA 中文里与切片相关的题目 | 选择题准确率 | 确认没有退化 |
| 英文对标（阶段二） | GraphRAG-Bench medical、DriftMed | 官方指标 | 对标 LightRAG |

**原则**：
- dev 集用于迭代，test 集用于验收。提示词和检索参数都不在 test 集上调。
- 回归门槛分两层（D17）：
  - 硬门槛看代码行为是否一致（逐行比对数据和查询编码）；
  - 软门槛要求检索指标的差异，不超过同一实现重跑的波动再加 2pp。

  Milvus 的 HNSW 检索加上并发写入，同一实现重跑的 top-10 重合率只有约 91%，所以不用"重合率接近 100%"这类标准。
- 数据子集由固定种子的脚本生成，并附带 manifest。

---

## 14. 测试策略

| 类型 | 依赖 | 说明 |
| --- | --- | --- |
| 单元测试 | 无 | 存储走降级实现，数据文件放在 pytest 的临时目录；模型走 `medrag.testing.fake_openai`，按文本哈希生成向量，M3 起还能按提示词名返回预置 JSON，用来测抽取和判定链路。不需要网络和 Docker |
| 集成测试 | Milvus / Redis / PostgreSQL / Neo4j | 标记为 `@pytest.mark.services(需要的服务...)`，不写参数时默认是 milvus + redis。服务不可用时自动跳过；设置 `MEDRAG_REQUIRE_SERVICES=1` 时必须执行 |
| 双后端一致性 | 同上 | 关系库（SQLite / PostgreSQL）、图投影（FalkorDBLite / Neo4j）、向量（Milvus Lite / Milvus）在两种后端上跑同一组断言 |
| 分层检查 | 无 | `tests/test_layering.py`：核心层不 import LangChain / LangGraph |
| 提示词 | 无 | 已发布版本未被修改；v1 与 v2 原文逐字一致 |
| 回归 | 真实 Embedding 服务 | A 层 Hit@k；`scripts/regression/` 下按里程碑提供回归脚本 |

---

## 15. 现状与差距

M0 的结果：
- 包名已改为 `medrag`，分层与依赖规则已经生效；
- 模型层、提示词、配置、四个服务的降级都已实现；
- 测试全部通过（69 个），端到端、会话、SQL、图的用例在降级实现和真实服务上各跑了一遍；
- A 层回归：`verify` 比对数据层 9,999 行、查询编码 198 题 × 3 路，均为 0 差异；检索指标在噪声范围内。

| 目标 | 现状 | 计划 |
| --- | --- | --- |
| `types.py` 中第 4 节的数据模型 | 只有过渡类型 `LegacyDocument`（字段与 LangChain Document 一致：`page_content` / `metadata`） | M1 |
| `base.py`、`medrag.py`、`build.py` | 尚未创建 | M1 |
| `MilvusVectorStorage`（三个 collection） | 仍是 v2 的 `MedicalHybridKnowledgeBase`：单行多向量布局；主键是 `md5(summary)`，summary 为空时会撞车；内部自己创建 embedding 客户端 | M1 |
| `knowledge_store.py`、`graph_projection.py` | 尚未创建；底层的 `SqlDatabase`、`GraphDatabase` 已实现，但还没有调用方 | M1 |
| `SqlDatabase` 的事务 | 每条语句自动提交 | M1 |
| `ingest/` | `pipeline.py` 是 v2 形态：只区分 qa 与其他来源，QA 文本的中文前缀写死 | M1 重写；M2 增加 parsers |
| `retrieval/` | 只有 `sparse/`（pkuseg 词表的 BM25）；融合在 Milvus 内完成；上下文组装在 `apps/memory.py` 的 `TokenBudget` 里 | M1 增加融合与 rerank；M6 增加 Claim 通道 |
| `extract/`、`concepts/`、`governance/`、`answer/`、`guard/`、`datasets/`、`cli.py` | 尚未创建 | M2–M6 |
| `eval/` | 已有 `runner.py`、`adapters.py`、`scorers/`（ragas） | M2 增加 RunManifest；M7 增加时间切片评测 |
| `apps/` | `rag_base.py`、`simple_rag.py`、`dialogue_rag.py`、`memory.py`、`retriever.py`、`chat_history.py`、`timing.py`、`agent/` 都可用，直接使用 `MedicalHybridKnowledgeBase` | 阶段二改为调用 `aquery` |
| `scripts/` | 01–08 使用示例基于 v2 形态的接口；`regression/m0_parity.py` 是 M0 一致性回归 | 随门面类逐步更新 |
| `software/` | 冻结；backend 用的是异步化之前的接口，目前跑不起来 | 阶段三（D11） |

**过渡期规则**：M1 完成之前，检索结果仍用 `LegacyDocument`，知识库仍是 v2 形态的 `MedicalHybridKnowledgeBase`。新代码不要在这些过渡类型上继续扩展功能，扩展一律放到 M1 的数据模型和存储抽象上。
