# 架构设计

> 本文定义 medrag 的结构、技术方案和用法。项目要做什么、为什么这样做、往哪里走，见 [solution.md](solution.md)。
> 本文描述的是目标形态，并标注哪些部分「已有」、哪些「待建」；当前代码与目标的差距见第 16 节。开发阶段的划分另行讨论。

---

## 1. 设计原则

第 1～6 条是搬家（v2 → medrag）时已经落地的公共基础，新代码必须遵守；第 7～10 条是知识层新增的约定。

1. **只用异步**：IO 方法一律写成 `async def`，不保留同步版本。CPU 密集的操作（pkuseg 分词、PDF 解析）和只有同步驱动的库（SQLite）用 `asyncio.to_thread`。
2. **依赖注入**：存储、模型客户端、提示词注册表、领域包都由调用方创建后传入，组件内部不自己创建；组件只接收自己用到的那一段配置，不接收整个 `AppConfig`。按配置组装默认组件的事，只在组装根（`build.py`）里做。
3. **模型层统一走 OpenAI 兼容协议**：LLM 和 Embedding 都用 `openai` SDK（`AsyncOpenAI`）访问。核心层不 import LangChain / LangGraph，例外只有三处：`apps/`、`llm/langchain_adapter.py`、`eval/scorers/`。由 `tests/test_layering.py` 检查。
4. **存储细节不外泄**：对外只暴露 `QueryParam` 这类领域参数。Milvus 的 `anns_field`、`metric_type`、`expr` 只出现在 `storage/milvus.py`；Cypher 只出现在 `storage/graph*.py`；SQL 只出现在 `storage/` 下。这些细节都不交给 LLM 或上层应用。
5. **提示词放在文件里**：代码里不写提示词字符串，通过注入的 `PromptRegistry` 按名称获取（第 12 节）。
6. **服务可以不配地址**：Milvus、Redis、PostgreSQL、Neo4j 没配地址时，分别降级为 Milvus Lite、fakeredis、SQLite、FalkorDBLite（第 7.5 节）。新写的存储代码必须同时支持两种后端，并在两种后端上各测一遍。
7. **内核领域无关**：内核代码里不出现医学概念，包括实体类型名、谓词名、术语体系、具体的工具。这些都由医疗领域包提供（第 5 节）。
8. **关系库是唯一真相源**：文档、Claim、概念、治理事件都以关系库为准。Milvus 和图数据库是派生索引，随时可以从关系库重建。
9. **回答路径只读**：查询不写知识库。写入只经由入库流程或后台治理，每次写入都伴随一条治理事件。
10. **可复现**：每次回答的 trace 都记录提示词版本、模型、命中的 Claim 和当时的治理事件序号；每次评测都写出 RunManifest（第 14 节）。

---

## 2. 分层

整体分三层：

```
┌──────────────────────────────────────────────────────────────────────┐
│ 应用层  apps/                                                          │  领域相关
│   医生助手 · 患者助手 · 安全策略 · SimpleRAG · DialogueRAG · MedicalAgent │  可以依赖 LangChain / LangGraph
├──────────────────────────────────────────────────────────────────────┤
│ 领域包  domains/medical/                                               │  领域相关
│   schema · 文档档案 · 术语导入 · 来源优先级 · 冲突规则 · 风险规则 ·        │  由组装根注入内核
│   条件判定 · 确定性工具 · 提示词                                         │
├──────────────────────────────────────────────────────────────────────┤
│ 内核（其余所有模块）                                                     │  领域无关
│   门面   medrag.py：ainsert / aquery / aretrieve / aclaims / averify /    │
│          aconflicts / agovern / areplay / adelete                     │
│   能力   ingest · extract · concepts · governance · retrieval · answer · guard │
│   基础   storage · llm · types · base · config · prompts · eval         │
└──────────────────────────────────────────────────────────────────────┘
```

**依赖规则**：

1. **内核**不 import `domains/` 和 `apps/`。内核通过 `base.py` 中的 `DomainPack` 接口使用领域包，也不按领域名分支（不写 `if domain == "medical"`）。
2. **领域包**只依赖内核的数据模型和接口（`types.py`、`base.py`），不依赖存储实现。
3. **应用层**可以依赖内核和领域包。
4. **内核内部**依赖方向只能是门面 → 能力 → 基础。能力模块之间只允许以下依赖，不能成环：
   - `extract/`、`concepts/`、`retrieval/`、`guard/` 只依赖基础层；
   - `governance/` 可以依赖 `extract/`、`concepts/`；
   - `ingest/` 是入库的编排者，可以依赖 `extract/`、`concepts/`、`governance/`；
   - `answer/` 可以依赖 `retrieval/`、`guard/`。
5. **安全策略**（拦截、改写、转人工、按受众调整）只放在应用层。内核和领域包只产出事实和风险信号。

前两条规则和检查 LangChain 依赖的做法一样，由 `tests/test_layering.py` 检查。

---

## 3. 目录结构

```
src/medrag/
├── __init__.py                 # 已有；公共 API 收拢到这里：MedRAG、QueryParam、Document、Claim……
├── medrag.py                   # 待建：门面类
├── build.py                    # 待建：组装根，按配置创建默认组件并注入领域包（未配地址时自动降级）
├── types.py                    # 已有（只有过渡类型 LegacyDocument）；改为第 4 节的数据模型
├── base.py                     # 待建：抽象接口（第 6 节），包括 DomainPack
├── config/                     # 已有：配置模型与多 YAML 加载器；新增 knowledge 段
├── llm/                        # 已有：openai_compat.py、langchain_adapter.py、utils.py
├── prompts/                    # 已有：registry.py、templates/、prompts.lock；放领域无关的通用模板
├── storage/
│   ├── backends.py             # 已有：未配地址时的降级选择，选择逻辑只在这里
│   ├── milvus.py               # 已有（v2 的 MedicalHybridKnowledgeBase）；改造为 MilvusVectorStorage
│   ├── milvus_io.py            # 已有：批量写入辅助；之后并入 milvus.py
│   ├── sql.py                  # 已有：SqlDatabase（PostgreSQL / SQLite）；需要增加事务
│   ├── graph.py                # 已有：GraphDatabase（Neo4j / FalkorDBLite），执行 Cypher
│   ├── redis_session.py        # 已有：会话存储（Redis / fakeredis）
│   ├── knowledge_store.py      # 待建：关系库上的文档、Claim、概念、冲突集、事件、任务（真相源）
│   └── graph_projection.py     # 待建：消费治理事件，维护图投影（两种后端共用的 openCypher 子集）
├── ingest/
│   ├── pipeline.py             # 已有（v2 形态）；重写为带版本语义的增量入库编排
│   ├── chunkers.py             # 待建：QAPairChunker / SectionParagraphChunker / TokenChunker
│   └── parsers/                # 待建：PDF / HTML → 带章节的文本（通用部分）
├── extract/                    # 待建：按领域 schema 抽取 Claim、证据对齐
├── concepts/                   # 待建：概念链接（别名表 → 向量 → LLM）、概念层级
├── governance/                 # 待建：有效期、冲突检测与分型、合并、任务队列、worker
├── retrieval/
│   ├── sparse/                 # 已有：pkuseg 自管理词表的 BM25；待加 Milvus 托管 BM25
│   ├── fusion.py               # 待建：RRF / 加权融合
│   ├── rerank.py               # 待建：BaseReranker 与一个 OpenAI 兼容的实现
│   ├── claims.py               # 待建：按概念取 Claim、层级扩展、条件匹配
│   └── context.py              # 待建：按 token 预算组装上下文、给证据编号
├── answer/                     # 待建：查询理解（概念 + 条件）、作答、逐句核查
├── guard/                      # 待建：风险信号、工具注册（具体工具在领域包里）
├── domains/
│   └── medical/                # 待建：医疗领域包（第 5 节）
├── datasets/                   # 待建：下载与校验、清洗规则、子集与 manifest、loaders/
├── eval/                       # 已有：runner.py、adapters.py、scorers/；待加 RunManifest、时间切片评测
├── apps/                       # 已有：应用层（SimpleRAG、DialogueRAG、memory、agent/）
├── tools/                      # 已有：数据标注等非核心工具
├── testing/                    # 已有：fake_openai.py；待加"按提示词名返回预置 JSON"
└── cli.py                      # 待建：medrag data | ingest | govern | eval | prompts
```

包外的目录：

| 目录 | 内容 |
| --- | --- |
| `scripts/` | 使用示例和回归脚本 |
| `tests/` | 单测与集成测试 |
| `deploy/` | 四个服务的 docker compose；`deploy/storage.yaml` 是连接它们的配置 |
| `software/` | 前后端示例应用，冻结，到产品阶段再适配 |
| `.medrag/` | 降级实现的本地数据，不进 git |

---

## 4. 数据模型（`types.py`）

数据模型领域无关。凡是取值随领域变化的字段（实体类型、谓词、条件变量、限定符、文档类型、权威度），在内核里都是字符串，由领域包的 schema 校验。

```python
class Document(BaseModel):
    """入库的最小单位：一个 QA 对、一本教材、一份指南的一个版本、一份说明书的一个版本"""
    doc_id: str                         # 一个具体版本的唯一 ID，如 "guideline/中国高血压防治指南@2024"
    work_id: str = ""                   # 同一作品的各个版本共用；缺省等于 doc_id（没有版本概念的文档）
    version: str = ""                   # "2024年修订版"、"2017-03 修订"
    content: str
    doc_type: str                       # 由领域包的文档档案定义，如医疗的 guideline / drug_label / textbook
    lang: Literal["zh", "en"]           # 必须与 workspace 的语言一致
    title: str = ""
    publisher: str = ""                 # 发布机构
    published_at: date | None = None    # 发布 / 修订日期
    effective_from: date | None = None  # 开始生效的日期，缺省取 published_at
    supersedes: list[str] = []          # 被本版本取代的旧版本 doc_id；不同机构的文档之间不是取代关系
    region: str = ""                    # 适用地区，如 "CN"
    authority: str = ""                 # 权威度等级，如医疗的证据等级；排序规则由领域包定义
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
    concept_id: str                     # 锚定时用 "<编码体系>:<编码>"，如 "ATC:A10BA02"；未锚定时为 "local:<hash>"
    type: str                           # 由领域 schema 定义，如医疗的 ingredient / disease / lab_test
    name: str                           # 规范名
    aliases: list[str] = []
    codes: dict[str, str] = {}          # 各编码体系下的编码
    anchored: bool                      # 允许未锚定的概念存在，但必须标记
    merged_into: str | None = None      # 被合并后指向保留的概念；原记录不物理删除，方便拆回


class ConceptRelation(BaseModel):
    """粒度不同的概念之间用关系连接，不合并"""
    src: str
    rel: str                            # 由领域 schema 定义（如 is_a、has_ingredient）；内核保留 same_as_candidate
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
    variable: str                       # 领域 schema 中声明的条件变量，如医疗的 eGFR、pregnancy、age
    op: Literal["<", "<=", ">", ">=", "between", "=", "in", "present", "absent"]
    value: float | list[float] | str | list[str] | None = None
    unit: str = ""
    text: str                           # 原文措辞；结构化失败时只保留它，此时不参与自动匹配


class Claim(BaseModel):
    claim_id: str
    subject: str                        # concept_id
    predicate: str                      # 由领域 schema 定义，如医疗的 contraindication、dosage
    object: str | None = None           # concept_id；剂量、控制目标这类陈述可以没有客体
    value: dict[str, Any] = {}          # 数值内容，结构由 schema 中该谓词的定义决定
    conditions: list[Condition] = []
    polarity: Literal["affirm", "negate"] = "affirm"
    modality: str | None = None         # 该谓词允许的模态，如推荐的强弱、禁忌的绝对 / 相对
    certainty: Literal["certain", "possible", "uncertain"] = "certain"
    qualifiers: dict[str, Any] = {}     # 领域限定符，如不良反应的发生频率、原文的推荐级别
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
                  "authority", "misinformation", "extraction_error"]
    resolution: str = ""                # 处理结论，如"新版本 preferred，旧版本 deprecated"
    status: Literal["open", "resolved"] = "open"
    reviewed_by: str = ""


class GovernanceEvent(BaseModel):
    """只追加。图投影、审计回放、回滚、领域插件的订阅都基于它"""
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
| 新的 `doc_id`，`supersedes` 指向旧版本 | 新版说明书、新版指南 | 修订：旧版本标记为 superseded，**不删除**；冲突检测把旧版本的 Claim 标为 deprecated 并写上 `valid_to` |
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
    context: dict[str, Any] = {}        # 用户给出的条件，键是 schema 中的条件变量，如 {"eGFR": 40, "pregnancy": False}
    expand_hierarchy: bool = True       # 沿可继承关系扩展概念集合
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

## 5. 领域包（`domains/medical/`）

领域包回答两个问题：这个领域的知识长什么样（schema），这个领域的规则是什么（优先级、冲突、条件判定、工具）。能用 YAML 声明的就声明，声明不了的才写成插件。内核对每个切面都有默认行为，领域包只在默认行为不够用时才提供插件。

**扩展点只开医疗实际用到的**，没有实际内容的扩展点一律不加：

| 切面 | 形式 | 内核在哪里用到 | 医疗领域包的内容 |
| --- | --- | --- | --- |
| schema | `schema.yaml` | 生成抽取用的 JSON Schema、入库校验、条件匹配、图扩展 | 实体类型、谓词（含主宾类型约束、允许的模态和限定符）、条件变量与单位、可继承关系 |
| 文档档案 | `doc_profiles.yaml`，必要时加解析器插件 | 入库：切块、决定哪些章节抽 Claim | 说明书按【禁忌】【用法用量】【不良反应】等章节切分；指南抽推荐条目；教材和问答只切块，不抽 Claim |
| 术语导入 | 插件 | 概念链接、概念层级 | ICD-10、ATC、医保药品编码、LOINC 的导入器 |
| 来源优先级 | `policies.yaml` | 排序；判定"权威度不同"型冲突 | 证据等级：系统综述 / Meta 分析 > RCT > 队列研究 > 病例报告 > 专家意见；文档类型：指南、说明书 > 教材 > 问答 |
| 冲突规则 | `policies.yaml` | 冲突检测后的处理 | 时效型自动标 deprecated；学术分歧只并列呈现；错误信息降权并进审核队列 |
| 风险规则 | `policies.yaml` | 产出风险信号 | 高风险谓词（剂量、禁忌、相互作用）；红旗症状 |
| 条件判定 | 插件（按条件变量注册） | 条件匹配；没注册的变量用内核默认的数值 / 类别比较 | 单位换算（如血肌酐 μmol/L 与 mg/dL）；年龄段、妊娠期 |
| 确定性工具 | 插件 | 注册到 `guard/`，供作答和应用层调用 | 剂量与肾功能调整、相互作用核对 |
| 提示词 | 模板目录 | 抽取、链接、作答；覆盖内核的通用模板 | 医学的抽取指令与示例 |

需要在某个时机插入领域逻辑时（比如 Claim 抽出后做领域校验、发现冲突后通知审核），不另造 AOP 机制，由领域插件订阅治理事件即可。

**目录**：

```
domains/medical/
├── __init__.py          # MedicalDomainPack：加载下面的声明文件，注册插件
├── schema.yaml
├── doc_profiles.yaml
├── policies.yaml
├── vocab/               # 术语导入器
├── evaluators.py        # 条件判定插件
├── tools/               # 确定性工具
└── prompts/             # 医学提示词（覆盖通用模板）
```

**schema 示意**（节选；首版清单要用金标准来定稿）：

```yaml
entity_types: [disease, symptom, ingredient, product, drug_class, lab_test, procedure, population, organism, anatomy]

relations:                        # inherit_claims：Claim 是否沿这条关系向下继承
  is_a:           {inherit_claims: true}    # 依那普利 is_a ACEI：类别级的禁忌适用于依那普利
  has_ingredient: {inherit_claims: true}    # 制剂继承成分的 Claim
  brand_of:       {inherit_claims: true}

predicates:
  contraindication: {subject: [ingredient, product, drug_class, procedure], object: [disease, population],
                     modality: [absolute, relative]}
  recommendation:   {subject: [ingredient, drug_class, procedure, lab_test], object: [disease, population],
                     modality: [strong_for, weak_for, weak_against, strong_against], qualifiers: [evidence_grade]}
  adverse_effect:   {subject: [ingredient, product, drug_class], object: [symptom, disease], qualifiers: [frequency]}
  dosage:           {subject: [ingredient, product], value: {dose: number, unit: string, freq: string}}
  # indication / caution / interaction / dose_adjustment / monitoring / diagnostic_criterion / treatment_target ……

conditions:
  eGFR:      {type: numeric, unit: "mL/min/1.73m²", concept: lab_test/eGFR}
  age:       {type: numeric, unit: 岁}
  pregnancy: {type: presence, concept: population/妊娠}

qualifiers:
  frequency:      {enum: [十分常见, 常见, 偶见, 罕见, 十分罕见]}
  evidence_grade: {type: string}            # 原文给出的推荐级别或证据等级
```

---

## 6. 核心接口（`base.py`）

```python
class LLMFunc(Protocol):                         # 已有实现：llm/openai_compat.py 的 OpenAICompatLLM
    model: str
    async def __call__(self, messages: list[Message], *, json_schema: dict | None = None, **params) -> LLMResult: ...
    def stream(self, messages: list[Message], **params) -> AsyncIterator[LLMDelta]: ...

class EmbeddingFunc(Protocol):                   # 已有实现：OpenAICompatEmbedding（分批、并发、保持顺序）
    model: str
    dim: int
    async def __call__(self, texts: list[str]) -> list[list[float]]: ...

class DomainPack(Protocol):                      # 第 5 节
    name: str
    schema: DomainSchema
    doc_profiles: dict[str, DocProfile]
    policies: DomainPolicies                     # 来源优先级、冲突规则、风险规则
    vocabularies: list[VocabularyImporter]
    condition_evaluators: dict[str, ConditionEvaluator]
    tools: list[DomainTool]
    prompts_dir: Path

class BaseSparseEncoder(ABC):                    # pkuseg 自管理词表 / Milvus 托管 BM25
    managed_by_store: bool                       # True 时由向量库自己计算，encode 不会被调用
    async def encode_documents(self, texts: list[str]) -> list[dict[int, float]]: ...
    async def encode_query(self, text: str) -> dict[int, float]: ...

class BaseVectorStorage(ABC):                    # Milvus / Milvus Lite；一个 workspace 三个 collection（第 7.2 节）
    async def initialize(self) -> None: ...      # 建表、建索引（幂等）
    async def upsert(self, kind: IndexKind, records: list[IndexRecord]) -> None: ...   # kind: chunk / claim / alias
    async def delete(self, kind: IndexKind, ids: list[str]) -> int: ...
    async def search(self, kind: IndexKind, query: QueryVectors, param: QueryParam) -> list[Hit]: ...
    async def drop(self) -> None: ...
    async def close(self) -> None: ...

class KnowledgeStore(ABC):                       # 关系库上的真相源（第 7.1 节）
    documents: DocumentRepo                      # 文档、版本链、入库状态
    chunks: ChunkRepo
    concepts: ConceptRepo                        # 概念、别名、概念关系、提及
    claims: ClaimRepo                            # Claim、条件、证据；按概念 / 谓词 / 有效期 / rank 查询
    conflicts: ConflictRepo
    events: EventLog                             # 只追加；按 seq 读取
    tasks: TaskQueue                             # 后台任务
    def transaction(self) -> AsyncContextManager: ...   # 状态变更和对应的事件在同一个事务里写入

class GraphProjection(ABC):                      # Neo4j / FalkorDBLite（第 7.3 节）
    async def apply(self, events: list[GovernanceEvent]) -> int: ...       # 幂等；返回投影到的最新 seq
    async def rebuild(self, store: KnowledgeStore) -> None: ...            # 清空后按事件重放
    async def expand(self, concept_ids: list[str], rels: list[str], max_hops: int) -> list[str]: ...
    async def paths(self, src: str, dst: str, max_hops: int) -> list[GraphPath]: ...
    async def last_seq(self) -> int: ...

class BaseParser(ABC):                           # PDF / HTML → 带章节的文本
    async def parse(self, raw: bytes, *, mime: str) -> ParsedDocument: ...

class BaseChunker(ABC):
    name: str; version: str                      # 写进 chunk 元数据和运行记录；切块策略一变，评测结果就不可比
    def chunk(self, doc: Document) -> list[Chunk]: ...

class BaseClaimExtractor(ABC):                   # 按 DomainSchema 生成的 JSON Schema 抽取
    async def extract(self, doc: Document, chunks: list[Chunk]) -> list[ClaimDraft]: ...   # 草稿：尚未链接概念

class BaseConceptLinker(ABC):
    async def link(self, mentions: list[str], *, type_hint: str | None = None) -> list[LinkResult]: ...

class ConflictDetector(ABC):                     # 分组与分型由内核完成，处理方式查 DomainPolicies
    async def detect(self, new: list[Claim], store: KnowledgeStore) -> list[ConflictSet]: ...

class BaseReranker(ABC):
    async def rerank(self, query: str, items: list[Hit], top_k: int) -> list[Hit]: ...
```

---

## 7. 存储

### 7.1 关系库：唯一真相源（PostgreSQL / SQLite）

基于已有的 `storage/sql.py` 中的 `SqlDatabase`。表名都带 workspace 前缀：

| 表 | 内容 |
| --- | --- |
| `documents`、`doc_status` | 文档正文与元数据、版本链（`work_id`、`supersedes`）、入库状态（pending / processing / processed / failed） |
| `chunks` | 文本块正文与字符偏移（向量在 Milvus） |
| `concepts`、`concept_aliases`、`concept_relations`、`mentions` | 概念、别名、粒度关系、原文提及 |
| `claims`、`claim_conditions`、`claim_evidence` | Claim 本体、结构化条件（便于区间查询）、证据区间 |
| `conflict_sets`、`conflict_members` | 冲突集与成员 |
| `events` | 治理事件，只追加，`seq` 单调递增 |
| `tasks` | 后台任务队列：`project_graph` / `extract_uncovered` / `judge_merge` / `recheck_conflict` / `human_review` |

约定：
- 沿用 `SqlDatabase` 的规则：统一用 `?` 占位符；只用两种数据库都支持的语法（`CREATE TABLE IF NOT EXISTS`、`INSERT ... ON CONFLICT ... DO UPDATE`）；SQLite 走 `asyncio.to_thread`。
- `SqlDatabase` 目前每条语句都自动提交，需要增加事务（`async with db.transaction()`）。状态变更和对应的事件必须在同一个事务里写入。
- 表结构领域无关：领域相关的值（实体类型、谓词、条件变量）都按字符串存储，由领域 schema 校验。
- 有效期查询（`as_of`）、按条件区间筛选、冲突分组，都在这一层用 SQL 完成。

### 7.2 向量索引：派生（Milvus / Milvus Lite）

一个 workspace 有三个 collection：

| collection | 主键 | 向量 | 用于过滤的标量字段 | 用途 |
| --- | --- | --- | --- | --- |
| `{ws}_chunks` | `chunk_id` | `content_dense`、可选的 `summary_dense`、`content_sparse` | `doc_id`、`doc_type`、`section`、`status`（是否已被取代） | 文本块通道，延续已有能力 |
| `{ws}_claims` | `claim_id` | Claim 文本化表述的稠密向量与稀疏向量 | `subject`、`predicate`、`rank`、`valid_from`、`valid_to` | Claim 的语义召回兜底 |
| `{ws}_aliases` | `alias_id` | 别名的稠密向量 | `concept_id`、`type` | 概念链接 |

- Claim 的文本化表述形如"二甲双胍｜不推荐起始用药｜eGFR 30–45 mL/min/1.73m²｜某说明书 第 3 版"。它只负责"找到"；条件是否满足，由关系库里的结构化字段判断。
- 稀疏检索有两种实现，放在同一个接口之后：pkuseg 自管理词表（医学分词质量好，但 IDF 在建词表时就冻结了）和 Milvus 原生 BM25（统计量由 Milvus 维护，天然支持增量）。最终选哪个，在文本块回归基准上做 A/B 实验后再定。
- 沿用搬家时实测得出的规则：
  - 批量写入后必须先 flush，再建索引、再检索；
  - VARCHAR 上限是 65,535 **字节**（中文约 2.1 万字），切块器必须保证不超过；
  - analyzer 按 workspace 的语言选择：中文用 jieba，英文用 standard。一个 workspace 只放一种语言。

### 7.3 图投影：派生（Neo4j / FalkorDBLite）

基于已有的 `storage/graph.py` 中的 `GraphDatabase`。图名或标签带 workspace 前缀。

```
(:Document {id, work_id, version, status})  -[:SUPERSEDES]->  (:Document)
(:Concept  {id, type, name, anchored})       -[:REL {type}]->  (:Concept)      # type 取自领域 schema 的 relations
(:Claim    {id, predicate, rank, valid_from, valid_to})
     -[:SUBJECT]-> (:Concept)   -[:OBJECT]-> (:Concept)   -[:CONDITION]-> (:Concept)
     -[:FROM]-> (:Document)     -[:CONFLICTS_WITH {type}]-> (:Claim)
```

- **怎么维护**：投影器按 `seq` 顺序消费治理事件，用幂等的 `MERGE` 写入，并记下已投影到的 `seq`。入库流程结束时运行一次，后台 worker 里也持续运行。任何时候都可以清空后按事件重放（`rebuild`）。
- **用来做什么**：
  - 类别继承：沿 schema 中 `inherit_claims: true` 的关系上溯，例如查"依那普利"时，带出"ACEI"这一类别级的 Claim；
  - 多跳查询：多药合用时的相互作用、合并症；
  - 解释路径：说明"为什么得出这个结论"；
  - 人工审核时的可视化。
- **语法限制**：FalkorDB 支持的是 openCypher 的子集，投影和查询只用两边都支持的语法，并在两种后端上各跑一遍测试。关系类型来自领域 schema，所以统一用带 `type` 属性的 `REL` 边，而不是为每种关系创建一个边标签。
- **一致性**：判断以关系库为准，图只负责扩展和路径。图的投影滞后时，trace 会同时记录关系库的 `seq` 和图已投影到的 `seq`。

### 7.4 会话（Redis / fakeredis）

原样沿用已有的 `RedisSessionStore`：key 规划、分布式锁和 TTL 都不变。多轮对话的历史和摘要、Agent 的跨轮状态都存在这里，所以应用层的组件本身无状态，可以多进程部署。fakeredis 的多个客户端要共用一个 `FakeServer` 才能共享数据。

### 7.5 未配置服务地址时的降级（已实现）

| 服务 | 判断依据 | 降级实现 | 数据位置 |
| --- | --- | --- | --- |
| Milvus | `milvus.uri` 为空 | Milvus Lite（pymilvus 在本进程内拉起） | `milvus.lite_path`，默认 `.medrag/milvus_lite.db` |
| Redis | `redis.host` 为空 | fakeredis（含 Lua，会话锁可用） | 进程内存，不跨进程，退出即丢失 |
| PostgreSQL | `postgres.host` 为空 | SQLite（标准库，经 `asyncio.to_thread` 调用） | `postgres.sqlite_path`，默认 `.medrag/medrag.sqlite3` |
| Neo4j | `neo4j.uri` 为空 | FalkorDBLite（嵌入式 FalkorDB，兼容 openCypher 子集） | `neo4j.falkordb_path`，默认 `.medrag/falkordb.db` |

- 选择逻辑只在 `storage/backends.py`。包内默认的 `storage.yaml` 不配地址，所以开箱就是降级模式；连接 `deploy/` 启动的服务时，用 `deploy/storage.yaml` 整段覆盖（`MEDRAG_CONFIG_OVERRIDES=deploy/storage.yaml`）。
- 每种服务第一次降级时打一条 warning，避免生产环境漏配地址后悄悄跑在嵌入式实现上。降级实现只适合单进程的开发、测试和小规模使用。
- 实测过的边界：
  - Milvus Lite 需要 pymilvus 2.6 + milvus-lite 3.x，分词器只有 `standard` 和 `jieba`，没有 Windows 版本；
  - redislite（FalkorDBLite）的异步 `close` 不会落盘，`GraphDatabase.close` 里会先显式 `SAVE`。

### 7.6 写入顺序与失败恢复

一次写入（入库或治理）按以下顺序进行：

1. 在关系库的一个事务里写入状态变更和事件；
2. 写向量索引；
3. 投影到图。

第 2、3 步失败时不回滚第 1 步，而是在 `tasks` 里登记重试。向量和图都是派生的，可以补写或重建，真相源始终保持一致。

---

## 8. 数据流

方括号里标注的是这一步用到的领域包切面。

### 8.1 入库（增量、幂等、带版本语义）

```
Document[]
  → 文档状态与版本判断
      content_hash 不变，且状态为 processed   → 跳过
      同一 doc_id 内容变化（更正）            → 撤回旧的文本块与 Claim（Claim 软撤回）后重新处理
      新 doc_id，且 supersedes 指向旧版本     → 旧版本标记为 superseded，不删除
  → 标记 processing
  → parser → chunker → 文本块写入关系库                              [文档档案]
  → embedding（分批 + 并发 + 重试） ┐ 两者并行
  → sparse.encode_documents（可选） ┘ → 写入 {ws}_chunks
  → 档案中标明要抽 Claim 的章节：                                     [文档档案]
      extractor（由 schema 生成的 JSON Schema 约束输出）                [schema、提示词]
      → 校验：类型约束、模态、限定符                                   [schema]
      → 证据对齐（原句 + 字符区间；对不上的丢弃并计数）
      → concept linker（别名表 → 向量 → LLM）                         [术语导入]
      → 在一个事务内写入 Claim 和事件
      → 冲突检测：与已有的同主体 + 同谓词（+ 同客体）Claim 比较，分型后写入冲突集，
        再按规则处理（如时效型自动设置 valid_to 和 rank）               [来源优先级、冲突规则]
      → Claim 的文本化表述写入 {ws}_claims
  → 投影器消费新事件，更新图
  → 标记 processed；出错标记 failed 并记录错误信息，可以续跑
```

进程中断后重跑时，状态为 `processing` 和 `failed` 的文档会被重新处理。各处写入都以 ID 为主键做 upsert，重复写入是安全的。

### 8.2 查询（只读）

```
query + QueryParam
  → 理解：概念提及 → 链接（别名表 → {ws}_aliases 向量 → LLM）；
          条件抽取（schema 中声明的条件变量）；问题类型（对应的谓词）            [schema]
  → Claim 通道：沿可继承关系扩展概念集合（图）                               [schema]
      → 关系库按 主体 / 客体 ∈ 概念集合、谓词、as_of 时有效、rank ≠ deprecated 取候选；
        用 {ws}_claims 做语义召回兜底
      → 条件匹配：context 与每条 Claim 的条件逐一比较 → 满足 / 不满足 / 缺失      [条件判定]
  → 文本块通道：稠密 + 稀疏融合（可选 rerank）
  → 冲突：命中的 Claim 在冲突集中时，带上冲突类型
  → 缺少关键条件 → status=need_info，返回 missing（是否追问由上层决定）
     否则         → 组装上下文（Claim 原句 + 文本块）→ prompts.render → llm → 逐句引用   [提示词]
  → 核查：逐句对照所引证据，标出未被支持的句子；按规则产出风险信号                [风险规则]
  → trace：提示词版本、模型、命中的 claim_id、关系库 seq、图投影 seq
  → 投递后台任务：未被 Claim 覆盖的文本块、合并候选
```

### 8.3 后台治理

`governance/` 的 worker 从 `tasks` 里取任务执行。每个任务都在事务里写入事件，再由投影器同步到图。

| 任务 | 触发 |
| --- | --- |
| `project_graph` | 入库或治理产生了新事件 |
| `recheck_conflict` | 有新的 Claim 入库，或者人工修改了 rank |
| `extract_uncovered` | 查询命中了未被 Claim 覆盖、但档案标明要抽 Claim 的文本块 |
| `judge_merge` | 查询或入库时发现候选碎片，依次用规则 → 本体编码 → LLM 判定 |
| `human_review` | 风险规则标出的高风险类别，自动结论又低于置信门槛 |

回滚：每种事件都有对应的逆操作（合并 ↔ 拆分、改 rank ↔ 恢复原 rank），回滚本身也写成事件。

---

## 9. 门面类用法

使用体验沿用 LightRAG 式的门面：组件由调用方注入，统一提供 `ainsert` / `aquery` 等方法。

```python
import asyncio
from datetime import date

from medrag import Document, MedRAG, QueryParam
from medrag.build import build_medrag
from medrag.config import ConfigLoader

cfg = ConfigLoader().config


async def main():
    # 组装根：按配置创建默认组件、注入领域包（knowledge.domain，默认 medical），未配地址时自动降级
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
        print(result.missing)        # 例如：是否已在服用二甲双胍（起始用药和继续用药的建议不同）
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

需要替换组件时，可以跳过组装根直接注入：

```python
from medrag.domains.medical import MedicalDomainPack
from medrag.llm import OpenAICompatEmbedding, OpenAICompatLLM
from medrag.prompts import PromptRegistry
from medrag.storage.graph import GraphDatabase
from medrag.storage.graph_projection import CypherGraphProjection
from medrag.storage.knowledge_store import SqlKnowledgeStore
from medrag.storage.milvus import MilvusVectorStorage
from medrag.storage.sql import SqlDatabase

domain = MedicalDomainPack()
rag = MedRAG(
    workspace="cardiometabolic_zh",              # 映射为 collection 名、表名前缀、图名；一个 workspace 一种语言
    domain=domain,
    llm=OpenAICompatLLM(cfg.llm),
    embedding=OpenAICompatEmbedding(cfg.embedding.text_dense),
    vectors=MilvusVectorStorage(cfg.milvus),     # 三个 collection：chunks / claims / aliases
    store=SqlKnowledgeStore(SqlDatabase(cfg.postgres)),       # 真相源；未配 postgres.host 时是 SQLite
    graph=CypherGraphProjection(GraphDatabase(cfg.neo4j)),    # 图投影；未配 neo4j.uri 时是 FalkorDBLite
    sparse=MilvusBM25Encoder(analyzer="jieba"),
    prompts=PromptRegistry.from_config(cfg.prompts, extra_dirs=[domain.prompts_dir]),
    config=cfg.knowledge,                        # 只传自己用到的那一段配置
)
```

| 方法 | 作用 |
| --- | --- |
| `initialize` / `aclose` | 建表、建索引、建图约束（幂等）；释放连接 |
| `ainsert` / `adelete` | 增量入库（带版本语义）/ 撤下文档 |
| `aretrieve` | 只检索文本块 |
| `aclaims` | 按概念、谓词、时间点取 Claim |
| `aconflicts` | 列出冲突集 |
| `aquery` | 按条件作答 |
| `averify` | 核查一句话是否有依据 |
| `areplay` | 审计回放 |
| `agovern` | 处理后台任务 |

---

## 10. 应用层（`apps/`）

- **已有组件继续保留**：`SimpleRAG`、`MultiDialogueRag`、`MedicalAgent` / `SearchGraph`（LangGraph）。会话状态存在 Redis，组件本身无状态；流式接口原样透传 LangChain 的 `astream_events(v2)` 事件，由外部转换成 SSE / WebSocket。之后改为调用内核的 `aquery`。
- **内核和领域包作为 Agent 的工具**：`claims_lookup`、`verify`、`conflicts`，以及领域包注册的确定性工具（如剂量核对）。工具只接收领域参数，检索计划由调用方固定，LLM 不接触存储细节。
- **安全策略只在这一层**。面对同一个 `QueryResult`，医生助手和患者助手的处理不同：

| 结果 | 医生助手 | 患者助手 |
| --- | --- | --- |
| `need_info` | 列出缺少的条件 | 用通俗的话追问；问不到就建议就医 |
| 学术分歧型冲突 | 呈现双方，附机构和证据等级 | 说明存在不同标准，建议遵医嘱 |
| 剂量、处方类内容 | 给出原文和版本 | 不给剂量，提示咨询医生或药师 |
| 急症红旗 | 提示紧急处理要点 | 立即给出就医指引，不继续问答 |

- 医生助手和患者助手到产品阶段再做，届时一并适配 `software/` 的前后端。

---

## 11. 模型层（`llm/`，已实现）

- 统一使用 `openai` 官方 SDK（`AsyncOpenAI`）走网络请求。`base_url`、`api_key_env`、`model`、超时、重试和并发都可以配置。DashScope、vLLM、Ollama（`/v1` 端点）、OpenAI 都按 OpenAI 兼容协议接入。以后接入其他模型商，只需再实现一个满足 `LLMFunc` / `EmbeddingFunc` 协议的类。
- 以下逻辑统一收拢在 `llm/`：去掉思考内容（`<think>…</think>` 和 `reasoning_content`）；以接口返回的 `usage` 统计 token；Embedding 按 `max_batch` 分批（DashScope 单次最多 10 条）；用信号量控制并发。
- 抽取、判定类的调用用 `json_schema` 约束输出，便于校验。
- `apps/` 里的 LangGraph 需要 LangChain 的 ChatModel，由 `llm/langchain_adapter.py` 用**同一份配置**构造 `ChatOpenAI`，保证两边连的是同一个模型。

---

## 12. 提示词（`prompts/`，已实现）

**原则**：
- 提示词是数据：每个版本单独存一个文件，可以在 PR 里看 diff。
- 已发布（released）的版本不可修改，要改就新建一个版本。改动了已发布版本的文件，测试会失败。
- 代码只按名称取提示词，具体用哪个版本由配置决定。
- 每次调用都记录 `name@version#sha256`，写进 trace 和 RunManifest。
- 中英文分开维护，找不到对应语言时直接报错，不会退回另一种语言。

**文件布局**：

```
src/medrag/prompts/templates/<名称>/
├── _meta.yaml        # 说明、变量定义（缺少必填变量或传了未声明的变量都会报错）、别名（stable / latest）
├── v1.zh.yaml        # version / parent / status（draft | released | deprecated）/ changelog / params / messages
└── v2.zh.yaml
```

- **模板语法**：新版本默认用 Jinja2 + StrictUndefined：变量缺失时直接报错，JSON 示例不用转义花括号。从 v2 迁移的 15 个提示词是 `v1`，标注 `format: fstring`，与 v2 原文逐字一致（`tests/test_prompts.py` 按 sha256 比对）。
- **内核与领域包的分工**：
  - 内核的 `prompts/templates/` 放领域无关的通用模板：`claim_extract`、`concept_link`、`conflict_type`、`query_understand`、`answer_with_claims`、`verify_sentence`。这些模板由 schema 驱动，渲染时注入实体类型、谓词、条件变量等；
  - 领域包的 `prompts/` 放同名模板的领域版本，补充领域指令和示例。它作为额外的模板目录加载，同名优先。
- **用法**：

```python
prompt = registry.get("claim_extract", lang="zh")      # 版本由配置决定；领域包里有同名模板时优先使用
messages = prompt.render(doc=doc, chunks=chunks, schema=domain.schema)
result = await llm(messages, json_schema=domain.schema.claim_json_schema(), **prompt.params)
trace.prompts.append(prompt.ref)                        # PromptRef(name, version, lang, sha256)
```

- **配置**：`prompts.versions: {claim_extract: v2}`；自定义模板目录放在 `prompts.dirs`，同名优先使用自定义的。
- **命令行**：已有 `python -m medrag.prompts list | show | check | lock`；`diff | new | release` 随 `cli.py` 补齐，届时统一为 `medrag prompts ...`。
- **迭代流程**：
  1. 新建 draft 版本，在 **dev 集**上反复修改、评测；
  2. 用 `eval compare` 和 stable 版本对比，主要指标的提升超出置信区间、逐题检查也没有系统性退化，才 release；
  3. release 后把 `stable` 指过去，在 test 集上跑一次作为正式结果，并执行 `python -m medrag.prompts lock`。

  不要在 test 集上调提示词。

---

## 13. 配置（`config/`，已实现）

配置按领域拆成多个 YAML，每一段都可以缺省：

| 文件 | 内容 |
| --- | --- |
| `storage.yaml` | `milvus`、`redis`、`postgres`、`neo4j`（默认都不配地址，即降级模式） |
| `models.yaml` | `llm`、`embedding` |
| `prompts.yaml` | `prompts` |
| `data.yaml` | v2 的入库字段映射（之后由各数据集的 loader 取代） |
| `dialogue.yaml`、`agent.yaml` | 多轮对话、Agent 策略 |
| `knowledge.yaml`（待建） | `workspace`、`lang`、`domain`（领域包，默认 `medical`）；`extract`（抽取用的模型与并发）；`concepts`（链接阈值）；`governance`（自动合并门槛）；`retrieval`（默认的 `QueryParam`） |

领域相关的设置（哪些章节抽 Claim、优先级、冲突规则）放在领域包自己的 YAML 里，不放进 `knowledge.yaml`。

```python
ConfigLoader()                                    # 读取包内 config/，或环境变量 MEDRAG_CONFIG_DIR 指定的目录
ConfigLoader("my_conf/")                          # 读取自己的配置目录
ConfigLoader(overrides=["deploy/storage.yaml"])   # 默认配置 + 整段替换某几段
# 环境变量 MEDRAG_CONFIG_OVERRIDES 追加覆盖文件（多个用 : 分隔）
```

- 密钥只写环境变量名（`api_key_env`，默认 `DASHSCOPE_API_KEY`），不写明文。
- 中间件的密码由 `deploy/start.sh` 生成到 `deploy/.env`，`ConfigLoader` 会自动加载。
- `scripts/` 读取 `scripts/conf/` 下自己的一套配置，经 `scripts/_common.py` 加载。

---

## 14. 评测与可复现

**RunManifest**：每次评测都写出一份，和结果文件放在一起（`runs/<dataset>/<run_id>/`）。

```yaml
run_id: 20261101-1432-a1b2
git: {commit: …, dirty: false}
dataset: {name: cardiometabolic_zh, subset: timesliced_v1, split: dev, manifest_sha: …}
domain: {name: medical, schema_sha: …}
prompts: {claim_extract: v1#3fa2…, answer_with_claims: v1#9c01…}
models: {llm: qwen3.5-plus, embedding: text-embedding-v4, judge: …}
knowledge: {governance_seq: 18234, graph_seq: 18234}
config_sha: …          # 检索、切块、抽取等全部配置的哈希
chunker: {name: section_paragraph, version: v1}
seed: 42
```

`eval compare <run_a> <run_b>` 输出三类内容：
- 各指标的差值和 95% 置信区间；
- 分组对比；
- 逐题差异：前对后错、前错后对的题目。

两次运行的 manifest 只要有字段不同，报告开头就列出这些不同项，避免把数据、提示词或 schema 的变化误当成算法本身的提升。

**评测集与指标**：

| 评测 | 数据 | 指标 |
| --- | --- | --- |
| 时间切片评测（主） | 切片语料按发布时间分批入库；300～500 题，每题有"截至某日"的答案和出处版本 | 按时间点作答的正确率；条件类问题的追问率和按条件作答的正确率；分歧类问题是否呈现双方；错误信息类问题是否排除 |
| Claim 抽取 | 金标准约 300 条 | 严格 / 宽松匹配下的准确率、召回率、F1；条件的完整度；证据对齐率 |
| 概念链接、冲突分型 | 标注集；CHIP-CDN 作为外部参照 | 准确率 |
| claim 级忠实度 | 回答逐句拆分 | 有依据的比例、无依据率、矛盾率、安全关键错误率 |
| 文本块回归基准 | huatuo QA：198 题 + 9,802 篇干扰文档，组成 1 万篇语料（seed=42）；用 `new_question` 检索，以 `question_hash` 判定是否命中 | Hit@k、MRR |
| 静态知识对照 | MedQA 中文中与切片相关的题目 | 选择题准确率 |
| 英文对标（开源库方向） | GraphRAG-Bench medical、DriftMed | 官方指标 |

**原则**：
- dev 集用于迭代，test 集用于验收。提示词和检索参数都不在 test 集上调。
- 改动存储或文本块检索时，先在文本块回归基准上做回归。门槛分两层：
  - 硬门槛看代码行为是否一致：逐行比对数据和查询编码；
  - 软门槛要求检索指标的差异，不超过同一实现重跑的波动再加 2pp。

  Milvus 的 HNSW 检索加上并发写入，同一实现重跑的 top-10 重合率约 91%，所以不用"重合率接近 100%"这类标准。
- 数据子集由固定种子的脚本生成，并附带 manifest。

---

## 15. 测试策略

| 类型 | 依赖 | 说明 |
| --- | --- | --- |
| 单元测试 | 无 | 存储走降级实现，数据文件放在 pytest 的临时目录；模型走 `medrag.testing.fake_openai`：按文本哈希生成向量，并支持按提示词名返回预置 JSON，用来测抽取和判定链路。不需要网络和 Docker |
| 集成测试 | Milvus / Redis / PostgreSQL / Neo4j | 标记为 `@pytest.mark.services(需要的服务...)`，不写参数时默认 milvus + redis。服务不可用时自动跳过；设置 `MEDRAG_REQUIRE_SERVICES=1` 时必须执行 |
| 双后端一致性 | 同上 | 关系库（SQLite / PostgreSQL）、图投影（FalkorDBLite / Neo4j）、向量（Milvus Lite / Milvus）在两种后端上跑同一组断言 |
| 分层检查 | 无 | `tests/test_layering.py`：核心层不 import LangChain / LangGraph；内核不 import `domains/` 和 `apps/` |
| 领域包 | 无 | schema 自洽（谓词引用的实体类型都存在，条件变量都有类型）；医疗领域包的各个声明文件都能加载 |
| 提示词 | 无 | 已发布版本未被修改；v1 与 v2 原文逐字一致 |
| 回归 | 真实 Embedding 服务 | 文本块回归基准的 Hit@k；脚本在 `scripts/regression/` |

---

## 16. 现状与差距

**搬家（v2 → medrag）的结果**：
- 包名改为 `medrag`，分层与依赖规则已生效。
- 模型层、提示词、配置、四个服务的降级都已实现，并升级到 Python 3.12。
- 测试全部通过（69 个）。端到端、会话、SQL、图的用例在降级实现和真实服务上各跑了一遍。
- 回归：`scripts/regression/m0_parity.py verify` 比对数据层 9,999 行、查询编码 198 题 × 3 路，均为 0 差异；检索指标在噪声范围内。

| 目标 | 现状 |
| --- | --- |
| `types.py` 中第 4 节的数据模型 | 只有过渡类型 `LegacyDocument`，字段与 LangChain Document 一致：`page_content` / `metadata` |
| `base.py`、`medrag.py`、`build.py` | 尚未创建 |
| `domains/medical/` | 尚未创建 |
| `MilvusVectorStorage`（三个 collection） | 仍是 v2 的 `MedicalHybridKnowledgeBase`：单行多向量布局；主键是 `md5(summary)`，summary 为空时会撞车；内部自己创建 embedding 客户端 |
| `knowledge_store.py`、`graph_projection.py` | 尚未创建；底层的 `SqlDatabase`、`GraphDatabase` 已实现，还没有调用方 |
| `SqlDatabase` 的事务 | 每条语句自动提交 |
| `ingest/` | `pipeline.py` 是 v2 形态：只区分 qa 与其他来源，QA 文本的中文前缀写死 |
| `retrieval/` | 只有 `sparse/`（pkuseg 词表的 BM25）；融合在 Milvus 内完成；上下文组装在 `apps/memory.py` 的 `TokenBudget` 里 |
| `extract/`、`concepts/`、`governance/`、`answer/`、`guard/`、`datasets/`、`cli.py` | 尚未创建 |
| `eval/` | `runner.py`、`adapters.py`、`scorers/`（ragas）已有；RunManifest、时间切片评测尚未实现 |
| `apps/` | `rag_base.py`、`simple_rag.py`、`dialogue_rag.py`、`memory.py`、`retriever.py`、`chat_history.py`、`timing.py`、`agent/` 都可用，直接使用 `MedicalHybridKnowledgeBase` |
| 内核里的医学内容 | 从 v2 迁移来的提示词（如 `medical_qa_annotation`）、`tools/annotation.py` 的科室映射、pkuseg 的 `medicine` 领域分词，都还在内核里。重写对应模块时再按三层归位，不单独搬动 |
| `scripts/` | 01–08 使用示例基于 v2 形态的接口；`regression/m0_parity.py` 是搬家时的新旧一致性回归 |
| `software/` | 冻结；backend 用的是异步化之前的接口，目前跑不起来 |

**过渡期规则**：在新的数据模型和存储抽象落地之前，检索结果仍用 `LegacyDocument`，知识库仍是 v2 形态的 `MedicalHybridKnowledgeBase`。新代码不要在这些过渡类型上继续扩展功能。
