# 第一阶段规划：Schema、存储抽象、增量入库、评测基线

> 分支：`v2.0.0-rebuild`。目标架构见 [architecture.md](../architecture.md)，数据见 [data.md](data.md)，决策记录见 [decisions.md](../decisions.md)。
> 原则：**先搬家、后改造**。每个里程碑单独提交，**都要先通过 A 层回归才能进入下一步**。

---

## 1. 范围

**包含**

- 新的包结构；
- OpenAI 兼容的模型层；
- 版本化提示词；
- 统一的 `Document / Chunk` Schema；
- Milvus 和本地两种存储实现；
- 增量入库，并记录文档状态；
- 三层数据的子集和清洗；
- 可复现的评测（RunManifest + compare）；
- 三条基线，以及和 LightRAG 的对标。

**不包含**

- 实体关系抽取和图谱、本体对齐、断言检测（属于第二、三阶段）；
- 安全层（引用校验、剂量计算工具、PHI 脱敏）；
- PDF 解析；
- 新前端。`software/` 在第一阶段冻结，只修启动失败的问题（D11）。

---

## 2. 里程碑

### M0 搬家（行为不变）　**状态：已完成，门槛中的重合率一项待定（D17）**，见 [m0-report.md](m0-report.md)

1. 新建 `src/medrag/` 目录结构和根目录的 `pyproject.toml`（extras：`milvus / redis / pg / agent / eval / data`），删除 `src/setup.py`。
2. 实现 `llm/openai_compat.py`，替换掉 `core/utils.py` 里的 LangChain 客户端；`apps/` 改为通过 `langchain_adapter` 获取 ChatModel。
3. 实现 `PromptRegistry`，把 `templates.py` 里的提示词原样迁移成 `v1.zh.yaml`（状态为 released），并加一致性测试：新旧渲染结果逐字一致。
4. 按第 3 节的迁移清单搬迁其余模块，同时修好清单里列出的已知问题。
5. 更新 `scripts/`、`debug/` 和 `tests/` 的 import。

**门槛**

- 在旧版 v2 实现上跑出 A 层 198 题的 top-10 结果，并保存为基准文件；
- 新实现在同一份配置下，top-10 重合率 ≥ 99%，Hit@1/5/10 的差值在 ±1pp 以内；
- `scripts/02–08` 都能跑通；
- 提示词一致性测试通过。

### M1 统一 Schema 与存储抽象

1. 实现 `types.py` 和 `base.py`；Milvus 的 collection 由 `Chunk` 生成（见架构文档 9 节）。
2. 实现 `LocalVectorStorage`，以及确定性的假 LLM / 假 Embedding，单测不再依赖外部服务。
3. 实现 `BaseSparseEncoder` 的两种实现（D3）。
4. 实现 `QAPairChunker`：每个 QA 是一个 Document，`doc_id = question_hash`；超长答案切块时，每一块都带上问题，并保证不超过 VARCHAR 的字节上限。
5. 实现 `MedRAG.ainsert / aretrieve`。

**门槛**

- A 层 Hit@k 不低于 M0；
- 同一份数据重复入库，新增行数为 0；
- 单测在没有 Docker 的环境下全部通过。

### M2 增量入库

1. 实现 `DocStatusStorage`（JSON 版和 PostgreSQL 版，D4）。
2. 实现增量入库的完整流程：按 `content_hash` 决定跳过、更新还是新增；支持断点续传和失败重试。
3. 实现 `MedRAG.adelete`。
4. 记录稀疏检索在增量入库后统计量的变化（D3 的 A/B 实验在这里做）。

**门槛**

- 入库中途中断，续跑后的结果和一次性入库完全一致；
- 删除后检索不到；
- 修改一篇文档后重新入库，旧的 chunk 全部被替换；
- 记录吞吐量（docs/s），可以先用 A 层的 1 万篇测，再视情况用 Huatuo 百科压测。

### M3 数据子集、清洗与评测框架

1. 实现 `datasets/`：下载、清洗规则、子集生成和 manifest（见 data.md）。产出 `a10k`、`graphrag_bench:medical`、`c_mini` 三个子集。
2. 实现 `SectionParagraphChunker`，切块时保留章节路径。
3. 实现 `eval/`：RunManifest、Hit@k / MRR、选择题评分、GraphRAG-Bench 输出格式、`eval compare`。
4. 实现 `MedRAG.aquery`（上下文组装 + 引用编号），并把 `apps/` 改为调用 `aquery`。
5. 用 LLM 对 c_mini 做 OCR 纠错（D13），产出纠错 diff 日志，完成人工抽检。

**门槛**

- 三个子集都能由脚本重新生成，并且 sha256 一致；
- 同一份配置连续跑两次，非 LLM 指标完全一致；LLM 指标的差异在置信区间以内。

### M4 基线与对标

| 编号 | 基线 | A（Hit@k） | B（官方指标） | C（准确率） |
| --- | --- | --- | --- | --- |
| B0 | 不检索 | — | ✓ | ✓ |
| B1 | 稠密 + 稀疏融合 | ✓ | ✓ | ✓ |
| B2 | LightRAG（hybrid / mix），模型与 B1 相同 | — | ✓ | c_mini（视预算） |
| B3 | B1 + rerank | ✓ | ✓ | ✓ |

**产出**：一份对比报告。回答两个问题：
- 在哪些题型或科目上，混合检索不如 LightRAG；
- LLM 纠错前后，c_mini 的检索和答题指标差了多少（作为对照数据，不设门槛）。

这份报告作为第二阶段（图谱）优先级的依据。

**合理性检查**

- B 层 B1 的分数应该和论文里的 "RAG w/o rerank" 在同一量级，如果差距超过 10pp，先排查流程问题。
- C 层的 B1 应该高于 B0。如果不是，先检查语料和题目是否对得上，再去调检索。

---

## 3. 模块迁移清单

| 现有模块 | 去向 | 动作 | 需要修复的问题 |
| --- | --- | --- | --- |
| `config/loader.py` | `config/loader.py` | 直接迁移 | — |
| `config/models.py` | `config/models.py`、`types.py` | 拆分 | 各段改为可选；`DataConfig` 由 loader 取代；`SearchRequest` 不再对外 |
| `core/KnowledgeBase.py` | `storage/milvus.py` | 改造 | 写死的 11 个字段改为由 `Chunk` 生成；embedding 改为外部注入 |
| `core/insert.py` | `storage/milvus.py` | 迁移 | 批大小写死为 20；`delete_by_ids` 每批只删 1 条；`insert_rows_has_id` 没有传递 `progress_fn` |
| `core/HybridRetriever.py` | `apps/` | 迁移 | 只有 LangChain 应用层还在用 |
| `core/IngestionPipeline.py` | `ingest/pipeline.py` | 重写 | `md5(summary)` 在 summary 为空时会撞车；只认 qa 和 literature 两种来源；`get_stopwords` 没有任何调用方 |
| `core/utils.py` | `llm/` | 重写 | 改用 openai SDK（D2） |
| `core/memory.py` | `apps/memory.py` | 直接迁移 | — |
| `core/session_store.py` | `storage/redis_session.py` | 直接迁移 | — |
| `embed/sparse.py`、`bm25.py` | `retrieval/sparse/` | 改造 | `vec if vec is not None` 是死代码；IDF 冻结问题；第三方 `stopwords` 包和本地停用词表并存，统一成一份 |
| `rag/*` | `apps/` | 迁移 | M3 改为调用 `aquery` |
| `agent/*` | `apps/agent/` | 迁移 | 工具参数改为 `query + filters`（D10） |
| `prompts/templates.py` | `prompts/templates/**` | 转换为文件 | 映射表移到 `tools/annotation.py` |
| `data/annotation.py` | `tools/annotation.py` | 迁移 | — |
| `eval/*` | `eval/` | 迁移并扩展 | — |
| `src/setup.py`、`src/pyproject.toml` | 根目录 `pyproject.toml` | 重写 | — |
| `software/backend` | 保持在包外 | 第一阶段冻结 | `api/app.py:589` 还在 import 已删除的 `RagEvaluate` |

---

## 4. 风险

| 风险 | 应对 |
| --- | --- |
| M0 的回归门槛受 Embedding 服务波动影响 | 基准结果固定下来以后，同一份配置再跑两次，用两次之间的差异作为噪声底线 |
| Milvus 原生 BM25 的中文分词不如 pkuseg | D3 的 A/B 实验数据说了算；必要时给 jieba 加载医学词典 |
| LightRAG 版本升级导致 `run_lightrag.py` 的改动方式失效 | 固定 LightRAG 版本，并记录到 RunManifest |
| LLM 纠错改坏原文 | 长度比、编辑距离、数字序列三项机器校验，不通过就保留原文；人工抽检 100 段，误改率超过 2% 就重来；保留原文副本 |
| 范围蔓延 | 第二阶段的事项一律记入 decisions.md 的待办，不进入第一阶段 |
