# 决策记录

> 这里记录需要维护者把握方向的取舍。状态说明：
> - **已定**：维护者已经确认；
> - **采纳建议·待确认**：先按建议推进，维护者可以随时推翻；
> - **待定**：还没开始，到对应里程碑时再定。
>
> 推翻某条决策时，修改这里的状态和理由，再同步修改受影响的文档。

| 编号 | 问题 | 状态 | 结论 | 影响范围 |
| --- | --- | --- | --- | --- |
| D1 | 包名 | **已定** | 改为 `medrag`，不保留 `MedicalRag` 兼容层（M0 已完成） | M0：全部 import、打包 |
| D2 | 模型接入方式，以及 LangChain 用在哪一层 | **已定** | 统一用 `openai` SDK 走网络请求，`base_url`、密钥等可配置，其他模型商以后再加；核心层不依赖 LangChain，`apps/` 通过适配器使用 LangChain | M0 |
| D3 | 稀疏检索 | 采纳建议·待确认 | 两种实现放在同一接口之后，默认用 Milvus 原生 BM25；M2 做 A/B 实验后最终确定 | M1 的 schema、M2 |
| D4 | 文档状态存在哪里 | 采纳建议·待确认 | 生产用 PostgreSQL，测试用本地 JSON | M2 |
| D5 | 算力和预算 | **已定** | 跟随 D2，走 OpenAI 兼容的网络请求；先用子集（c_mini），LightRAG 在 C 层先不跑全量 | M4 |
| D6 | 数据清洗 | **已定** | 先用子集，子集要清洗；上全量时另想办法 | M3 |
| D7 | A 层的参考答案 | 采纳建议·待确认 | 检索评测按 `question_hash`；生成评测用原始 `answer`；`new_question` 作为检索查询 | M3 |
| D8 | C 层的题目口径 | 采纳建议·待确认 | 5 选项；在 dev 集上迭代，在 test 集上验收；先用 c_mini，再扩到 c_sub | M3、M4 |
| D9 | 提示词管理 | **已定方向** | 版本化文件管理，人性化、可迭代、可对比。设计见 [design/prompt-management.md](design/prompt-management.md) | M0 |
| D10 | Agent 工具参数 | 采纳建议·待确认 | M0 已改为 `query + top_k`，检索计划由调用方固定，LLM 不再接触 `anns_field` 和 `expr`；`filters` 随 M1 的 `QueryParam` 加入。`agent.auto_search_param` 已删除，新增 `agent.db_search_top_k` | M0 / M1 |
| D11 | `software/` 前后端 | 采纳建议·待确认 | 第一阶段冻结，M0 未改动。注意：backend 在 M0 之前就已经跑不起来（用的是 v2 异步化改造之前的构造函数，如 `SearchGraph(config, ...)`、`IngestionPipeline(cfg)`；`run_api.py` 指向不存在的 `MedicalRag.api.app`），只改 import 救不回来，统一在 M3 之后适配新接口 | M3 之后 |
| D12 | 分支 | **已定** | 删除 `v2.0.0-schema`，直接在 `v2.0.0-rebuild` 上开发 | — |
| D13 | 中文教材的 OCR 纠错 | **已定** | 统一用 LLM 纠错，不维护人工错字表。纠错提示词 `ocr_correct` 纳入提示词版本管理，约束和抽检方式见 [phase1/data.md](phase1/data.md) 第 3 节 | M3 |
| D14 | A 层非医疗内容的过滤方式 | 采纳建议·待确认 | 用 LLM 判断，再人工抽检 100 条 | M3 |
| D15 | 提示词模板语法 | 采纳建议·待确认 | 新版本默认 Jinja2 + StrictUndefined（`jinja2` 已在依赖中）；从 v2 迁移的 v1 版本标注 `format: fstring`，保持与原文逐字一致，也能直接交给 LangChain 的 ChatPromptTemplate | M0 |
| D16 | 中英文数据如何入库 | **已定** | 暂时隔离：中英文分属不同的 workspace（不同的 collection 和文档状态命名空间），一个 workspace 只有一种语言和一种 analyzer。多语言混合检索留到以后 | M1 |
| D17 | M0 回归门槛中的"top-10 重合率 ≥ 99%" | **待定** | 实测达不到：同一实现重跑一次重合率只有 91.41%（Milvus 近似检索 + 并发写入顺序不固定），新旧之间是 90.66%，落在噪声范围内。数据层逐行比对与查询编码比对都是 0 差异，Hit@k 差值在 ±1pp 以内。建议把门槛改为：①数据层与查询编码逐项一致；②Hit@1/5/10 差值在 ±1pp 以内；③重合率不低于同一实现重跑的噪声底线减 2pp。详见 [phase1/m0-report.md](phase1/m0-report.md) | M0 验收、后续里程碑 |

## 第二阶段待办（不进入第一阶段）

- 医疗实体和关系类型的 schema；用 Neo4j 存图谱；查询模式 local / global / mix。
- 本体选型：ICD-10 国家临床版 2.0、ATC、自建同义词表；SNOMED CT 的授权问题。
- 数据扩充：Huatuo 百科全量、MIRAGE、CMB-Clin（用于 Agent 多轮问诊评测）。
