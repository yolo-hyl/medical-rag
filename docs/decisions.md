# 决策记录

> 这里记录需要维护者把握方向的取舍。状态说明：
> - **已定**：维护者已经确认；
> - **采纳建议·待确认**：先按建议推进，维护者可以随时推翻；
> - **待定**：还没开始，到对应里程碑时再定。
>
> 推翻某条决策时，修改这里的状态和理由，再同步修改受影响的文档。

| 编号 | 问题 | 状态 | 结论 | 影响范围 |
| --- | --- | --- | --- | --- |
| D1 | 包名 | 采纳建议·待确认 | 改为 `medrag`，不保留 `MedicalRag` 兼容层 | M0：全部 import、打包、backend |
| D2 | 模型接入方式，以及 LangChain 用在哪一层 | **已定** | 统一用 `openai` SDK 走网络请求，`base_url`、密钥等可配置，其他模型商以后再加；核心层不依赖 LangChain，`apps/` 通过适配器使用 LangChain | M0 |
| D3 | 稀疏检索 | 采纳建议·待确认 | 两种实现放在同一接口之后，默认用 Milvus 原生 BM25；M2 做 A/B 实验后最终确定 | M1 的 schema、M2 |
| D4 | 文档状态存在哪里 | 采纳建议·待确认 | 生产用 PostgreSQL，测试用本地 JSON | M2 |
| D5 | 算力和预算 | **已定** | 跟随 D2，走 OpenAI 兼容的网络请求；先用子集（c_mini），LightRAG 在 C 层先不跑全量 | M4 |
| D6 | 数据清洗 | **已定** | 先用子集，子集要清洗；上全量时另想办法 | M3 |
| D7 | A 层的参考答案 | 采纳建议·待确认 | 检索评测按 `question_hash`；生成评测用原始 `answer`；`new_question` 作为检索查询 | M3 |
| D8 | C 层的题目口径 | 采纳建议·待确认 | 5 选项；在 dev 集上迭代，在 test 集上验收；先用 c_mini，再扩到 c_sub | M3、M4 |
| D9 | 提示词管理 | **已定方向** | 版本化文件管理，人性化、可迭代、可对比。设计见 [design/prompt-management.md](design/prompt-management.md) | M0 |
| D10 | Agent 工具参数 | 采纳建议·待确认 | 改为 `query + filters`，LLM 不再接触 `anns_field` 和 `expr` | M0 / M3 |
| D11 | `software/` 前后端 | 采纳建议·待确认 | 第一阶段冻结，只修启动失败的 import | — |
| D12 | 分支 | **已定** | 删除 `v2.0.0-schema`，直接在 `v2.0.0-rebuild` 上开发 | — |
| D13 | 中文教材的 OCR 纠错 | 待定（M3） | 建议在 c_mini 上做三组对比实验：不清洗 / 只用错字表 / 错字表 + LLM 纠错，用数据决定 | M3 |
| D14 | A 层非医疗内容的过滤方式 | 采纳建议·待确认 | 用 LLM 判断，再人工抽检 100 条 | M3 |
| D15 | 提示词模板语法 | 采纳建议·待确认 | Jinja2 + StrictUndefined（新增依赖 `jinja2`） | M0 |
| D16 | 中英文混合入库 | 待定（M1） | 先验证 Milvus 2.6 的多语言 analyzer；不支持的话就按语言分 workspace | M1 |

## 第二阶段待办（不进入第一阶段）

- 医疗实体和关系类型的 schema；用 Neo4j 存图谱；查询模式 local / global / mix。
- 本体选型：ICD-10 国家临床版 2.0、ATC、自建同义词表；SNOMED CT 的授权问题。
- 数据扩充：Huatuo 百科全量、MIRAGE、CMB-Clin（用于 Agent 多轮问诊评测）。
