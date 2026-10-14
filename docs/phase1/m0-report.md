# M0 报告：搬家（行为不变）

> 分支 `v2.0.0-rebuild`，基线是 v2 的 `cff9b8e`。M0 的范围和门槛见 [plan.md](plan.md)。

## 1. 做了什么

| 规划项 | 结果 |
| --- | --- |
| 新包结构 + 根目录 pyproject（extras） | `src/MedicalRag` → `src/medrag`；删除 `src/setup.py`、`src/pyproject.toml`、`src/config/extension.toml`；extras：`milvus / redis / pg / zh / agent / websearch / eval / dev / all` |
| OpenAI 兼容模型层 | `llm/openai_compat.py`：`OpenAICompatLLM`、`OpenAICompatEmbedding`（分批、并发、顺序保持、可选 `dimensions`、思考内容剥离、流式 usage）；`apps/` 通过 `llm/langchain_adapter.py` 用同一份配置构造 `ChatOpenAI` |
| 版本化提示词 | `PromptRegistry` + 15 个提示词的 `v1.zh.yaml`（`format: fstring`，与 v2 原文逐字一致）+ `prompts.lock` + `python -m medrag.prompts list/show/check/lock` |
| 模块迁移 | 按 plan.md 第 3 节迁移；核心层（`apps/` 以外）不再 import LangChain，由 `tests/test_layering.py` 检查 |
| scripts / debug / tests | 全部改为 `medrag`；`debug/common.py` 里绕开 v2 已知问题的三处补丁已删除（包里已修复） |

### 迁移过程中修复的问题

| 问题 | 修复 |
| --- | --- |
| 入库后不 flush，新写入的数据停留在未封存的增长段，检索走临时索引；同一份数据的检索结果随 Milvus 自动封存的时机变化（实测 Hit@10 相差约 13pp） | 入库流水线在建索引前先 `flush` |
| `delete_by_ids` 每批只删 1 条；`insert_rows` 批大小写死；upsert 路径丢了 `progress_fn` | 批大小可配置，进度回调统一处理 |
| `create_embedding_client` 总会带 `dimensions`，不支持 Matryoshka 的模型直接 400 | `DenseConfig.send_dimensions` |
| `SingleSearchRequest.metric_type` 不能写 `BM25`，Milvus 托管稀疏向量时默认检索配置会报错 | 允许 `BM25`；`default_search_config(kb)` 按稀疏向量的管理方式选择指标 |
| Agent 的 `database_search` 工具让 LLM 自己填 collection、检索字段、指标、过滤表达式，经常取不到数据 | 工具只接收 `query` 和 `top_k`，检索计划由调用方固定（D10） |
| `stopwords==1.0.1` 与 `stopwords-zh` 写同一个目录，安装顺序决定 `filter_stopwords` 是否存在 | 只保留 `stopwords-zh`（pyproject 与 environment.yml） |
| `BM25Vectorizer.build_sparse_vec_from_tokens` 中 `vec if vec is not None` 的死代码；`IngestionPipeline.get_stopwords` 无调用方 | 删除 |
| `DataConfig.chunk_id_field` 类型写成 `int` | 改为 `str`（它是字段名） |

### 有意保留、留到 M1 的问题

- 主键 `hash_id = md5(summary)`：summary 为空的文献 chunk 会互相覆盖。
- 入库只区分 `qa` 与其他来源，QA 文本的中文前缀写死。
- `MedicalHybridKnowledgeBase` 仍在内部创建 embedding 客户端；对外的 `SearchRequest` 仍暴露 `anns_field` 等存储细节（只是不再交给 LLM）。

## 2. 验证

### 2.1 测试

`pytest tests -q`：**46 passed**。其中集成测试在容器内的 Milvus 2.6.0 + Redis 7.4 上执行，模型请求走本地假 OpenAI 服务。

| 测试 | 覆盖 |
| --- | --- |
| `test_prompts.py` | 15 个 v1 提示词与 v2 原文逐字一致（sha256）；已发布版本锁定；渲染时的变量校验；Jinja2；自定义目录覆盖；版本解析；不跨语言回退 |
| `test_llm.py` | 分批与顺序、`dimensions` 开关、额外参数透传、流式与 usage |
| `test_config.py` | 包内默认配置可加载、各段可缺省、overrides 整段替换、重复 key 报错 |
| `test_layering.py` | 核心层不依赖 LangChain |
| `test_session_store.py`、`test_services.py` | 沿用 v2 的用例，改到新包 |
| `test_apps_e2e.py` | 入库 → 检索 → SimpleRAG → 多轮 RAG（含流式）→ 检索图工具调用 → MedicalAgent 两轮 |

另外，用假服务跑通了 `scripts/02_ingest_data.py`、`scripts/04_basic_rag.py`，以及 `debug/01_config.py`、`debug/06_fake_fast.py`（11 个检查点全部通过）。

### 2.2 A 层回归（`scripts/regression/m0_parity.py`）

- **语料**：`new_qa_200` 的 198 篇标准文档，加上 `qa_50000` 中随机抽取的 9,802 篇干扰文档（seed=42），共 10,000 篇。去重后 Milvus 中有 9,999 行。
- **词表**：用这份语料构建 pkuseg medicine 词表（98,713 个词），新旧实现共用同一个词表文件。
- **embedding**：本地假服务（字符二元组哈希，1024 维）。这是因为容器里没有 DashScope 密钥。假 embedding 只有字面相似度，所以下表的 Hit@k 绝对值没有意义，只看新旧之间的差异。
- **检索配置**：v2 默认配置，即 summary_dense(COSINE) + text_sparse(IP) 加权融合 0.8 / 0.2，返回 top-10。
- **流程**：两边都是入库 → flush → 等索引建完 → 检索。

| 对比 | top-10 集合重合率 | 排序完全一致 | Hit@1 | Hit@5 | Hit@10 | MRR |
| --- | --- | --- | --- | --- | --- | --- |
| v2（`cff9b8e`） | — | — | 33.84% | 47.98% | 48.99% | 0.3963 |
| medrag（M0） | 90.66%（相对 v2） | 79/198 | 34.34% | 47.98% | 48.99% | 0.3984 |
| medrag 重跑一次（噪声底线） | 91.41%（相对上一行） | 85/198 | 34.34% | 48.48% | 49.49% | 0.4018 |

`verify` 子命令排除了 Milvus 的不确定性，直接比对代码行为：

- **数据层**：两个集合 9,999 行逐行比对文本、两路稠密向量、稀疏向量，**0 处不一致**。
- **查询编码**：198 题 × 3 路（两路稠密 + 稀疏），新旧代码的编码结果**0 处不一致**。

### 2.3 门槛判定

| 门槛 | 结果 |
| --- | --- |
| Hit@1/5/10 差值在 ±1pp 以内 | **通过**（+0.5 / 0 / 0） |
| top-10 重合率 ≥ 99% | **未通过（90.66%）**。同一实现重跑一次也只有 91.41%：Milvus HNSW 是近似检索，加上 `concurrency=4` 时写入顺序不固定、段的组成不同，结果本身就有约 9% 的波动。这条门槛在当前的定义下达不到，与代码是否一致无关 |
| `scripts/02–08` 跑通 | 02、04 用假服务跑通；03 的第 2、3 个例子把 `text_sparse` 固定写成 IP 指标，需要自管理词表；05–08 需要真实模型或交互输入，未在容器里执行 |
| 提示词一致性 | **通过** |

门槛怎么调整需要你来定，见 [decisions.md](../decisions.md) D17。

## 3. 需要你在本地补跑的

1. 用真实 embedding（DashScope `text-embedding-v4`）重跑一次 A 层回归。用法见脚本文件头；把 `--base-url`、`--model`、`--api-key-env DASHSCOPE_API_KEY` 换成真实服务即可，入库约需 1 万条 × 2 次 embedding 调用。
2. 在有模型的环境里跑一遍 `scripts/05–08` 和 `debug/03、04、05、07`。
