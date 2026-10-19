# scripts：使用示例与回归脚本

示例脚本读取本目录下的 `conf/`（与包内默认配置分开），想怎么试就怎么改，不会影响库本身。
所有脚本都在**仓库根目录**运行：`python scripts/04_basic_rag.py`。

## 配置：`conf/`

| 文件 | 内容 | 默认值 |
| --- | --- | --- |
| `storage.yaml` | Milvus / Redis / PostgreSQL / Neo4j | 不配地址 → 降级为 Milvus Lite / fakeredis / SQLite / FalkorDBLite，数据在 `.medrag/` |
| `models.yaml` | LLM 与 Embedding（OpenAI 兼容协议） | DashScope，密钥读环境变量 `DASHSCOPE_API_KEY` |
| `data.yaml` | 入库字段映射与批量 | huatuo QA |
| `prompts.yaml` | 提示词版本 | 全部 stable |
| `dialogue.yaml` / `agent.yaml` | 多轮对话、Agent 策略 | — |

- 连接 `deploy/` 启动的真实服务：`export MEDRAG_CONFIG_OVERRIDES=deploy/storage.yaml`（整段覆盖 `storage.yaml` 里的四段）。
- 换一整套配置目录：`export MEDRAG_CONFIG_DIR=/path/to/conf`。
- 加载逻辑在 `_common.py`。

## 示例

| 脚本 | 做什么 | 需要 |
| --- | --- | --- |
| `01_build_vocab.py` | 构建自管理 BM25 词表（`text_sparse.provider: self` 时需要） | extras `zh` |
| `02_ingest_data.py` | 入库 `data/qa_50000.jsonl` 的前 100 条 | Embedding 服务 |
| `03_search_data.py` | 单路 / 多路混合检索；第 2、3 个例子用自管理词表的 IP 稀疏检索 | Embedding 服务、词表 |
| `04_basic_rag.py` | 单轮 RAG | LLM + Embedding |
| `05_eval_rag.py` | 生成答案 → 落盘 → ragas 打分 | LLM + Embedding、extras `eval` |
| `06_muti_dialogue_rag.py` | 多轮 RAG（交互式） | LLM + Embedding |
| `07_single_dialogue_agent.py` | 单轮检索图：工具调用 + 事实校验 + 联网搜索 | LLM + Embedding、腾讯云凭据（联网搜索） |
| `08_medical_agent.py` | 多轮问诊 Agent（交互式） | LLM + Embedding |

不需要 Redis、Milvus 等服务：`storage.yaml` 默认就是降级模式。

## 回归：`regression/`

`regression/m0_parity.py`：M0 新旧实现一致性回归（用法见文件头，验收结论见 `docs/solution.md` 研制路线中的 M0）。它的配置全部通过命令行参数传入，不读 `conf/`。
