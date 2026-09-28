# debug/ 本地联调脚本

`v2.0.0-anyc` 异步化改造的验证脚本，以及一个查看 Agent 轨迹的极简页面。

## 前置条件

```bash
bash deploy/start.sh          # 起 Milvus 与 Redis
uv pip install -e .           # 已装好，包是 editable 指向 src/MedicalRag
```

配置在 `debug/conf/`，是一份独立的配置目录，通过新增的 `ConfigLoader(config_dir)`
加载，**不会动到 `src/MedicalRag/config/` 里的默认配置**：

| 文件 | 和默认配置的差别 |
| --- | --- |
| `storage.yaml` | 集合名用 `medrag_debug`，不碰 `medical_knowledge` |
| `models.yaml` | 指向本地 vLLM：`Qwen3.8-27B`(8001) + `Octen-Embedding-8B`(8000)；`text_sparse` 用 Milvus 托管 BM25，省掉构建词表 |
| `dialogue.yaml` / `agent.yaml` | `console_debug: false`，输出干净些 |
| `data.yaml` | 与默认相同 |

每个脚本跑完都会删掉自己建的集合和 Redis key。

## 运行

直接跑，不需要设 `PYTHONPATH`：

```bash
python debug/01_config.py
```

| 脚本 | 需要 | 耗时 | 验证内容 |
| --- | --- | --- | --- |
| `01_config.py` | 无 | 秒级 | 配置拆分前后 `AppConfig` 相等、`overrides` 整段覆盖、重复顶层 key 报错、来源文件记录 |
| `02_session_store.py` | Redis | 秒级 | 历史读写与 `atrim`、TTL、JSON 缺省值、同会话串行 / 不同会话并行、锁参数 |
| `06_fake_fast.py` | Milvus + Redis + Embedding | ~10s | **改完代码先跑这个。** 假 LLM 跑完整链路：并发收益、会话状态进 Redis、新实例续接、并发不交错、流式、摘要压缩、Agent 跨轮状态 |
| `03_rag.py` | 全部 | ~1min | 真模型：入库、语义检索、单轮 RAG 各阶段耗时、并发 vs 串行、多轮改写带上历史、新实例续接、并发不交错、token 级流式 |
| `04_agent.py` | 全部 | ~5min | 真模型：`SearchGraph` 真工具调用、`MedicalAgent` 追问循环、跨轮状态进 Redis、中途换实例续接、会话隔离、流式节点 |
| `05_eval.py` | Milvus + 全部模型 | ~5min | 统一评测流程：固定 seed 采样可复现、JSONL 往返、ragas 打分、重复打分 |
| `07_web.py` | 全部 | 常驻 | Agent 轨迹查看页，见下节 |
| `08_ingest_qa.py` | Milvus + Embedding | 全量 ~40min | 把 `data/qa_50000.jsonl` 入库到独立集合 `medrag_qa50k`（`--limit N` 只入前 N 条），给轨迹页用真实数据 |

`04_agent.py` 想快一点可以 `MEDRAG_AGENT_FAST=1 python debug/04_agent.py`，
把 Agent 的 `mode` 降成 `fast`，跳过事实校验回路。

## Agent 轨迹页

```bash
python debug/07_web.py
```

启动后打开 <http://127.0.0.1:8100>。启动日志会打印用的哪个模型、哪个集合、联网检索开没开。

默认用内置 8 条样例（启动时入库、退出时删集合）。想用真实数据：

```bash
git lfs pull --include data/qa_50000.jsonl
python debug/08_ingest_qa.py                               # 入库到 medrag_qa50k
MEDRAG_WEB_COLLECTION=medrag_qa50k python debug/07_web.py  # 直接用已有集合，不写样例、退出不删
```

`medrag_qa50k` 不会被其他脚本删掉（它们只删 `medrag_debug`）；重跑 `08_ingest_qa.py` 会重建它。

页面就是一个输入框加一条时间线：把 `MedicalAgent.stream()` 产出的
`{节点名: 更新}` 按顺序显示出来，每一步标出节点名和累计耗时。

- `ask` —— 是否需要追问、追问了什么
- `extract_ask_and_reply` / `check_update_background` —— 抽取出的用户背景
- `split_query` —— 拆成几个子查询，或改写成什么
- `search_one` —— 每个子查询检索到几篇文档、小结是什么
- `answer` —— 最终回答

一轮要 1~3 分钟。Agent 判定信息不足时会先追问，在输入框里直接回答再发送即可；
换个 `session_id` 就是另一个会话；「重置会话」清掉该会话在 Redis 里的状态。

服务端 `debug/07_web.py` 约 130 行，页面 `debug/web.html` 约 90 行，没有构建步骤。
轨迹文字在服务端的 `describe()` 里生成，页面只负责渲染。

**全程真模型，没有 mock。** 联网检索按凭据决定：配了
`TENCENTCLOUD_SECRET_ID` / `TENCENTCLOUD_SECRET_KEY` 就用真的腾讯云检索，
没配就直接关掉联网检索、只用本地知识库——不会拿假数据顶。
（`llm_network_search` 在 `remain_doc_index` 为空时会清掉本地检索结果，
拿假数据顶会把真实检索效果冲掉。）只有 `06_fake_fast.py` 用桩件，
那个脚本本来就是假模型的快速回归。

## 两个需要注意的地方

### 1. 代理环境变量会被清掉

`common.py` 在导入 MedicalRag 之前会清空 `*_proxy`，原因有两个：

- 你 shell 里 `ALL_PROXY=socks://localhost:10808`，`httpx` 不认这个 scheme，
  `import langchain_ollama` 会在 import 阶段就抛 `ValueError`（`ollama` 包在
  import 时会构造一个 `httpx.Client`）；
- `http_proxy` 会把 `10.88.88.6` 的请求也绕进代理，而 `no_proxy` 里没有它。

需要保留代理时设 `MEDRAG_KEEP_PROXY=1`。**这一条对 `scripts/` 下的正式脚本同样成立**，
你在终端直接跑 `python scripts/04_basic_rag.py` 也会踩到，先 `unset ALL_PROXY all_proxy`。

### 2. `common.py` 里有三处补丁，是绕已知问题用的

包侧修好后这些都可以删。对应我已经提的任务卡片：

| 补丁 | 原因 |
| --- | --- |
| `embedder.dimensions = None` | `create_embedding_client` 总会传 `dimensions`，而 `Octen-Embedding-8B` 不支持 Matryoshka，会直接 400 |
| `dense_search_config()` 不用 `text_sparse` | Milvus 托管 BM25 时 metric 必须是 `BM25`，而 `SingleSearchRequest.metric_type` 只允许 `COSINE` / `IP`，表达不出来 |
| `build_kb()` 里包装 `kb.search` 校正请求 | `database_search` 工具让 LLM 自己填 `collection_name`（默认 `medical_knowledge`）而没有钉在 `kb` 上，LLM 还常挑 `text_sparse`+`IP`；不校正的话 Agent 里这个工具永远取不到数据 |

如果把 `models.yaml` 换成自管理词表（`text_sparse: provider: self`）并先跑
`scripts/01_build_vocab.py` 建好 `vocab.pkl.gz`，后两处补丁就不需要了。

## 数据

`data/eval/new_qa_200.jsonl` 和 `data/qa_50000.jsonl` 都是未拉取的 Git LFS 指针
（`git lfs pull` 才有内容），所以脚本用的是 `common.py` 里内置的 8 条医疗 QA
——够验证检索与生成链路，不够评估回答质量。拉下真数据后把
`03_rag.py` / `05_eval.py` 里的数据来源换掉即可。
