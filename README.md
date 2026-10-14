# medrag：专攻医疗领域的通用 RAG 库

> **v2.0.0-rebuild 重构中**：项目正在从 v2（以 QA 数据为中心的医疗问答应用）重构为像 [LightRAG](https://github.com/HKUDS/LightRAG) 那样分层抽象、专攻医疗领域的通用 RAG 库。
> 当前处于第一阶段的 **M0（搬家）**：包名从 `MedicalRag` 改为 `medrag`，代码按新的分层迁移，检索与问答行为与 v2 保持一致。
> 目标架构见 [docs/architecture.md](docs/architecture.md)，第一阶段规划见 [docs/phase1/plan.md](docs/phase1/plan.md)，方向性决策见 [docs/decisions.md](docs/decisions.md)。

## 能做什么

- **混合检索**：问题向量 + 正文向量 + BM25 稀疏向量（pkuseg 医疗分词的自管理词表，或 Milvus 托管 BM25），RRF / 加权融合。
- **单轮 RAG、多轮 RAG、问诊 Agent**：多轮对话的历史、摘要、Agent 跨轮状态存在 Redis 里，组件本身无状态，可以多进程部署；都提供流式接口。
- **不装服务也能跑**：Milvus、Redis、PostgreSQL、Neo4j 的地址都可以不配，自动降级为 Milvus Lite、fakeredis、SQLite、FalkorDBLite，`pip install` 之后不需要 Docker 就能跑通全部链路。
- **统一走 OpenAI 兼容协议**：DashScope、vLLM、Ollama（`/v1` 端点）、OpenAI 都按 `base_url` + 密钥环境变量接入。
- **版本化提示词**：提示词是带版本的 YAML 文件，已发布版本锁定不可改，版本由配置选择，每次调用可追溯到具体版本。
- **可复现评测**：统一的答题函数与样本格式，答案落盘后可重复打分、横向对比。

## 目录

```
medical-rag/
├── src/medrag/                 # 库本体（分层见 docs/architecture.md 第 2、3 节）
│   ├── config/                 # 配置模型与多 YAML 加载器（storage / models / data / prompts / dialogue / agent）
│   ├── llm/                    # OpenAI 兼容的 LLM / Embedding 客户端；langchain_adapter 只给应用层用
│   ├── prompts/                # PromptRegistry + templates/<名称>/<版本>.<语言>.yaml + prompts.lock
│   ├── storage/                # Milvus 知识库、Redis 会话、SQL、图；未配置地址时的降级在 backends.py
│   ├── retrieval/sparse/       # 自管理词表的 BM25（pkuseg 医疗分词）
│   ├── ingest/                 # 入库流水线
│   ├── apps/                   # 应用层：SimpleRAG、多轮 RAG、记忆管理、MedicalAgent / SearchGraph
│   ├── eval/                   # 统一评测流程；scorers/ 下是 ragas 打分
│   ├── tools/                  # 数据标注等非核心工具
│   └── testing/                # 本地假 OpenAI 服务（测试、离线联调用）
├── scripts/                    # 01–08 使用示例（配置在 scripts/conf/，可随意修改）；regression/ 下是里程碑回归脚本
├── tests/                      # 单测（存储走降级实现，无外部依赖）+ 集成测试（@pytest.mark.services）
├── deploy/                     # Milvus / PostgreSQL / Redis / Neo4j 的 docker compose；storage.yaml 是连接它们的配置
├── data/                       # 示例数据（git lfs）与 GraphRAG-Bench
├── docs/                       # 架构、规划、决策、设计文档
└── software/                   # 基于本包的前后端示例应用（第一阶段冻结，见 D11）
```

## 快速开始

### 1. 安装

```bash
git clone https://github.com/yolo-hyl/medical-rag && cd medical-rag
git lfs pull                                    # 拉取 data/ 下的示例数据

conda create -n rag python=3.12 -y && conda activate rag   # 要求 Python >= 3.12（FalkorDBLite 的要求）
pip install -e ".[all]"                         # 核心 + Milvus / Redis / PostgreSQL / 图 / Agent / 评测 / 测试（含各自的降级实现）
# 自管理词表的中文 BM25 需要 pkuseg（要先装 numpy、cython，并关闭构建隔离）
pip install numpy cython && pip install --no-build-isolation -e ".[zh]"
# 需要腾讯云联网搜索时
pip install -e ".[websearch]"
```

依赖按用途拆成了 extras：`milvus`、`redis`、`pg`、`graph`、`zh`、`agent`、`websearch`、`eval`、`dev`。
`milvus` 带 milvus-lite，`redis` 带 fakeredis，`graph` 带 neo4j 驱动和 FalkorDBLite；SQLite 用标准库。
核心包本身只依赖 `openai`、`pydantic`、`pyyaml`、`jinja2` 等少数几个库。

> 不要同时安装 `stopwords` 和 `stopwords-zh`：两个包写的是同一个 `stopwords/` 目录，后装的会覆盖先装的。本项目只需要 `stopwords-zh`。

### 2. 基础服务（可选）

**不启动任何服务也能用**：包内默认的 `storage.yaml` 没有配置服务地址，四个服务都会降级为嵌入式实现，数据放在当前目录的 `.medrag/` 下：

| 服务 | 没有配置地址时 | 适用范围 |
|------|----------------|----------|
| Milvus（`milvus.uri`） | Milvus Lite（`.medrag/milvus_lite.db`） | 单进程；不支持 Windows |
| Redis（`redis.host`） | fakeredis（进程内存） | 单进程，进程退出即丢失 |
| PostgreSQL（`postgres.host`） | SQLite（`.medrag/medrag.sqlite3`） | 单进程 |
| Neo4j（`neo4j.uri`） | FalkorDBLite（`.medrag/falkordb.db`） | 单进程 |

降级实现适合开发、测试和小规模使用，首次降级时会打一条 warning。多进程部署或数据量较大时，用 `deploy/` 启动真实服务。

所有真实服务都通过 `deploy/` 下的 docker compose 管理。脚本会拉取镜像、启动容器，并等待健康检查通过；以 0 退出即表示服务可用。

```bash
bash deploy/start.sh                  # 拉取镜像、启动全部服务并等待就绪
bash deploy/start.sh milvus redis     # 只启动需要的服务
bash deploy/stop.sh                   # 停止（保留数据）
bash deploy/stop.sh --purge           # 删除容器和全部数据（需确认）
```

| 服务 | 用途 | 默认地址 |
|------|------|----------|
| Milvus 2.6 | 向量库 | `http://127.0.0.1:19530` |
| Redis 7.4 | 多轮对话与 Agent 的会话状态、会话锁 | `127.0.0.1:6379` |
| PostgreSQL 16 | 文档入库状态（M2 起使用） | `127.0.0.1:5432` |
| Neo4j 5.26 Community | 知识图谱（第二阶段起使用） | `bolt://127.0.0.1:7687` |

- 启动后让代码连接这些服务：用 `deploy/storage.yaml` 覆盖存储配置，例如 `export MEDRAG_CONFIG_OVERRIDES=deploy/storage.yaml`，或在代码里 `ConfigLoader(overrides=["deploy/storage.yaml"])`。
- 首次运行会由 `deploy/.env.example` 生成 `deploy/.env`，并为各服务生成随机密码；`ConfigLoader` 会自动读取，无需手动导出。
- Docker Hub 拉取受限（429）时，在 `deploy/.env` 中设置 `REGISTRY=mirror.gcr.io`（或 `docker.m.daocloud.io`）后重试。
- 端口默认只绑定 `127.0.0.1`，需要远程访问时在 `.env` 中设置 `BIND_ADDR=0.0.0.0`。

### 3. 配置模型

配置在 `src/medrag/config/` 下，按领域拆成多个文件，每一段都可以缺省：

| 文件 | 包含的配置 | 什么时候改 |
| --- | --- | --- |
| `storage.yaml` | `milvus`、`redis`、`postgres`、`neo4j` | 换部署环境 |
| `models.yaml` | `llm`、`embedding` | 换模型 / 服务商 |
| `data.yaml` | `data` | 换数据集的字段映射 |
| `prompts.yaml` | `prompts` | 切换提示词版本 |
| `dialogue.yaml` | `multi_dialogue_rag` | 调整多轮对话策略 |
| `agent.yaml` | `agent` | 调整 Agent 策略 |

```python
from medrag.config import ConfigLoader

ConfigLoader()                                    # 默认读取包内 config/（或环境变量 MEDRAG_CONFIG_DIR 指定的目录）
ConfigLoader("my_conf/")                          # 读取自己的配置目录
ConfigLoader(overrides=["exp/models_local.yaml"]) # 默认配置 + 整段替换某几段
# 环境变量 MEDRAG_CONFIG_OVERRIDES 追加覆盖文件（多个用 : 分隔），例如 export MEDRAG_CONFIG_OVERRIDES=deploy/storage.yaml
```

模型统一按 OpenAI 兼容协议配置，密钥只写环境变量名，不写明文：

```yaml
# models.yaml
llm:
  model: qwen3.5-plus
  base_url: https://dashscope.aliyuncs.com/compatible-mode/v1
  api_key_env: DASHSCOPE_API_KEY      # 本地 vLLM / Ollama 不校验密钥，可以不写
  temperature: 0.1
  extra_body: {}                      # 透传给接口的额外参数，例如 {enable_thinking: false}
embedding:
  summary_dense:                      # 问题向量
    model: text-embedding-v4
    base_url: https://dashscope.aliyuncs.com/compatible-mode/v1
    api_key_env: DASHSCOPE_API_KEY
    dimension: 1024
    max_batch: 10                     # 单次请求最多几条；DashScope 上限是 10
    send_dimensions: true             # 不支持 Matryoshka 的模型（如 Octen-Embedding-8B）设为 false
  text_dense: { ... }                 # 正文向量，字段同上
  text_sparse:
    provider: self                    # self = pkuseg 自管理词表；Milvus = Milvus 托管 BM25
    domain_model: medicine
```

本地 Ollama 的写法：`base_url: http://localhost:11434/v1`，不写 `api_key_env`。

### 4. 跑示例

示例脚本读取 `scripts/conf/` 下自己的一套配置（结构与包内 `src/medrag/config/` 相同），想换模型、换服务地址就改这里，详见 [scripts/README.md](scripts/README.md)。在仓库根目录运行：

```bash
python scripts/01_build_vocab.py           # 构建自管理 BM25 词表（text_sparse.provider=self 时需要）
python scripts/02_ingest_data.py           # 入库 qa_50000 的前 100 条
python scripts/03_search_data.py           # 单路 / 多路混合检索
python scripts/04_basic_rag.py             # 单轮 RAG
python scripts/05_eval_rag.py              # 生成答案 → 落盘 → ragas 打分
python scripts/06_muti_dialogue_rag.py     # 多轮 RAG（交互式，需要 Redis）
python scripts/07_single_dialogue_agent.py # 单轮检索图（工具调用 + 事实校验 + 联网搜索）
python scripts/08_medical_agent.py         # 多轮问诊 Agent（交互式，需要 Redis）
```

没有 API 密钥时，可以先用本地的假 OpenAI 服务把链路跑通（回答是固定的，只用来验证链路）：

```bash
python -m medrag.testing.fake_openai --port 8999     # 然后把 base_url 指向 http://127.0.0.1:8999/v1
```

## 在代码里使用

所有 IO 方法都是 `async`；知识库、会话存储、提示词注册表由调用方创建后注入，一个进程里各一个实例即可：

```python
import asyncio

from medrag.apps.agent import MedicalAgent
from medrag.apps.dialogue_rag import MultiDialogueRag
from medrag.config import ConfigLoader
from medrag.llm.langchain_adapter import create_chat_model
from medrag.prompts import PromptRegistry
from medrag.storage.milvus import MedicalHybridKnowledgeBase
from medrag.storage.redis_session import RedisSessionStore

cfg = ConfigLoader().config
kb = MedicalHybridKnowledgeBase(cfg.milvus, cfg.embedding)
store = RedisSessionStore(cfg.redis)
prompts = PromptRegistry.from_config(cfg.prompts)      # 不传时使用内置模板的 stable 版本
rag = MultiDialogueRag(cfg.llm, cfg.multi_dialogue_rag, kb, store, prompts=prompts)
agent = MedicalAgent(cfg.llm, cfg.agent, kb, store, power_model=create_chat_model(cfg.llm), prompts=prompts)


async def main():
    # 一个实例服务所有会话，用 session_id 区分
    result = await rag.answer("我有点肚子痛，该怎么办？", return_document=True, session_id="U123")
    print(result["answer"])

    # 流式：原样透传 LangChain astream_events(v2) 事件，由外部转换成 SSE / WebSocket
    async for event in rag.stream("那需要忌口吗？", session_id="U123"):
        if event["event"] == "on_chat_model_stream":
            print(event["data"]["chunk"].content, end="")

    state = await agent.answer("我这两天肚子痛，还拉肚子", session_id="U456")
    print(state["final_answer"] or state["ask_messages"].asked_messages[-1][-1].content)

    await kb.close()
    await store.close()

asyncio.run(main())
```

不依赖 LangChain 的模型客户端（核心层使用）：

```python
from medrag.llm import OpenAICompatEmbedding, OpenAICompatLLM

llm = OpenAICompatLLM(cfg.llm)
result = await llm([{"role": "user", "content": "高血压有哪些症状"}])
result.text, result.reasoning, result.usage.completion_tokens

emb = OpenAICompatEmbedding(cfg.embedding.text_dense)
vectors = await emb(["文本1", "文本2"])     # 按 max_batch 分批、按 concurrency 并发，顺序与输入一致
```

## 提示词

提示词放在 `src/medrag/prompts/templates/<名称>/` 下，每个版本一个文件，设计见 [docs/design/prompt-management.md](docs/design/prompt-management.md)：

```bash
python -m medrag.prompts list                    # 列出提示词、版本、状态与别名
python -m medrag.prompts show basic_rag          # 查看 stable 版本的原文
python -m medrag.prompts check                   # 校验已发布版本没有被改动
python -m medrag.prompts lock                    # 新版本发布后登记哈希
```

- 迁移自 v2 的 15 个提示词都是 `v1`（`format: fstring`，与 v2 原文逐字一致，有测试保证）。新版本默认使用 Jinja2 语法。
- 已发布（`status: released`）的版本不能修改：要改就复制成新版本，改完在 dev 集上对比，再把 `stable` 别名指过去。
- 版本由配置选择：`prompts.versions: {basic_rag: v2}`；自定义模板目录放在 `prompts.dirs`，同名提示词优先使用自定义的。

## 测试

```bash
pytest tests -q                                  # 不需要任何外部服务；集成测试在服务不可用时自动跳过
bash deploy/start.sh && MEDRAG_REQUIRE_SERVICES=1 pytest tests -q   # 要求集成测试必须执行（里程碑验收时用）
```

- 单测里的模型请求都走 `medrag.testing.fake_openai`，不需要密钥和网络；存储走降级实现，数据文件放在 pytest 的临时目录。
- 端到端、会话存储、SQL、图的用例在降级实现和真实服务上各跑一遍同一组断言（`tests/test_backends.py`、`test_session_store.py`、`test_apps_e2e.py`）。
- `tests/test_layering.py` 检查核心层没有 import LangChain（只允许 `apps/`、`llm/langchain_adapter.py`、`eval/scorers/`）。
- `tests/test_apps_e2e.py` 跑通入库 → 检索 → SimpleRAG → 多轮 RAG（含流式）→ 检索图 → Agent 多轮。

里程碑回归（M0 新旧实现一致性）的用法见 `scripts/regression/m0_parity.py` 文件头，结果记录在 [docs/phase1/m0-report.md](docs/phase1/m0-report.md)。

## 注意事项

- ⚠️ 本项目仅供学习研究使用，不能替代专业医疗建议。任何医疗决策都应咨询专业医生。
- 医疗数据需符合相关法规，建议在私有环境部署。
- v2 README 中的"检索效果对比"等数字没有可复现的评测记录，已移除；之后的效果数字一律以 `docs/phase1/` 中可复现的评测结果为准。

## 许可证

MIT，详见 [LICENSE](LICENSE)。
