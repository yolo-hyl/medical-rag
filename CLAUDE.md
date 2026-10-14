# CLAUDE.md

## 提交规范 (Commit Conventions)

本仓库对 Claude Code 生成的 git 提交采用以下约定:

- 提交作者 (author) 使用仓库维护者的身份：`hy <zjl4376@gmail.com>`。
- 提交信息不需要附加 `Co-Authored-By: Claude ...` 或 `Claude-Session: ...` 署名行。
- Pull request 描述不需要附加 "Generated with Claude Code" 相关署名行。
- 提交信息 (commit message) 使用中文书写。

## 项目概况

医疗领域的 RAG 库。正在从 v2（以 QA 数据为中心的应用）重构为像 LightRAG 那样分层抽象、专攻医疗的通用 RAG 库。

- **开发分支**：`v2.0.0-rebuild`。`v2.0.0` 是重构前的基线。
- **当前状态**：M0（搬家）已完成，包名为 `medrag`，代码在 `src/medrag/`；下一步是 M1（统一 Schema 与存储抽象）。M0 门槛里的"重合率"一项待维护者确认（D17）。动手之前先看 `docs/phase1/plan.md`，确认当前处于哪个里程碑。
- **过渡期的类型**：M1 之前检索结果仍用 `medrag.types.LegacyDocument`（字段与 LangChain Document 相同），知识库仍是 v2 形态的 `MedicalHybridKnowledgeBase`；新代码不要在这些过渡类型上继续扩展功能，扩展放到 M1 的 Document / Chunk / BaseVectorStorage 上。
- **Python 3.12**（D19，FalkorDBLite 的要求）。
- **服务可以不配地址**（D18）：Milvus / Redis / PostgreSQL / Neo4j 没有配置地址时，分别降级为 Milvus Lite / fakeredis / SQLite / FalkorDBLite，选择逻辑只在 `src/medrag/storage/backends.py`。包内默认 `storage.yaml` 不配地址；连接 `deploy/` 的服务用 `deploy/storage.yaml` 覆盖。新写的存储代码必须同时支持两种后端，并在两种后端上各测一遍。
- **文档入口**：`docs/README.md`。架构在 `docs/architecture.md`，方向性决策在 `docs/decisions.md`。

## 目录速览

| 路径 | 说明 |
| --- | --- |
| `src/medrag/` | 库本体。分层与目录见 `docs/architecture.md` 第 2、3 节（第 3 节末尾有 M0 现状对照表） |
| `src/medrag/prompts/templates/` | 版本化提示词，`<名称>/<版本>.<语言>.yaml`；`prompts.lock` 登记已发布版本的哈希 |
| `src/medrag/testing/fake_openai.py` | 本地假 OpenAI 兼容服务，测试和离线联调用 |
| `scripts/regression/` | 里程碑回归脚本（M0：新旧实现一致性） |
| `scripts/` | 01–08 号使用示例（构建词表 → 入库 → 检索 → RAG → 评测 → 多轮 → Agent），也是唯一的试用 / 联调入口；读取 `scripts/conf/` 下的用户自定义配置（经 `scripts/_common.py` 加载），在仓库根目录运行 |
| `deploy/` | Milvus / PostgreSQL / Redis / Neo4j 的 docker compose；`deploy/storage.yaml` 是连接它们的存储配置 |
| `.medrag/` | 降级实现的本地数据（Milvus Lite / SQLite / FalkorDBLite），**不进 git** |
| `software/` | 前后端示例应用。第一阶段冻结（D11）；backend 用的是 v2 异步化之前的接口，目前跑不起来，M3 之后统一适配 |
| `data/` | `qa_50000.jsonl`、`eval/new_qa_200.jsonl`（git lfs）；`GraphRAG-Bench/` 已提交 |
| `data/benchmarks/`、`runs/` | 下载的数据集和评测产出，**不进 git** |
| `docs/` | 架构、规划、决策、设计文档 |

## 常用命令

```bash
git lfs pull                                   # 拉取 data/ 下的大文件
pip install -e ".[all]"                        # 在仓库根目录安装（Python >= 3.12）；extras 见 pyproject.toml
pip install numpy cython && pip install --no-build-isolation -e ".[zh]"   # 自管理词表的中文 BM25（pkuseg）
pytest tests -q                                # 不需要任何服务：存储走降级实现；@pytest.mark.services 的用例在服务不可用时自动跳过
bash deploy/start.sh                           # 启动真实服务；Docker Hub 429 时在 deploy/.env 设 REGISTRY=mirror.gcr.io
MEDRAG_REQUIRE_SERVICES=1 pytest tests -q      # 验收时用：集成测试必须执行（降级实现与真实服务各跑一遍）
export MEDRAG_CONFIG_OVERRIDES=deploy/storage.yaml   # 让 scripts/ 等连接 deploy/ 启动的服务
python -m medrag.prompts check                 # 校验已发布提示词未被修改（单测里也会跑）
python -m medrag.testing.fake_openai --port 8999   # 没有密钥时的假模型服务，base_url 指向 http://127.0.0.1:8999/v1
```

- 不要同时安装 `stopwords` 和 `stopwords-zh`：两者写同一个 `stopwords/` 目录，后装的会覆盖；只需要 `stopwords-zh`。
- `@pytest.mark.services("postgres")` 的参数是需要的服务名（milvus / redis / postgres / neo4j），不写时默认 milvus + redis。
- 降级实现的已知坑：Milvus Lite 需要 pymilvus 2.6 + milvus-lite 3.x（D20），分词器只有 `standard` / `jieba`；fakeredis 的多个客户端要共用一个 `FakeServer` 才共享数据；redislite（FalkorDBLite）的异步 close 不落盘，`GraphDatabase.close` 里显式 `SAVE`。
- 在 `src/medrag/` 目录下直接运行 Python 时，包里的 `types.py` 会遮蔽标准库；用 `python -P` 或换个目录执行。

模型服务通过环境变量提供密钥（配置里写 `api_key_env`，默认 `DASHSCOPE_API_KEY`；本地 vLLM / Ollama 可以不写）。中间件的密码由 `deploy/start.sh` 生成到 `deploy/.env`，`ConfigLoader` 会自动加载。配置目录可用环境变量 `MEDRAG_CONFIG_DIR` 替换。

## 架构约定（新代码必须遵守）

1. **只用异步**：IO 方法一律写成 `async def`，不保留同步版本；CPU 密集的操作用 `asyncio.to_thread`。
2. **依赖注入**：存储、模型客户端、提示词注册表都由调用方创建后传入，组件内部不自己创建；组件只接收自己用到的那一段配置，不接收整个 `AppConfig`。
3. **模型层**：LLM 和 Embedding 统一通过 `openai` SDK（`AsyncOpenAI`）以 OpenAI 兼容协议访问，`base_url` 和密钥环境变量可配置（D2）。核心层**不 import LangChain / LangGraph**，例外只有 `apps/`、`llm/langchain_adapter.py`（给应用层构造 ChatOpenAI）和 `eval/scorers/`（封装 ragas），`tests/test_layering.py` 会检查。
4. **存储细节不外泄**：对外只暴露 `QueryParam`（mode / top_k / filters 等）。`anns_field`、`metric_type`、Milvus `expr` 只能出现在 `storage/milvus.py` 里，不能交给 LLM 或上层应用。
5. **提示词放在文件里**：不在代码里写提示词字符串，通过注入的 `PromptRegistry` 按名称获取（未注入时用 `PromptRegistry.builtin()`）。已发布（released）的版本不能修改，要改就新建一个版本，发布后执行 `python -m medrag.prompts lock`。v1 是从 v2 原样迁移的 f-string 版本，新版本默认 Jinja2。详见 `docs/design/prompt-management.md`。
6. **可复现**：评测必须写出 RunManifest（git commit、配置哈希、提示词版本、数据子集版本、模型、种子）；数据子集由固定种子的脚本生成，并附带 manifest。
7. **注释和文档用中文**，专有名词保留原文。注释写"为什么"，不复述代码本身。

## 数据与评测

- 三层数据见 `docs/phase1/data.md`：
  - A 层：`huatuo_qa:a10k`，回归门槛，指标是 Hit@k；
  - B 层：GraphRAG-Bench medical，用来对标 LightRAG；
  - C 层：`medqa_zh:c_mini`，中文主评测。
- **dev 集用于迭代，test 集用于验收**。提示词和检索参数都不要在 test 集上调。
- 每个里程碑都必须先通过 A 层回归（`docs/phase1/plan.md` 中的门槛）才能进入下一步。
- Milvus 批量写入后必须先 flush 再建索引、再检索：未封存的增长段只能走临时索引，召回会明显偏低，且结果随封存时机变化（M0 实测）。
- Milvus 的 HNSW 检索加上并发写入，同一实现重跑的 top-10 重合率约 91%；判断"行为是否一致"时用 `scripts/regression/m0_parity.py verify`（逐行比对数据和查询编码），不要只看检索结果。
- 注意数据里已知的坑：
  - `new_qa_200` 实际是 198 条，其中 `new_answer` 是 LLM 改写的，**不能当参考答案**；
  - CMB 的 test 集没有答案；
  - MedQA 中文教材的 OCR 错字很多，统一用 LLM 纠错（D13），丢失的数字不补；
  - GraphRAG-Bench 的证据句基本不在原文里，只能做语义匹配。
- 中英文数据暂时隔离（D16）：一个 workspace 只放一种语言。
- 大文件（数据集、向量词表、评测产出）不提交。下载通过 `medrag data download`（M3 实现）完成，并校验 sha256。

## 需要先和维护者确认的事

以下情况不要自行决定，先提出选项和建议，等维护者确认：

- `docs/decisions.md` 中状态为"待定"的事项，或者需要推翻"已定"的决策；
- 新增第三方依赖、更换存储或模型服务商；
- 修改 Milvus collection schema、评测协议（数据子集、指标、门槛）或已发布的提示词；
- 删除数据、删除分支、强制推送；
- 任何超出当前里程碑范围的改动（记入 `docs/decisions.md` 的待办即可）。

架构或决策有变化时，同步更新 `docs/architecture.md`、`docs/decisions.md` 和本文件。
