# CLAUDE.md

## 提交规范 (Commit Conventions)

本仓库对 Claude Code 生成的 git 提交采用以下约定:

- 提交作者 (author) 使用仓库维护者的身份：`hy <zjl4376@gmail.com>`。
- 提交信息不需要附加 `Co-Authored-By: Claude ...` 或 `Claude-Session: ...` 署名行。
- Pull request 描述不需要附加 "Generated with Claude Code" 相关署名行。
- 提交信息 (commit message) 使用中文书写。

## 项目概况

医疗领域的 RAG 库，定位是**医学知识的可验证层**：把指南、说明书、教材等知识文本变成带条件、带有效期、带出处的陈述（Claim），然后在此之上：

- 按时间点和适用条件作答；
- 如实呈现分歧；
- 逐句可核查。

项目的方向是：论文（引子）→ 开源库（共享的内核 + 医疗领域包）→ 产品（医生助手、患者助手）。

- **开发分支**：`v2.0.0-rebuild`。`v2.0.0` 是重构前的基线。
- **文档**：只有两份。
  - `docs/solution.md` 是宏观方案：要解决的问题、系统框架、解决思路、产品方向、数据、已确定和待讨论的方向；
  - `docs/architecture.md` 是技术方案：分层、目录、数据模型、领域包、接口、存储、数据流和用法。

  动手之前先看这两份文档。
- **当前状态**：搬家（v2 → medrag）已完成，包名为 `medrag`，代码在 `src/medrag/`。系统框架和产品方向还在讨论，**开发阶段的划分尚未确定**，不要自行编排阶段、编号或门槛。
- **三层结构**：
  - 领域无关的内核；
  - 医疗领域包 `src/medrag/domains/medical/`；
  - 应用层 `src/medrag/apps/`。

  只做医疗，领域包只开医疗实际用到的扩展点，不为其他领域做预留。
- **过渡期的类型**：新的数据模型和存储抽象落地之前，检索结果仍用 `medrag.types.LegacyDocument`（字段与 LangChain Document 相同），知识库仍是 v2 形态的 `MedicalHybridKnowledgeBase`。新代码不要在这些过渡类型上继续扩展功能。
- **Python 3.12**（FalkorDBLite 的要求）。
- **服务可以不配地址**：Milvus / Redis / PostgreSQL / Neo4j 没有配置地址时，分别降级为 Milvus Lite / fakeredis / SQLite / FalkorDBLite。
  - 选择逻辑只在 `src/medrag/storage/backends.py`。
  - 包内默认的 `storage.yaml` 不配地址；连接 `deploy/` 的服务时，用 `deploy/storage.yaml` 覆盖。
  - 新写的存储代码必须同时支持两种后端，并在两种后端上各测一遍。

## 目录速览

| 路径 | 说明 |
| --- | --- |
| `src/medrag/` | 库本体。分层与目录见 `docs/architecture.md` 第 2、3 节，现状与差距见第 16 节 |
| `src/medrag/domains/medical/` | 医疗领域包（待建）：schema、文档档案、术语导入、优先级与冲突规则、条件判定、确定性工具、医学提示词 |
| `src/medrag/prompts/templates/` | 版本化提示词，`<名称>/<版本>.<语言>.yaml`；`prompts.lock` 登记已发布版本的哈希 |
| `src/medrag/testing/fake_openai.py` | 本地假 OpenAI 兼容服务，测试和离线联调用 |
| `scripts/regression/` | 回归脚本（`m0_parity.py`：搬家时的新旧实现一致性） |
| `scripts/` | 01–08 号使用示例（构建词表 → 入库 → 检索 → RAG → 评测 → 多轮 → Agent），也是唯一的试用 / 联调入口。读取 `scripts/conf/` 下的用户自定义配置（经 `scripts/_common.py` 加载），在仓库根目录运行 |
| `deploy/` | Milvus / PostgreSQL / Redis / Neo4j 的 docker compose；`deploy/storage.yaml` 是连接它们的存储配置 |
| `.medrag/` | 降级实现的本地数据（Milvus Lite / SQLite / FalkorDBLite），**不进 git** |
| `software/` | 前后端示例应用，已冻结，到产品阶段再适配。backend 用的是 v2 异步化之前的接口，目前跑不起来 |
| `data/` | `qa_50000.jsonl`、`eval/new_qa_200.jsonl`（git lfs）；`GraphRAG-Bench/` 已提交 |
| `data/benchmarks/`、`runs/` | 下载的数据集和评测产出，**不进 git** |
| `docs/` | `solution.md`（宏观方案）和 `architecture.md`（技术方案），只保留这两份 |

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

- 不要同时安装 `stopwords` 和 `stopwords-zh`：两者写同一个 `stopwords/` 目录，后装的会覆盖。只需要 `stopwords-zh`。
- `@pytest.mark.services("postgres")` 的参数是需要的服务名（milvus / redis / postgres / neo4j），不写时默认 milvus + redis。
- 降级实现的已知坑：
  - Milvus Lite 需要 pymilvus 2.6 + milvus-lite 3.x，分词器只有 `standard` / `jieba`；
  - fakeredis 的多个客户端要共用一个 `FakeServer` 才共享数据；
  - redislite（FalkorDBLite）的异步 close 不落盘，`GraphDatabase.close` 里显式 `SAVE`；
  - FalkorDB 只支持 openCypher 子集，图相关代码只用两边都支持的语法。
- 在 `src/medrag/` 目录下直接运行 Python 时，包里的 `types.py` 会遮蔽标准库；用 `python -P` 或换个目录执行。

模型服务通过环境变量提供密钥：配置里写 `api_key_env`，默认 `DASHSCOPE_API_KEY`；本地 vLLM / Ollama 可以不写。中间件的密码由 `deploy/start.sh` 生成到 `deploy/.env`，`ConfigLoader` 会自动加载。配置目录可以用环境变量 `MEDRAG_CONFIG_DIR` 替换。

## 架构约定（新代码必须遵守）

1. **只用异步**：IO 方法一律写成 `async def`，不保留同步版本；CPU 密集的操作和只有同步驱动的库用 `asyncio.to_thread`。
2. **依赖注入**：存储、模型客户端、提示词注册表、领域包都由调用方创建后传入，组件内部不自己创建；组件只接收自己用到的那一段配置，不接收整个 `AppConfig`。按配置组装默认组件，只在组装根（`build.py`）里做。
3. **模型层**：LLM 和 Embedding 统一通过 `openai` SDK（`AsyncOpenAI`）以 OpenAI 兼容协议访问，`base_url` 和密钥环境变量可配置。核心层**不 import LangChain / LangGraph**，例外只有三处，`tests/test_layering.py` 会检查：
   - `apps/`；
   - `llm/langchain_adapter.py`（给应用层构造 ChatOpenAI）；
   - `eval/scorers/`（封装 ragas）。
4. **三层依赖**：
   - 内核不 import `domains/` 和 `apps/`，也不按领域名分支；
   - 内核代码里不出现医学概念（实体类型名、谓词名、术语体系、具体工具），这些由领域包提供；
   - 领域包只依赖内核的数据模型和接口（`types.py`、`base.py`）；
   - 能声明的用 YAML 声明，声明不了的才写插件；
   - 没有实际内容的扩展点不加。
5. **存储细节不外泄**：对外只暴露 `QueryParam`（mode / top_k / filters / as_of / context 等）。
   - `anns_field`、`metric_type`、Milvus `expr` 只能出现在 `storage/milvus.py`；
   - Cypher 只能出现在 `storage/graph*.py`；
   - SQL 只能出现在 `storage/` 下。

   这些细节都不能交给 LLM 或上层应用。
6. **提示词放在文件里**：不在代码里写提示词字符串，通过注入的 `PromptRegistry` 按名称获取（未注入时用 `PromptRegistry.builtin()`）。
   - 已发布（released）的版本不能修改，要改就新建一个版本，发布后执行 `python -m medrag.prompts lock`。
   - v1 是从 v2 原样迁移的 f-string 版本，新版本默认用 Jinja2。
   - 通用模板放在内核，医学指令和示例放在领域包。

   详见 `docs/architecture.md` 第 12 节。
7. **知识层的存储**：
   - 关系库（PostgreSQL / SQLite）是唯一真相源；Milvus 和图数据库（Neo4j / FalkorDBLite）都是派生索引，必须能从关系库重建；
   - 状态变更和对应的治理事件在同一事务里写入；
   - 回答路径只读，写入只经由入库流程或后台治理。
8. **安全策略在应用层**：内核和领域包只产出事实和风险信号；拦截、改写、转人工这类策略放在 `apps/`。
9. **可复现**：评测必须写出 RunManifest，包括 git commit、配置哈希、领域 schema 哈希、提示词版本、数据子集版本、模型、治理事件序号、种子。数据子集由固定种子的脚本生成，并附带 manifest。
10. **注释和文档用中文**，专有名词保留原文。注释写"为什么"，不复述代码本身。

## 数据与评测

- 数据需求、获取渠道和版权原则见 `docs/solution.md` 第 7 节；评测集与指标见 `docs/architecture.md` 第 14 节。
  - 主评测：切片语料上的时间切片评测。切片是糖尿病、高血压、心血管一级预防，中文优先。
  - huatuo QA：文本块通道的回归基准，指标是 Hit@k。
  - MedQA 中文：解释性文本和静态知识对照。
  - GraphRAG-Bench medical：在开源库方向上对标 LightRAG。
- **dev 集用于迭代，test 集用于验收**。提示词和检索参数都不要在 test 集上调。
- 改动存储或文本块检索时，先在文本块回归基准上做回归。门槛分两层：
  - 硬门槛：代码行为是否一致（逐行比对数据和查询编码）；
  - 软门槛：检索指标的差异，不超过同一实现重跑的波动再加 2pp。

  不要用"重合率接近 100%"这类比 Milvus 自身波动还紧的标准。
- Milvus 批量写入后必须先 flush 再建索引、再检索。未封存的增长段只能走临时索引，召回会明显偏低，结果还会随封存时机变化（搬家时实测）。
- Milvus 的 HNSW 检索加上并发写入，同一实现重跑的 top-10 重合率约 91%。判断"行为是否一致"时用 `scripts/regression/m0_parity.py verify`（逐行比对数据和查询编码），不要只看检索结果。
- 数据里已知的坑：
  - `new_qa_200` 实际是 198 条，其中 `new_answer` 是 LLM 改写的，**不能当参考答案**；
  - CMB 的 test 集没有答案；
  - MedQA 中文教材的 OCR 错字很多，统一用 LLM 纠错：只改错字，不补内容，丢失的数字不补；
  - GraphRAG-Bench 的证据句基本不在原文里，只能做语义匹配；
  - 期刊指南和商业说明书库只能用于研究，评测集对外只发布标注和出处，不分发原文。
- 一个 workspace 只放一种语言；中文优先，英文后续支持。
- 大文件（数据集、向量词表、评测产出）不提交。下载通过 `medrag data download` 完成并校验 sha256（待实现）。

## 需要先和维护者确认的事

以下情况不要自行决定，先提出选项和建议，等维护者确认：

- `docs/solution.md` 中"倾向（待确认）"和"待讨论"的事项，或者需要推翻"已确定"的方向；
- 开发阶段的划分：顺序、节点、验收标准；
- 新增第三方依赖，更换存储或模型服务商；
- 修改以下内容：
  - Milvus collection schema、关系库表结构、图投影的 schema；
  - 领域 schema；
  - 评测协议（数据子集、指标、门槛）；
  - 已发布的提示词。
- 删除数据、删除分支、强制推送；
- 超出当前讨论范围的改动（先记进 `docs/solution.md` 的"待讨论"即可）。

架构或方向有变化时，同步更新 `docs/solution.md`、`docs/architecture.md` 和本文件。
