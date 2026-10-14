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
- **当前状态**：代码仍是 v2 旧结构（`src/MedicalRag/`），正在按第一阶段规划迁移到 `src/medrag/`。动手之前先看 `docs/phase1/plan.md`，确认当前处于哪个里程碑（M0–M4）。
- **文档入口**：`docs/README.md`。架构在 `docs/architecture.md`，方向性决策在 `docs/decisions.md`。

## 目录速览

| 路径 | 说明 |
| --- | --- |
| `src/MedicalRag/` | v2 旧代码：core / rag / agent / embed / eval / prompts / config。迁移完成后删除 |
| `src/medrag/` | 重构目标，结构见 `docs/architecture.md` 第 3 节 |
| `scripts/` | 01–08 号使用示例脚本（构建词表 → 入库 → 检索 → RAG → 评测 → 多轮 → Agent） |
| `debug/` | 本地联调脚本，使用独立的配置目录 `debug/conf/` |
| `deploy/` | Milvus / PostgreSQL / Redis / Neo4j 的 docker compose |
| `software/` | 基于本包的前后端示例应用。第一阶段冻结（D11） |
| `data/` | `qa_50000.jsonl`、`eval/new_qa_200.jsonl`（git lfs）；`GraphRAG-Bench/` 已提交 |
| `data/benchmarks/`、`runs/` | 下载的数据集和评测产出，**不进 git** |
| `docs/` | 架构、规划、决策、设计文档 |

## 常用命令

```bash
git lfs pull                                   # 拉取 data/ 下的大文件
bash deploy/start.sh                           # 启动 Milvus / PostgreSQL / Redis / Neo4j（集成测试和脚本需要）
cd src && pip install -e .                     # v2 旧结构的安装方式；M0 之后改为在仓库根目录 pip install -e ".[all]"
pytest tests -v                                # 当前测试都依赖 deploy/ 里的服务
python scripts/02_ingest_data.py               # 示例：入库前 100 条 QA
```

模型服务通过环境变量提供密钥，默认是 `DASHSCOPE_API_KEY`。中间件的密码由 `deploy/start.sh` 生成到 `deploy/.env`，`ConfigLoader` 会自动加载。

## 架构约定（新代码必须遵守）

1. **只用异步**：IO 方法一律写成 `async def`，不保留同步版本；CPU 密集的操作用 `asyncio.to_thread`。
2. **依赖注入**：存储、模型客户端、提示词注册表都由调用方创建后传入，组件内部不自己创建；组件只接收自己用到的那一段配置，不接收整个 `AppConfig`。
3. **模型层**：LLM 和 Embedding 统一通过 `openai` SDK（`AsyncOpenAI`）以 OpenAI 兼容协议访问，`base_url` 和密钥环境变量可配置（D2）。核心层（`apps/` 以外）**不 import LangChain / LangGraph**。
4. **存储细节不外泄**：对外只暴露 `QueryParam`（mode / top_k / filters 等）。`anns_field`、`metric_type`、Milvus `expr` 只能出现在 `storage/milvus.py` 里，不能交给 LLM 或上层应用。
5. **提示词放在文件里**：不在代码里写提示词字符串，通过 `PromptRegistry` 按名称获取。已发布（released）的版本不能修改，要改就新建一个版本。详见 `docs/design/prompt-management.md`。
6. **可复现**：评测必须写出 RunManifest（git commit、配置哈希、提示词版本、数据子集版本、模型、种子）；数据子集由固定种子的脚本生成，并附带 manifest。
7. **注释和文档用中文**，专有名词保留原文。注释写"为什么"，不复述代码本身。

## 数据与评测

- 三层数据见 `docs/phase1/data.md`：
  - A 层：`huatuo_qa:a10k`，回归门槛，指标是 Hit@k；
  - B 层：GraphRAG-Bench medical，用来对标 LightRAG；
  - C 层：`medqa_zh:c_mini`，中文主评测。
- **dev 集用于迭代，test 集用于验收**。提示词和检索参数都不要在 test 集上调。
- 每个里程碑都必须先通过 A 层回归（`docs/phase1/plan.md` 中的门槛）才能进入下一步。
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
