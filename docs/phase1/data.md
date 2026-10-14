# 第一阶段数据：观察、子集与清洗

> 所有数据集都已实际下载并检查过（2026-10），本文的数字都来自这次检查。
> 方向（已定，D6）：**先用子集，子集要清洗**；上全量数据时再另想办法，比如批量清洗、抽样质检。

---

## 1. 数据集观察

### 1.1 A 层：huatuo QA（仓库内，git lfs）

| 项 | 观察 |
| --- | --- |
| 文件 | `data/qa_50000.jsonl`（166 MB，50,000 行）、`data/eval/new_qa_200.jsonl` |
| 字段 | `question / answer / text / question_hash`；所有行都满足 `text == question + "\n\n" + answer`，是冗余字段 |
| 长度 | answer 平均 541 字，中位数 629，最长 18,848；超过 2,000 字的有 347 条，超过 8,000 字的有 10 条 |
| 重复 | 有 57 个 `question_hash` 重复 |
| 来源 | 是 Huatuo-Encyclopedia-QA 的子集，问题能对上 50,879 条 |
| 内容 | 混合了疾病百科、药品说明书和患者咨询三种文体；用法用量 930 条、副作用 606 条、禁忌 495 条 |
| 噪声 | 非医疗内容（如"水松的栽培要点"）；机翻残句（如"该反应在其他由引起的疾病中也可见"）；药品条目的模板话术（如"注意：同种药品可由于不同的包装规格有不同的用法或用量。本文只供参考……"） |
| 评测集 | **实际是 198 条**；198 个 `question_hash` 全部能在语料里找到，可以直接算 Hit@k 和 MRR |
| 风险 | `new_answer` 是 qwen3:32b 改写的，会补充原文没有的事实，**不能当参考答案**（D7） |

### 1.2 B 层：GraphRAG-Bench Medical（仓库内 `data/GraphRAG-Bench/`）

| 项 | 观察 |
| --- | --- |
| 语料 | `medical.json` 只有 1 条记录，内部用 `\n` 分隔，实际是 **44 篇 NCCN Guidelines for Patients**（英文），去重后 41 篇（AML、CML、ALL 各重复一次） |
| 规模 | 174,610 词（约 23 万 token）；单篇 161 到 11,599 词 |
| 噪声 | 每篇开头都混有目录和页码，例如 "7 Adrenal glands 8 Adrenal tumors 9 Signs and symptoms…"；章节标题被压平进了正文 |
| 题目 | 2,062 题：Fact 1,098 / Reasoning 509 / Summarize 289 / Creative 166；参考答案平均长度为 11 / 17 / 30 / 121 词 |
| 证据 | 8,175 条证据句里只有 34 条能在原文中逐字找到，**检索评测只能用语义匹配**（官方脚本用 LLM 判断） |
| 基线 | 论文 Table 2 / Table 6（GPT-4o-mini，bge-large-en-v1.5）：Fact ACC 方面，RAG w/ rerank 64.73、LightRAG 63.32、HippoRAG2 66.28；Reasoning ACC 方面分别为 58.64、61.32、61.98 |

### 1.3 C 层：MedQA 中文（Google Drive，压缩包 126 MB，解压后 374 MB）

| 项 | 观察 |
| --- | --- |
| 题库 | Mainland 划分为 train 27,400 / dev 3,425 / test 3,426；默认 5 个选项，另有 4 选项版本；题干平均 46 字 |
| 科目 | test 集的 `meta_info` 有 25 类：历年真题 733、儿科疾病 389、精神神经 221、病理学 188、生理学 173、运动系统 151、医学免疫学 134、生物化学 127、药理学 125、医学微生物学 119、消化系统 119、代谢内分泌 116、其他疾病 105、血液系统 96、传染病 91、女性生殖 89、卫生法规 85、医学心理学 76、心血管 68、医学伦理学 58、模拟试题 57、呼吸系统 51、泌尿系统 34、风湿免疫 14、预防医学 7 |
| 教材 | 33 本中文教材，共 1,473 万字、116,216 个段落，段落平均约 125 字；最大的是内科学（104 万字） |
| 结构 | 章节标题可以识别，例如 "第二章急性上呼吸道感染和急性气管－支气筐炎"、"［实验室检查］" |
| 噪声 | **OCR 错字很多**：如"支气筐炎"（应为支气管炎）、"急性癌密性咽峡炎"（应为疱疹性）、"殖密"（应为疱疹）、"黄色服性分泌物"（应为脓性）、"体温可达到℃以上"（数字丢失） |

### 1.4 储备数据（只做 loader）

| 数据 | 观察 |
| --- | --- |
| MIRAGE（4.2 MB） | medqa 1,273 / medmcqa 4,183 / pubmedqa 500 / bioasq 618 / mmlu 1,089 |
| MedRAG/textbooks（202 MB） | 18 本英文教材，已经切成 12.6 万个块，每行 `{id, title, content}`；有 PDF 抽取噪声（版权页、双栏串行） |
| CMB（179 MB） | **test 集 11,200 题不带答案**；val 280 题有答案；train 26.9 万题；CMB-Clin 74 个病例 |
| Huatuo-Encyclopedia-QA（604 MB） | train 362,420 行；同一个问题有多种问法；答案平均 540 字 |

---

## 2. 子集定义

每个子集都由**固定种子 + 可复现脚本**生成，同时输出 `manifest.json`。manifest 记录源文件的 sha256、所用清洗规则及其版本、各步骤的计数，以及子集本身的 sha256。这份 manifest 的哈希会写进评测的 RunManifest，作为"数据版本"。

| 子集 | 语料 | 题目 | 规模 | 用途 |
| --- | --- | --- | --- | --- |
| **A `huatuo_qa:a10k`** | 清洗后的 qa_50000 中，198 篇标准文档全部保留，再随机抽 9,802 篇干扰文档（seed=42） | new_qa_200（198 题，用 `new_question` 检索） | 1 万篇，约 550 万字 | 回归门槛（Hit@k），入库幂等测试 |
| **B `graphrag_bench:medical`** | 41 篇（去重 + 去掉开头目录） | dev：按题型分层抽 500 题；test：全部 2,062 题 | 约 17 万词 | 对标 LightRAG |
| **C `medqa_zh:c_mini`** | 内科学、儿科学、病理学，共 3 本，约 201 万字 | 科目能对上这 3 本书的题：消化、心血管、呼吸、泌尿、代谢内分泌、血液（共 484）+ 儿科疾病 389 + 病理学 188，test 集合计 **1,061 题**；dev 集按同样规则筛选 | 3 本书 | 中文主评测、提示词迭代 |
| C `medqa_zh:c_sub` | 在 c_mini 基础上加神经病学、精神病学、生理学、药理学、传染病学、妇产科学，共 9 本，约 511 万字 | 在 c_mini 基础上加精神神经 221、生理学 173、药理学 125、传染病 91、女性生殖 89，test 集合计 **1,760 题** | 9 本书 | c_mini 跑稳之后再扩大 |

说明：
- 用 `meta_info` 匹配教材只是近似的。比如"消化系统"里有一部分题目属于外科学。这个误差在报告中如实说明即可，不追求完美覆盖。
- **dev 集用于迭代，test 集用于验收**：提示词、切块、检索参数都在 dev 集上调，test 集只在里程碑验收时跑。
- 按 test 集 1,061 题算，准确率在 60% 左右时，95% 置信区间约为 ±2.9pp；c_sub 的 1,760 题约为 ±2.3pp。

---

## 3. 清洗规则

每条规则都是一个带 `name@version` 的函数，输入一条记录，输出处理后的记录或"丢弃 + 原因"，同时累计计数，最后生成清洗报告。规则一旦改动就升版本号，子集也随之生成新版本。

### A：huatuo QA

| 规则 | 动作 | 说明 |
| --- | --- | --- |
| `dedupe_hash@v1` | 按 `question_hash` 去重，保留第一条 | 57 组 |
| `strip_boilerplate@v1` | 去掉药品条目的模板话术 | 用正则维护话术列表 |
| `normalize_text@v1` | 全角半角统一、合并连续空白、去掉控制字符 | |
| `filter_non_medical@v1` | 识别非医疗内容（D14） | 建议用 LLM 判断，1 万条成本不高，然后人工抽检 100 条 |
| `flag_garbled@v1` | 机翻残句**只打标记，不删除** | 很难自动修复，留作后续数据质量任务 |

如果某篇标准文档被清洗规则丢掉，对应的评测题也一并删除，并在报告里列出来。

### B：GraphRAG-Bench

| 规则 | 动作 |
| --- | --- |
| `split_docs@v1` | 按 `\n` 切成多个 Document；`doc_id` 由标题生成 |
| `dedupe_content@v1` | 内容哈希去重（44 篇 → 41 篇） |
| `strip_toc@v1` | 去掉开头"数字 + 标题"组成的目录串 |

题目和参考答案一律不改，以便和论文结果直接对比。

### C：MedQA 中文教材

| 规则 | 动作 | 说明 |
| --- | --- | --- |
| `normalize_text@v1` | 全角半角统一、合并连续空白 | |
| `parse_headings@v1` | 识别"第X章 / 第X节 / ［…］"，构建章节路径 | 供 `SectionParagraphChunker` 使用 |
| `drop_noise_lines@v1` | 去掉页眉页脚、孤立的页码、图表标题残行 | |
| `ocr_fix_llm@v1` | LLM 逐段纠正 OCR 错字（D13，已定：统一用 LLM，不维护人工错字表） | 约束和抽检方式见下文 |
| `flag_missing_digits@v1` | 标记"可达到℃"这类数字丢失的段落 | 无法恢复，只做标记 |

**LLM 纠错的约束**（`ocr_fix_llm@v1`）：

- 提示词 `ocr_correct` 纳入版本管理（见 design/prompt-management.md）。纠错用的提示词版本和模型都写进子集的 manifest，换提示词或换模型都会产生新的数据版本。
- 只允许改错字，不允许改写、增删句子或补充内容。丢失的数字（如"可达到℃"）**不补**，只打标记，因为补出来的数字是编造的。
- 每段纠错之后都做机器校验，任一项不通过就保留原文，并把该段记进报告：
  - 纠错前后长度比在 0.95–1.05 之间；
  - 字符级编辑距离不超过原文的 5%；
  - 阿拉伯数字和单位序列保持一致。
- 每处修改都以 `原文片段 → 纠正片段` 的形式写进 diff 日志；随机抽 100 段人工检查，误改率超过 2% 就修改提示词并升级版本，然后重跑。
- 保留未纠错的原文副本。纠错前后的语料都可以入库，评测时作为对照（不是门槛）。
- 成本：c_mini 约 201 万字，输入和输出各一遍，按段落并发调用。

A 层的机翻残句（如"由引起的"）**不使用 LLM 纠错**，仍然只打标记：这类句子是缺词，不是错字，让 LLM 去补就等于往医疗数据里加事实。

---

## 4. 下载与存放

```bash
medrag data download huatuo_qa graphrag_bench medqa      # 第一阶段必需
medrag data download mirage medrag_textbooks cmb huatuo_encyclopedia   # 储备
medrag data build-subset huatuo_qa:a10k                  # 清洗 + 生成子集 + 写出 manifest
```

- 原始数据放在 `data/benchmarks/<name>/raw/`，子集放在 `data/benchmarks/<name>/subsets/<subset>/<version>/`。这两个目录都在 `.gitignore` 中，**不进 git**。
- 原始文件的 sha256 写在 `datasets/download.py` 里，下载后自动校验。
- 下载源：

| 数据集 | 来源 |
| --- | --- |
| huatuo_qa、graphrag_bench | 仓库自带（git lfs / 已提交） |
| medqa | Google Drive `1ImYUSLk9JbgHXOemfvyiDiirluZHPeQw`（用 gdown 下载） |
| mirage | GitHub `Teddy-XiongGZ/MIRAGE` 的 `benchmark.json` |
| medrag_textbooks | HF `MedRAG/textbooks` |
| cmb | HF `FreedomIntelligence/CMB` |
| huatuo_encyclopedia | HF `FreedomIntelligence/huatuo_encyclopedia_qa` |

- 全部下载约 1.4 GB；第一阶段必需的部分约 550 MB（主要是 MedQA）。
