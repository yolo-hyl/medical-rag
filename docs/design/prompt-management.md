# 提示词管理设计

> 目标：让提示词**好改、有版本、能迭代、能对比**。
> 现状：v2 的提示词全部以 Python 字符串的形式写在 `prompts/templates.py` 里，用 `str.format` 渲染。存在四个问题：
> - 改一处就会悄悄影响所有调用方；
> - 无法知道某次评测用的是哪一版提示词；
> - JSON 示例里的花括号必须写成 `{{ }}` 转义；
> - 中文和英文无法区分。

---

## 1. 设计原则

1. **提示词是数据，不是代码**：每个版本单独存一个文件，可读，可以在 PR 里看 diff，不懂 Python 的人也能改。
2. **已发布的版本不可修改**：要改就新建一个版本。改动已发布版本的文件，测试会失败。
3. **由配置选择版本**：代码只按名称取提示词，具体用哪个版本由配置决定，实验时用 overrides 切换版本即可，不用改代码。
4. **每次调用都有记录**：渲染结果携带 `name@version#sha256`，写进 trace 和评测的 RunManifest。
5. **按语言区分**：同一个提示词的中文和英文版本分开维护，**找不到对应语言时直接报错**，不会退回另一种语言。

---

## 2. 文件布局

```
src/medrag/prompts/templates/
└── rag_answer/
    ├── _meta.yaml          # 该提示词的说明、变量定义、别名
    ├── v1.zh.yaml          # 从 v2 的 BASIC_RAG_* 原样迁移
    ├── v1.en.yaml          # 为 GraphRAG-Bench 新增
    └── v2.zh.yaml          # 迭代出的新版本（从 draft 开始）
```

`_meta.yaml`：

```yaml
name: rag_answer
description: 基于检索证据回答问题，并用 [n] 标注引用
variables:                           # 渲染时校验：缺少必填变量或传了未声明的变量都会报错
  question: {required: true,  description: 用户问题（改写后）}
  contexts: {required: true,  description: "证据列表，每项有 content / title / section"}
  history:  {required: false, description: 对话历史摘要}
aliases:
  stable: v1                         # 默认配置使用的版本
  latest: v2
```

单个版本，例如 `v2.zh.yaml`：

```yaml
version: v2
parent: v1                           # 从哪个版本演化而来，用于 diff 和对比报告
status: draft                        # draft（可修改）| released（锁定）| deprecated
author: hy
created: 2026-10-15
changelog: |
  - 要求每条结论后用 [n] 标注证据编号
  - 证据不足时明确说明，不要用常识补全
params: {temperature: 0.1}           # 建议参数；可以被调用方覆盖
messages:
  - role: system
    content: |
      你是一名专业的医学知识助手，只能依据给出的参考资料回答。
  - role: user
    content: |
      # 参考资料
      {% for c in contexts %}
      [{{ loop.index }}] {{ c.section or c.title }}
      {{ c.content }}
      {% endfor %}
      # 问题
      {{ question }}
      # 要求
      1. 每条结论后标注证据编号，例如 [1][3]。
      2. 参考资料不足以回答时，直接说明缺少哪些信息。
```

模板语法用 **Jinja2 + StrictUndefined**（D15，待确认）：
- 变量缺失时直接报错，不会渲染出空字符串；
- JSON 示例可以直接写，不用转义花括号；
- 支持循环，证据列表的格式可以写在模板里，不用在代码里拼字符串。

---

## 3. 使用方式

**代码中**：

```python
prompt = registry.get("rag_answer", lang="zh")           # 版本由配置决定
messages = prompt.render(question=q, contexts=ctx)       # 返回 messages，并附带 ref
result = await llm(messages, **prompt.params)
trace.prompts.append(prompt.ref)                         # PromptRef(name, version, lang, sha256)
```

**配置中**：

```yaml
prompts:
  dirs: []                    # 自定义模板目录，同名提示词优先使用这里的
  versions:
    rag_answer: stable        # 可以写别名或具体版本
    query_rewrite: v1
```

**做实验时**：

```bash
medrag eval run --dataset medqa_zh:c_mini --split dev --set prompts.versions.rag_answer=v1
medrag eval run --dataset medqa_zh:c_mini --split dev --set prompts.versions.rag_answer=v2
medrag eval compare runs/medqa_zh/<run_v1> runs/medqa_zh/<run_v2>
```

---

## 4. 命令行工具

M0 先提供 `python -m medrag.prompts list / show / check / lock`；`diff / new / release` 随评测框架在 M3 补齐，届时统一为 `medrag prompts ...`。

| 命令 | 作用 |
| --- | --- |
| `medrag prompts list` | 列出所有提示词，以及每个提示词的版本、状态和别名 |
| `medrag prompts show rag_answer@v2 --lang zh` | 用示例变量渲染一次，查看效果 |
| `medrag prompts diff rag_answer v1 v2` | 对比两个版本：messages 的文本 diff，以及 params 和变量的变化 |
| `medrag prompts new rag_answer --from v2` | 复制出 v3，状态为 draft |
| `medrag prompts release rag_answer v3` | 把状态改为 released，并写入 `prompts.lock` |
| `medrag prompts check` | 校验所有 released 版本的 sha 和 lock 文件一致（单测里也会调用） |

---

## 5. 迭代流程（建议）

1. `prompts new` 新建一个 draft 版本，在 **dev 集**上反复修改、评测。不要在 test 集上调提示词，否则会对测试集过拟合。
2. 用 `eval compare` 和当前的 stable 版本对比。只有满足以下两点才 release：主要指标的提升超出置信区间；逐题检查"前对后错"的题目，没有发现系统性退化。
3. release 之后，把 `_meta.yaml` 里的 `stable` 指向新版本，并在 test 集上跑一次，作为正式结果记录下来。

---

## 6. v2 提示词的迁移（M0 已完成）

- `templates.py` 里实际在用的 15 组提示词都迁移成了 `v1.zh.yaml`（状态为 released）；没有调用方的 `simple_qa` 没有迁移。
- 迁移版本标注 `format: fstring`，**原文一个字都没改**（包括 `{{ }}` 转义）。这样既与 v2 逐字一致，也能直接交给 LangChain 的 ChatPromptTemplate（`apps/` 里有提示词要和 MessagesPlaceholder 组装）。新版本默认 `format: jinja2`；两种语法由注册表统一渲染。
- **一致性测试**：`tests/test_prompts.py` 里记录了 v2 每段原文的 sha256，逐段比对。
- 含行尾空格的段落（如 `summary`）用 YAML 双引号字符串保存，其余都用可读的字面量块。
- 应用层需要自己组装消息时，用 `prompt.template("system")` 取原始模板；能整体渲染的地方用 `prompt.render(**变量)`。
- `DEPARTMENT_MAPPING` 和 `CATEGORY_MAPPING` 不是提示词，移到 `tools/annotation.py`。

---

## 7. 备选方案（当前不采用）

Langfuse、LangSmith 这类平台自带提示词管理界面，能和调用追踪联动。暂不采用，原因有两个：
- 需要额外部署或使用外部服务，医疗数据也可能被发送到外部；
- 文件加 git 已经满足版本管理和评审的需要。

`PromptRegistry` 的加载来源会做成接口（`PromptSource`），以后需要界面时，可以加一个 Langfuse 来源，现有调用方不用改。
