# Medical RAG API 文档

## 快速开始

```bash
# 设置 API Key（根据配置文件中的 provider 选择）
export DASHSCOPE_API_KEY=sk-...   # DashScope / Qwen
# 或
export OPENAI_API_KEY=sk-...       # OpenAI

# 启动服务
cd /path/to/medical-rag
python run_api.py

# 交互式文档（Swagger UI）
open http://localhost:8000/docs
```

默认监听 `0.0.0.0:8000`，可通过环境变量覆盖：

```bash
API_HOST=127.0.0.1 API_PORT=9000 python run_api.py
```

---

## 通用说明

### 错误响应格式

```json
{
  "detail": "错误描述",
  "type": "ExceptionClassName"
}
```

HTTP 状态码 `500` 表示服务端未预期错误。

### 注意事项

- 服务必须以 `workers=1` 启动（默认配置），因为会话状态存储在进程内存中
- `session_id` 区分不同用户/会话的历史上下文，建议每个用户分配唯一 ID
- 数据录入（`/api/ingest`）和评测（`/api/eval`）耗时较长，请勿设置过短的客户端超时

---

## SSE 流式响应说明

`/api/chat/stream` 和 `/api/agent/stream` 返回 `text/event-stream`（Server-Sent Events）。

### 事件格式

每行格式为：
```
data: {"type": "<事件类型>", ...字段}\n\n
```

### 事件类型速查表

| type | 含义 | 额外字段 |
|------|------|---------|
| `progress` | 阶段提示文字 | `message: str` |
| `documents` | 检索到的文档（生成前即推送） | `data: List[{content, source, source_name}]` |
| `clarification` | Agent 需要追问用户（仅 `/agent/stream`） | `questions: List[str]` |
| `sub_queries` | Agent 拆分的子查询（仅 `/agent/stream`） | `queries: List[str]` |
| `answer` | 最终答案 | `data: str` |
| `done` | 流结束，客户端可关闭连接 | — |
| `error` | 发生错误 | `message: str` |

### 前端接入示例（JavaScript）

```javascript
const response = await fetch('/api/chat/stream', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ question: '高血压怎么办？', session_id: 'user_1' }),
});

const reader = response.body.getReader();
const decoder = new TextDecoder();

while (true) {
  const { done, value } = await reader.read();
  if (done) break;

  const lines = decoder.decode(value).split('\n');
  for (const line of lines) {
    if (!line.startsWith('data: ')) continue;
    const event = JSON.parse(line.slice(6));

    if (event.type === 'progress') {
      showStatus(event.message);               // 显示"正在检索..."
    } else if (event.type === 'documents') {
      showDocuments(event.data);               // 提前展示检索结果
    } else if (event.type === 'answer') {
      showAnswer(event.data);                  // 显示最终答案
    } else if (event.type === 'done') {
      break;
    }
  }
}
```

---

## 端点详情

---

### 1. `GET /api/health` — 健康检查

检查服务是否正常启动，各 RAG 组件是否已加载。

**响应示例**

```json
{
  "status": "ok",
  "services": {
    "simple_rag": true,
    "multi_rag": true,
    "search_agent": true,
    "agent_sessions": true
  }
}
```

---

### 2. `POST /api/ingest` — 录入数据

将医疗问答记录批量写入 Milvus 向量数据库（自动生成多向量嵌入）。

**请求体**

```json
{
  "records": [
    {
      "question": "高血压有哪些症状？",
      "answer": "常见症状包括头痛、眩晕、心悸等。",
      "source": "qa",
      "source_name": "huatuo_qa"
    }
  ],
  "drop_old": false
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `records` | `List[Dict]` | 是 | 原始数据列表，字段需与配置中 `data` 字段映射一致 |
| `drop_old` | `bool` | 否（默认 `false`） | 是否先清空现有集合（谨慎使用） |

**响应示例**

```json
{
  "success": true,
  "records_count": 1,
  "message": "数据录入成功"
}
```

---

### 3. `POST /api/search` — 检索文档

直接在向量知识库中检索，返回原始文档列表（不经过 LLM 生成）。

**请求体**

```json
{
  "query": "高血压用药推荐",
  "limit": 5,
  "filter_expr": "source == \"qa\"",
  "use_hybrid": true
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `query` | `str` | 是 | 检索查询文本 |
| `limit` | `int` | 否（默认 5） | 返回文档数量，范围 1–50 |
| `filter_expr` | `str` | 否 | Milvus 过滤表达式，如 `source == "qa"` 或 `source == "literature"` |
| `use_hybrid` | `bool` | 否（默认 `true`） | `true`=稠密+稀疏混合检索，`false`=仅稠密向量检索 |

**响应示例**

```json
{
  "query": "高血压用药推荐",
  "documents": [
    {
      "source": "qa",
      "source_name": "huatuo_qa",
      "content_preview": "高血压患者可根据病情选择钙拮抗剂、ACEI..."
    }
  ],
  "search_time": 0.123
}
```

---

### 4. `POST /api/ask` — 单轮问答

无状态的单轮医疗问答（SimpleRAG），每次请求独立，不保留历史。

**请求体**

```json
{
  "question": "感冒了应该吃什么药？"
}
```

**响应示例**

```json
{
  "question": "感冒了应该吃什么药？",
  "answer": "感冒可分为病毒性和细菌性...",
  "sources": [
    {
      "source": "qa",
      "source_name": "huatuo_qa",
      "content_preview": "感冒通常由病毒引起..."
    }
  ],
  "search_time": 0.45,
  "generation_time": 3.21
}
```

---

### 5. `POST /api/chat` — 多轮对话（阻塞）

带会话历史的多轮医疗问答（MultiDialogueRag）。等待全部完成后一次性返回结果。

如需实时进度反馈，使用 [`/api/chat/stream`](#6-post-apichatstream--多轮对话sse流式)。

**请求体**

```json
{
  "question": "我最近血压偏高，有什么注意事项？",
  "session_id": "user_abc_123"
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `question` | `str` | 是 | 用户问题 |
| `session_id` | `str` | 否（默认 `"default"`） | 会话 ID，相同 ID 共享对话历史 |

**响应示例**

```json
{
  "session_id": "user_abc_123",
  "question": "我最近血压偏高，有什么注意事项？",
  "answer": "血压偏高时需注意低盐饮食、规律运动...",
  "sources": [...],
  "search_time": 0.38,
  "rewrite_time": 1.12,
  "generation_time": 5.67
}
```

---

### 6. `POST /api/chat/stream` — 多轮对话（SSE流式）

与 `/api/chat` 功能相同，但以 SSE 流式推送中间状态，显著改善用户等待体验。

**请求体**：同 `/api/chat`

**SSE 事件流示例**

```
data: {"type": "progress", "message": "正在理解您的问题..."}

data: {"type": "progress", "message": "正在检索相关文档..."}

data: {"type": "documents", "data": [{"content": "高血压患者...", "source": "qa", "source_name": "huatuo"}]}

data: {"type": "progress", "message": "正在生成回答..."}

data: {"type": "answer", "data": "血压偏高时需注意低盐饮食、规律运动..."}

data: {"type": "done"}
```

> **关键优化：** `documents` 事件在 LLM 生成开始前即推送，前端可提前展示参考资料，减少用户感知等待时间。

---

### 7. `POST /api/eval` — RAG 评测

使用 RAGAS 框架对 SimpleRAG 进行质量评测，返回四项指标。耗时较长（数分钟），请适当设置客户端超时。

**请求体**

```json
{
  "eval_file": "data/eval/new_qa_200.jsonl",
  "sample_size": 10,
  "query_field": "question",
  "reference_field": "answer"
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `eval_file` | `str` | 是 | 服务器端 JSONL 评测数据集路径（相对于工作目录） |
| `sample_size` | `int` | 否（默认 10） | 采样数量，越大越准确但耗时越长 |
| `query_field` | `str` | 否（默认 `"question"`） | 数据集中问题字段名 |
| `reference_field` | `str` | 否（默认 `"answer"`） | 数据集中参考答案字段名 |

**响应示例**

```json
{
  "sample_size": 10,
  "metrics": {
    "answer_relevancy": 0.87,
    "faithfulness": 0.91,
    "context_recall": 0.78,
    "context_precision": 0.83
  }
}
```

| 指标 | 含义 |
|------|------|
| `answer_relevancy` | 答案与问题的相关性（0–1，越高越好） |
| `faithfulness` | 答案是否忠实于检索到的文档（幻觉检测） |
| `context_recall` | 检索结果对参考答案的覆盖率 |
| `context_precision` | 检索结果的精准度 |

---

### 8. `POST /api/search-agent` — 单轮智能检索 Agent

基于 SearchGraph 的单轮智能问答，支持工具调用（数据库检索、可选联网搜索）、多步推理和事实核验。不保留对话历史。

**请求体**

```json
{
  "question": "糖尿病患者能吃西瓜吗？"
}
```

**响应示例**

```json
{
  "question": "糖尿病患者能吃西瓜吗？",
  "answer": "糖尿病患者可以少量食用西瓜，但需注意..."
}
```

---

### 9. `POST /api/agent` — 多轮智能医疗 Agent（阻塞）

完整的多轮医疗 Agent（MedicalAgent），支持：主动追问补充信息、子查询拆分、并行检索、多轮对话历史。等待全部完成后一次性返回。

如需实时进度反馈，使用 [`/api/agent/stream`](#10-post-apiagentstream--多轮智能医疗agentsse流式)。

**请求体**

```json
{
  "question": "帮我分析一下这个症状",
  "session_id": "patient_session_1"
}
```

**响应示例（需要追问）**

```json
{
  "session_id": "patient_session_1",
  "question": "帮我分析一下这个症状",
  "needs_clarification": true,
  "clarification_questions": [
    "请问您具体是哪个部位不舒服？",
    "症状持续多久了？",
    "是否有发热？"
  ],
  "answer": null
}
```

**响应示例（直接回答）**

```json
{
  "session_id": "patient_session_1",
  "question": "我头疼两天了，没有发热",
  "needs_clarification": false,
  "clarification_questions": [],
  "answer": "根据您描述的症状，持续头疼可能由以下原因..."
}
```

> 当 `needs_clarification: true` 时，请将 `clarification_questions` 展示给用户，收集回复后再次调用此接口（携带相同 `session_id`）。

---

### 10. `POST /api/agent/stream` — 多轮智能医疗 Agent（SSE流式）

与 `/api/agent` 功能相同，但以 SSE 流式推送每个推理步骤，让用户实时了解 Agent 的工作进展。

**请求体**：同 `/api/agent`

**SSE 事件流示例（直接回答场景）**

```
data: {"type": "progress", "message": "正在分析问题，准备检索..."}

data: {"type": "sub_queries", "queries": ["头痛原因", "持续性头痛治疗"]}

data: {"type": "documents", "data": [{"content": "紧张性头痛是最常见的...", "source": "literature"}]}

data: {"type": "documents", "data": [{"content": "头痛的药物治疗方案...", "source": "qa"}]}

data: {"type": "answer", "data": "根据您描述的症状，持续头疼可能由以下原因..."}

data: {"type": "done"}
```

**SSE 事件流示例（需要追问场景）**

```
data: {"type": "clarification", "questions": ["请问您具体是哪个部位不舒服？", "症状持续多久了？"]}

data: {"type": "done"}
```

---

## curl 快速测试

```bash
BASE=http://localhost:8000

# 健康检查
curl $BASE/api/health

# 录入数据
curl -X POST $BASE/api/ingest \
  -H "Content-Type: application/json" \
  -d '{"records": [{"question":"高血压症状","answer":"头痛眩晕","source":"qa","source_name":"test"}]}'

# 检索文档
curl -X POST $BASE/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "高血压用药", "limit": 3}'

# 单轮问答
curl -X POST $BASE/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "感冒了怎么办？"}'

# 多轮对话（阻塞）
curl -X POST $BASE/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "我血压偏高", "session_id": "u1"}'

# 多轮对话（SSE流式）
curl -N -X POST $BASE/api/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "继续上个问题，我还有头晕", "session_id": "u1"}'

# 评测（需要服务器上有评测数据文件）
curl -X POST $BASE/api/eval \
  -H "Content-Type: application/json" \
  -d '{"eval_file": "data/eval/new_qa_200.jsonl", "sample_size": 5}'

# 单轮智能检索 Agent
curl -X POST $BASE/api/search-agent \
  -H "Content-Type: application/json" \
  -d '{"question": "糖尿病饮食注意事项"}'

# 多轮智能 Agent（阻塞）
curl -X POST $BASE/api/agent \
  -H "Content-Type: application/json" \
  -d '{"question": "我有点不舒服", "session_id": "s1"}'

# 多轮智能 Agent（SSE流式）
curl -N -X POST $BASE/api/agent/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "我有点不舒服", "session_id": "s2"}'
```
