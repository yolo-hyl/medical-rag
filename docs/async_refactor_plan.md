# 全链路异步化改造方案（v2.0.0-anyc）

> 目标：把服务端调用链上的同步阻塞调用改成原生 `async`，让单个 uvicorn 进程靠事件循环扛住高并发。
> 原则：**只改 IO 路径，不引入新框架，不做多余抽象**。

---

## 1. 现状与瓶颈

当前 `api/app.py` 的 handler 虽然是 `async def`，但真正干活的全是同步代码，通过 `run_sync()`（`loop.run_in_executor(None, ...)`）扔进默认线程池：

| 问题 | 位置 | 影响 |
| --- | --- | --- |
| 每个请求在 LLM 调用（秒级）期间独占一个线程 | `app.py` 的 `run_sync` | 默认线程池只有 `min(32, cpu+4)` 个线程，**并发上限≈32**，再多就排队 |
| SSE 接口表面上是 `astream_events` / `astream`，节点内部仍是同步 `invoke` | `MultiDialogueRag._setup_chain`、Agent 各节点 | LangChain 会把同步节点丢进线程池，同样受上限约束 |
| handler 里直接调同步 SQLite | `chat` / `chat_stream` / `agent*` 中的 `_auth.upsert_session`、`save_message` | **直接阻塞事件循环** |
| `chat_stream` 里直接调 `_maybe_compress_history`（内部同步 `llm.invoke`） | `app.py:486` | 触发摘要时**阻塞事件循环数秒** |
| 检索器改写共享对象 | `HybridRetriever._get_relevant_documents` 中 `self.search_config.query = ...` | 并发下 query 串台（异步后更容易触发） |
| 同一进程创建了 4+ 份 `MedicalHybridKnowledgeBase` | `SimpleRAG` / `MultiDialogueRag` / `IngestionPipeline` / `DBFactory` | 多份 Milvus 连接、嵌入客户端、BM25 词表，浪费内存 |

---

## 2. 改造原则

1. **IO 密集 → 原生异步**
   - LLM：`invoke` → `ainvoke`（`ChatOpenAI` / `ChatOllama` 原生支持）
   - Embedding：`embed_query/embed_documents` → `aembed_query/aembed_documents`
   - Milvus 读写：`MilvusClient` → `AsyncMilvusClient`（pymilvus 2.5.14 已自带）
   - SQLite：`sqlite3` → `aiosqlite`
   - LangChain 链 / LangGraph 图：`invoke` → `ainvoke`
2. **CPU 密集 / 只有同步 SDK → `asyncio.to_thread`**
   - BM25 稀疏编码（pkuseg 分词）、腾讯云搜索 SDK、RAGAS 评测
3. **不保留同步双份实现**：方法直接改成 `async def`（沿用原方法名），脚本入口统一 `asyncio.run(main())`。
4. **全局只保留一个知识库实例**，在 FastAPI `lifespan` 中创建并注入，去掉 `DBFactory`。

---

## 3. 逐模块改动

### 3.1 `core/utils.py` —— LLM / Embedding 客户端

- 基本不动，`ChatOpenAI`、`OpenAIEmbeddings`、`ChatOllama`、`OllamaEmbeddings` 都原生支持 async。
- 唯一要改：配置了 `proxy` 时，异步请求走的是 `http_async_client`，需要补上：
  ```python
  kwargs["http_async_client"] = httpx.AsyncClient(proxy=config.proxy)
  ```
  （顺带修掉现在把 dict 传给 `http_client` 的写法，改为 `httpx.Client(proxy=...)`。）

### 3.2 `core/KnowledgeBase.py` —— 知识库（改动核心）

```python
class MedicalHybridKnowledgeBase:
    def __init__(self, app_config):
        ...
        self.client = MilvusClient(...)          # 仅用于建表/建索引等低频 DDL
        self.aclient = AsyncMilvusClient(...)    # 检索与写入
```

| 方法 | 改法 |
| --- | --- |
| `_encode_query` | `async`：dense 用 `await emb.aembed_query(q)`；自管理 BM25 用 `await asyncio.to_thread(bm25.embed_query, q)`；顺手修掉 `data = data` 的未定义变量 bug（Milvus 托管 BM25 时应直接返回原始 query） |
| `_search` / `_hybrid_search` | `async`：`await self.aclient.search(...)` / `await self.aclient.hybrid_search(...)` |
| `_hybrid_search` 的子请求编码 | 多路向量用 `asyncio.gather` **并发编码**，而不是串行 |
| `search` | `async def search(req)` |
| `add_documents` | `async`：改为**整批** `aembed_documents(summaries)` / `aembed_documents(texts)`（现在是循环里一条一条调 embedding API），稀疏编码走 `to_thread`，最后 `await insert_rows(...)` |
| `_create_collection` / `build_index` | `async` 外壳 + `await asyncio.to_thread(...)` 调同步 client。原因：pymilvus 2.5.14 的 `AsyncMilvusClient` 没有 `has_collection`，且建表只在入库时低频调用，没必要强行异步 |

⚠️ 注意：`AsyncMilvusClient` 底层是 `grpc.aio` channel，**必须在事件循环内创建**。所以知识库只能在 `lifespan`（或脚本的 `async main`）里实例化，不能在模块顶层创建。

新增 `async def close()`：关闭 `aclient` 和 `client`，在 `lifespan` 退出时调用（替代 `DBFactory` 里的 `atexit`）。

### 3.3 `core/insert.py`

`insert_rows` / `insert_rows_has_id` / `delete_by_ids` 改为 `async`，内部 `await client.insert/upsert/delete(...)`。tqdm 进度逻辑不变。

### 3.4 `core/HybridRetriever.py`

- 实现 `_aget_relevant_documents`（`BaseRetriever.ainvoke` 会走它），同步版 `_get_relevant_documents` 直接删掉或 `raise NotImplementedError`。
- **修复并发串台**：不再修改共享的 `self.search_config`，而是
  ```python
  req = self.search_config.model_copy(update={"query": inputs.get("input", "")})
  documents = await self.knowledge_base.search(req)
  ```

### 3.5 `core/IngestionPipeline.py`

`run` → `async def run`：`await kb._create_collection()` → 分批 `await kb.add_documents(batch)` → `await kb.build_index()`。
批与批之间保持串行即可（入库是后台管理操作，瓶颈在 embedding API 限流，不需要再加并发控制）。
构造函数改为接收外部传入的 `kb`，不再自己 new 一个。

### 3.6 `core/DBFactory.py` —— 删除

它的作用是"按进程缓存 KB 单例"。改为在 `lifespan` 中创建唯一 KB，并通过构造参数注入到 `SimpleRAG`、`MultiDialogueRag`、`SearchGraph/AgentTools`、`IngestionPipeline`。

### 3.7 `rag/SimpleRag.py`

- `__init__(config, kb, search_config=None)`：使用注入的 `kb`。
- `answer` → `async def answer`：`result = await self.rag_chain.ainvoke({"input": query})`。
- 链本身不用改：`RunnablePassthrough.assign(milvus_result=retriever)` 在 `ainvoke` 下会自动调用 `retriever.ainvoke`；`format_document_str`、`strip_think_and_time` 是纯 CPU 小函数，保持同步。

### 3.8 `rag/MultiDialogueRag.py`

| 位置 | 改法 |
| --- | --- |
| `_timed_llm_invoke` | 改成 `async def`，`await self.llm.ainvoke(...)`，用 `RunnableLambda(afunc)` 包装 |
| `do_retrieve` | `async def`：`await self.self_retriever.ainvoke(...)` |
| `_get_summary` | `async`：`await (summarize_prompt \| self.llm).ainvoke(...)` |
| `_maybe_compress_history` | `async`（因为调了 `_get_summary`） |
| `answer` | `async`：`await self._maybe_compress_history(...)`，`await self.rag_chain.ainvoke(...)` |
| `_build_document_context` / `_update_tokens_metadata` | 纯 CPU，保持同步 |

**会话级并发保护**：历史存在进程内 dict 中，同一个 `session_id` 的两个并发请求会交错读写历史/摘要。增加一个最简单的按会话锁：

```python
self._session_locks: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

async def answer(self, query, return_document=False, session_id="default"):
    async with self._session_locks[session_id]:
        ...
```

`chat_stream` 同样在生成器里 `async with multi_rag._session_locks[session_id]` 包住整段流程。不同会话之间完全并行，只串行化同一会话。

### 3.9 `agent/SearchGraph.py`

- 节点函数 `llm_db_search` / `llm_network_search` / `rag` / `judge` 全部改 `async def`：
  - `llm.invoke` → `await llm.ainvoke`
  - `db_tool_node.invoke` / `network_tool_node.invoke` → `await ...ainvoke`
  - `judge_chain.invoke` / `search_chain.invoke` → `await ...ainvoke`（`OutputFixingParser` 支持 async）
- `answer` / `run` → `async`，内部 `await self.search_graph.ainvoke(...)`。
- LangGraph 会自动识别 `partial(async_fn, ...)` 为异步节点，图定义部分不用改。

### 3.10 `agent/tools/AgentTools.py` & `TencentSearch.py`

- `database_search`：改为 `async def`，`await self.kb.search(search_config)`（`kb` 由构造函数注入）。`@tool` 装饰 async 函数即得到协程工具，`ToolNode.ainvoke` 直接支持。
- `web_search`：腾讯云 SDK 只有同步版本，工具改为 `async def`，内部 `await asyncio.to_thread(self.WEBSEARCH_FUNC, query, cnt)`。`TencentSearch.py` 本身不动。

### 3.11 `agent/MedicalAgent.py`

- 节点 `ask_judge` / `extract_background_info` / `check_update_background` / `judge_split_query` / `search_one` / `gather_answer` 全部改 `async`，`invoke` → `ainvoke`。
- `search_one` 中 `search_graph.run(...)` → `await search_graph.run(...)`。
  **收益**：`Send` 拆出的多个子查询原本在线程池里并行，现在在事件循环里以协程并发，不再占线程。
- `answer` → `async`：`self.state = await self.app.ainvoke(self.state)`。
- **会话级并发保护**：`MedicalAgent` 本身就是按 session 缓存的（`self.state` 是会话状态），给实例加一个 `self.lock = asyncio.Lock()`，`/api/agent` 与 `/api/agent/stream` 进入时 `async with agent.lock`。

### 3.12 `api/auth.py` —— SQLite 异步化

- 依赖新增 `aiosqlite`。
- 所有函数改 `async def`，`sqlite3.connect` → `aiosqlite.connect`，SQL 语句不变：
  ```python
  async def verify_token(token):
      async with _conn() as conn:
          cur = await conn.execute("SELECT ...", (token,))
          row = await cur.fetchone()
      ...
  ```
- `_conn()` 改为返回 `aiosqlite.connect(DB_PATH)` 并设置 `row_factory`、`PRAGMA foreign_keys`；写操作后 `await conn.commit()`。
- 顺带开启 `PRAGMA journal_mode=WAL`，缓解并发写时的 `database is locked`。

### 3.13 `api/app.py`

- 删除 `run_sync`、`threading`、`_agent_lock`。
- `lifespan`：
  ```python
  await _auth.init_db()
  kb = MedicalHybridKnowledgeBase(config)          # 必须在 loop 内创建
  state["kb"] = kb
  state["simple_rag"] = SimpleRAG(config, kb)
  state["multi_rag"] = MultiDialogueRag(config, kb)
  state["search_agent"] = SearchGraph(config, kb, power_model=...)
  yield
  await kb.close()
  ```
- `get_or_create_agent`：去掉线程锁（单线程事件循环里"检查 + 插入"之间没有 `await`，天然原子）。
- 各接口统一改为直接 `await`：
  - `/api/ask`：`await state["simple_rag"].answer(...)`
  - `/api/chat`：`await state["multi_rag"].answer(...)`
  - `/api/search`：`await kb.search(search_req)`
  - `/api/ingest`：`await IngestionPipeline(cfg, kb).run(records)`（`drop_old` 通过参数传入，不再 deepcopy 整份 config 重建 KB）
  - `/api/search-agent`：`await agent.search_graph.ainvoke(init_state)`
  - `/api/agent`：`async with agent.lock: await agent.answer(...)`
  - 鉴权与历史接口：`await _auth.xxx(...)`；`get_optional_user` 改 `async`
  - `chat_stream`：`await multi_rag._maybe_compress_history(...)`，持久化改 `await _auth.save_message(...)`
- `/api/eval`：见下文"不改的部分"。

### 3.14 `scripts/*.py`

入口统一改为：
```python
async def main():
    kb = MedicalHybridKnowledgeBase(config)
    rag = SimpleRAG(config, kb)
    result = await rag.answer(query, return_document=True)
    ...
    await kb.close()

if __name__ == "__main__":
    asyncio.run(main())
```
涉及 `02`~`08`；`01_build_vocab.py` 纯 CPU，不改。

---

## 4. 不改 / 仅包一层线程的部分

| 模块 | 处理 | 原因 |
| --- | --- | --- |
| `embed/sparse.py`、`embed/bm25.py` | 不改，调用处用 `asyncio.to_thread` | CPU 密集（pkuseg 分词 + 多进程建词表），协程化没有意义 |
| `rag/utils.py`（tiktoken 估算） | 不改 | 纯 CPU 且很快 |
| `data/annotation.py` | 本期不改 | 离线标注脚本，不在服务链路上；如需提速可后续单独加 `asyncio.gather + Semaphore` |
| `rag/RagEvaluate.py` / `/api/eval` | 取答案部分改为 `await rag.answer(...)`；`ragas.evaluate(...)` 本身用 `asyncio.to_thread` 执行 | RAGAS 内部自带事件循环（`nest_asyncio`），不能在主 loop 里直接跑 |
| `run_api.py` 的 `workers=1` | 不改 | 会话状态在进程内存里；本方案靠单进程异步提升并发，多 worker 需要把会话外置（Redis 等），不在本期范围 |

---

## 5. 依赖变更

- 新增：`aiosqlite`
- 其余均为现有依赖的已有能力：`pymilvus==2.5.14`（`AsyncMilvusClient`）、`langchain-core` / `langgraph`（`ainvoke`）、`httpx`（`AsyncClient`）。

---

## 6. 实施顺序（建议按提交拆分）

1. **底层**：`KnowledgeBase` + `insert` + `HybridRetriever` + 删除 `DBFactory`；`scripts/03_search_data.py` 验证检索结果与改造前一致。
2. **RAG**：`SimpleRag` + `MultiDialogueRag`；`scripts/04`、`06` 验证。
3. **Agent**：`SearchGraph` + `AgentTools` + `MedicalAgent`；`scripts/07`、`08` 验证。
4. **API**：`auth.py` → `aiosqlite`，`app.py` 去掉 `run_sync`；逐个接口冒烟（含两个 SSE 接口）。
5. **入库**：`IngestionPipeline` + `scripts/02`。

## 7. 验证方式

- **功能**：每个脚本改造前后对同一批问题跑一遍，对比检索到的文档 `pk` 与回答是否一致。
- **并发**：用 `hey` / `locust` 对 `/api/ask` 发 100 并发，对比改造前后吞吐与 P99；预期改造前在 ~32 并发处出现排队拐点，改造后仅受 LLM 服务端限流约束。
- **不阻塞事件循环**：开启 `PYTHONASYNCIODEBUG=1` 或 `loop.slow_callback_duration = 0.1`，压测期间日志中不应出现 >100ms 的慢回调。
