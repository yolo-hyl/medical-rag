# MedicalRag 包异步化改造方案（v2.0.0-anyc）

> 目标：把 `MedicalRag` 包对外暴露的能力（检索、入库、RAG 问答、Agent、标注、评测）改成原生 `async`，让任何上层调用方都能在一个事件循环里高并发地调用它。
> 范围：只改 `src/MedicalRag` 包本身；`api/`（FastAPI 服务）只是最后用来展示的调用方，不属于这次改造的重点，最后只适配调用点。
> 原则：**只改 IO 路径，不引入新框架，不保留同步 / 异步两份实现。**

---

## 1. 包内现状

包里所有对外方法都是同步的，而且全部在 IO 上阻塞：

| 模块 | 同步阻塞点 |
| --- | --- |
| `core/KnowledgeBase.py` | `MilvusClient.search / hybrid_search / insert / upsert`；`embed_query / embed_documents` 调远程 Embedding API |
| `core/insert.py` | 分批 `client.insert / upsert / delete` |
| `core/HybridRetriever.py` | 只实现了 `_get_relevant_documents`，`ainvoke` 时会被 LangChain 丢进线程池 |
| `core/IngestionPipeline.py` | 串行建表 → 入库 → 建索引 |
| `rag/SimpleRag.py`、`rag/MultiDialogueRag.py` | `rag_chain.invoke`、`llm.invoke`（改写、生成、摘要） |
| `rag/RagEvaluate.py` | 循环里逐条 `rag.answer` |
| `agent/SearchGraph.py`、`agent/MedicalAgent.py` | 每个图节点都是同步 `llm.invoke` / `tool_node.invoke`，图用 `app.invoke` |
| `agent/tools/*` | `database_search` 同步查 Milvus；腾讯云搜索 SDK 只有同步版本 |
| `data/annotation.py` | 逐条 `llm.invoke`，整个数据集串行标注 |

另外还有几个问题，异步化之后会暴露得更明显，这次一起处理：

- **检索参数串台**：`MedicalHybridRetriever` 通过 `self.search_config.query = ...` 修改共享对象，并发时 A 请求可能用上 B 请求的 query。
- **会话状态无保护**：`MultiDialogueRag` 的 `_histories` / `_running_summaries` / `_token_meta_store`，以及 `MedicalAgent.state`，同一会话并发调用时会被交错写乱。
- **知识库重复创建**：`SimpleRAG`、`MultiDialogueRag`、`IngestionPipeline`、`DBFactory` 各自 new 一个 `MedicalHybridKnowledgeBase`，同一进程里有多份 Milvus 连接、Embedding 客户端和 BM25 词表。
- **小 bug**：`_encode_query` 在 Milvus 托管 BM25 时执行 `data = data`，会报变量未定义；`add_documents` 在循环里逐条调用 Embedding API。

---

## 2. 设计要点

1. **IO 密集 → 原生异步**
   - LLM：`invoke` → `ainvoke`（`ChatOpenAI` / `ChatOllama` 原生支持）
   - Embedding：`embed_*` → `aembed_*`
   - Milvus 读写：`AsyncMilvusClient`（现有的 pymilvus 2.5.14 已自带）
   - LangChain 链、LangGraph 图、ToolNode：`invoke` → `ainvoke`
2. **CPU 密集或只有同步 SDK → `asyncio.to_thread`**：BM25 分词（pkuseg）、腾讯云搜索、RAGAS 评测。
3. **直接把原方法改成 `async def`，名字不变**，不保留同步版本。脚本和服务统一用 `await` / `asyncio.run`。
4. **知识库可注入**：各组件的构造函数增加可选参数 `kb`，传了就复用，不传就自己创建（向后兼容），调用方可以全局只用一个实例。删掉 `DBFactory`。
5. **包内保证并发安全**：检索改为请求级拷贝；有会话状态的组件内置按会话的 `asyncio.Lock`。不同会话完全并行，只串行化同一会话。

---

## 3. 改造后的对外 API

| 组件 | 改造前 | 改造后 |
| --- | --- | --- |
| `MedicalHybridKnowledgeBase` | `search(req)`、`add_documents(docs)`、`build_index()`、`_create_collection()` | 同名方法全部 `async`；新增 `async close()` |
| `IngestionPipeline(config)` | `run(records) -> bool` | `IngestionPipeline(config, kb=None)`；`async run(records)` |
| `SimpleRAG(config, search_config=None)` | `answer(query, return_document)` | 增加 `kb=None` 参数；`async answer(...)`，`batch_answer` 改为 `asyncio.gather` 并发 |
| `MultiDialogueRag(config, search_config=None)` | `answer(query, return_document, session_id)` | 增加 `kb=None` 参数；`async answer(...)`（内置会话锁） |
| `SearchGraph(config, power_model)` | `answer(query)`、`run(state)` | 增加 `kb=None` 参数；`async answer / run` |
| `MedicalAgent(config, power_model)` | `answer(user_input)` | 增加 `kb=None` 参数；`async answer(...)`（内置锁） |
| `RagasRagEvaluate` | `do_evaluate(...)` | `async do_evaluate(...)` |
| `SimpleAnnotator` / `AnnotationPipeline` | `annotate_single`、`annotate_dataset`、`run` | 全部 `async`，按 `batch_size` 限制并发 |

返回值结构保持不变，上层只需要把 `x.answer(...)` 改成 `await x.answer(...)`。

---

## 4. 逐模块改动

### 4.1 `core/utils.py`

客户端本身不用改，都原生支持 async。唯一要补的是配置了 `proxy` 时，异步请求走的是 `http_async_client`：

```python
if config.proxy:
    kwargs["http_client"] = httpx.Client(proxy=config.proxy)            # 顺带修掉现在传 dict 的写法
    kwargs["http_async_client"] = httpx.AsyncClient(proxy=config.proxy)
```

### 4.2 `core/KnowledgeBase.py`（核心）

**客户端**：保留同步 `MilvusClient` 只用于建表、建索引等低频 DDL（2.5.14 的 `AsyncMilvusClient` 没有 `has_collection`）；检索和写入走 `AsyncMilvusClient`。
`AsyncMilvusClient` 底层是 `grpc.aio`，必须在事件循环内创建，所以用**懒加载**，调用方不需要关心创建时机：

```python
@property
def aclient(self) -> AsyncMilvusClient:
    if self._aclient is None:          # 只会在协程里第一次访问，此时一定在 loop 内
        self._aclient = AsyncMilvusClient(uri=..., token=...)
    return self._aclient
```

| 方法 | 改法 |
| --- | --- |
| `_encode_query` | `async`：dense 用 `await emb.aembed_query(q)`；自管理 BM25 用 `await asyncio.to_thread(bm25.embed_query, q)`；Milvus 托管 BM25 直接返回原始 query（修掉 `data = data`） |
| `_search` / `_hybrid_search` | `async`：`await self.aclient.search / hybrid_search(...)`；多路子请求用 `asyncio.gather` **并发编码** |
| `search` | `async def search(req)`，Document 封装逻辑不变 |
| `add_documents` | `async`：整批 `aembed_documents(summaries)`、`aembed_documents(texts)` 并发执行；稀疏编码走 `to_thread`；最后 `await insert_rows(self.aclient, ...)` |
| `_create_collection` / `build_index` | `async` 外壳，内部 `await asyncio.to_thread(...)` 调同步 client |
| 新增 `close` | `await self._aclient.close()` + `self.client.close()` |

### 4.3 `core/insert.py`

`insert_rows` / `insert_rows_has_id` / `delete_by_ids` 改 `async`，内部 `await client.insert / upsert / delete(...)`，入参换成 `AsyncMilvusClient`。进度条逻辑不变。

### 4.4 `core/HybridRetriever.py`

- 实现 `async _aget_relevant_documents`，删掉同步版。
- 修复串台，不再修改共享配置：
  ```python
  req = self.search_config.model_copy(update={"query": inputs.get("input", "")})
  documents = await self.knowledge_base.search(req)
  ```

### 4.5 `core/IngestionPipeline.py`

`__init__(config, kb=None)`；`async run`：`await kb._create_collection()` → 分批 `await kb.add_documents(batch)` → `await kb.build_index()`。批次之间保持串行（瓶颈在 Embedding API 限流，单批内部已经是批量请求）。

### 4.6 `core/DBFactory.py`：删除

`AgentTools` 改为构造时接收 `kb`，不再走进程级 `lru_cache` 单例。

### 4.7 `rag/RagBase.py`

- 抽象方法 `answer` 声明为 `async`。
- `batch_answer` 改为 `await asyncio.gather(*(self.answer(q, ...) for q in queries))`。
- 构造函数接收 `kb=None`，统一在基类里执行 `self.knowledge_base = kb or MedicalHybridKnowledgeBase(config)`，两个子类不再各自创建。

### 4.8 `rag/SimpleRag.py`

`answer` → `async`：`result = await self.rag_chain.ainvoke({"input": query})`。
链本身不用改：`ainvoke` 下 `RunnablePassthrough.assign(milvus_result=retriever)` 会自动走 `retriever.ainvoke`；`format_document_str`、`strip_think_and_time` 是纯 CPU 小函数，保持同步。

### 4.9 `rag/MultiDialogueRag.py`

| 位置 | 改法 |
| --- | --- |
| `_timed_llm_invoke` | `async def`，`await self.llm.ainvoke(...)` |
| `do_retrieve` | `async def`，`await self.self_retriever.ainvoke(...)` |
| `_get_summary` | `async`，`await (summarize_prompt \| self.llm).ainvoke(...)` |
| `_maybe_compress_history` | `async` |
| `answer` | `async`，整个流程包在 `async with self._session_locks[session_id]` 里 |
| `_build_document_context` / `_update_tokens_metadata` / `_avg_estimate_over_max_token` | 纯 CPU，保持同步 |

会话锁：

```python
self._session_locks: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
```

同时对外提供 `session_lock(session_id)`，方便上层做流式输出（`rag_chain.astream_events`）时也能拿同一把锁。

### 4.10 `agent/SearchGraph.py`

- 节点 `llm_db_search` / `llm_network_search` / `rag` / `judge` 全部改 `async def`，`llm.invoke`、`tool_node.invoke`、`judge_chain.invoke`、`search_chain.invoke` 统一改为 `await ...ainvoke`（`OutputFixingParser` 支持 async）。
- `finish_success` / `finish_fail` / `judge_router` 是纯状态操作，保持同步。
- `answer` / `run` → `async`，`await self.search_graph.ainvoke(...)`。
- LangGraph 能识别 `partial(async_fn, ...)` 为异步节点，图定义不用改。

### 4.11 `agent/tools/`

- `AgentTools.__init__(app_config, kb)`。
- `database_search` → `async def`，`await self.kb.search(search_config)`。
- `web_search` → `async def`，`await asyncio.to_thread(self.WEBSEARCH_FUNC, query, cnt)`。`TencentSearch.py` 本身不动。
- `calculator` 是纯 CPU，不改。
- 顺手修掉 `raise "未注册网络检索工具"`（raise 字符串会触发 TypeError），改为 `raise RuntimeError(...)`。

### 4.12 `agent/MedicalAgent.py`

- 节点 `ask_judge` / `extract_background_info` / `check_update_background` / `judge_split_query` / `search_one` / `gather_answer` 改 `async`，`invoke` → `ainvoke`；`search_one` 里 `await search_graph.run(...)`。
  收益：`Send` 拆出来的多个子查询由线程并行变为协程并发。
- 路由函数 `route_entry` / `route_ask_again` / `route_to_subgraphs` 保持同步。
- `MedicalAgent` 实例本身就是一个会话，增加 `self.lock = asyncio.Lock()`，`answer` 内部 `async with self.lock`。
- `answer` → `async`：`self.state = await self.app.ainvoke(self.state)`。

### 4.13 `rag/RagEvaluate.py`

`do_evaluate` → `async`：
1. 先用 `asyncio.gather` + `Semaphore` 并发获取所有样本的答案（`await self.rag.answer(...)`）；
2. `ragas.evaluate(...)` 内部自带事件循环（`nest_asyncio`），不能在当前 loop 里直接跑，用 `await asyncio.to_thread(evaluate, ...)`。

### 4.14 `data/annotation.py`

- `annotate_single` → `async`：`await self.llm.ainvoke(messages)`，重试逻辑不变。
- `annotate_dataset` → `async`：用 `asyncio.Semaphore(self.batch_size)` 限制并发，`asyncio.gather` 处理整个数据集，替代现在的串行循环。临时文件、最终文件的写入逻辑不变。
- `AnnotationPipeline.run` / `run_annotation` → `async`。

---

## 5. 不改的部分

| 模块 | 原因 |
| --- | --- |
| `config/`（`loader.py`、`models.py`） | 只在启动时读写一次 YAML，没必要异步化 |
| `embed/sparse.py`、`embed/bm25.py` | CPU 密集（pkuseg 分词、多进程建词表），由调用方用 `to_thread` 包装 |
| `prompts/templates.py`、`rag/utils.py`、`agent/utils.py` | 纯字符串 / 纯计算 |
| `agent/tools/TencentSearch.py` | 只有同步 SDK，由 `AgentTools` 用 `to_thread` 包装 |

---

## 6. 依赖变更

不新增依赖：`AsyncMilvusClient`、`ainvoke`、`httpx.AsyncClient` 都是现有依赖已有的能力。

---

## 7. 实施顺序

1. **core**：`KnowledgeBase` + `insert` + `HybridRetriever` + `IngestionPipeline`，删除 `DBFactory`；用 `scripts/02`、`03` 验证入库、检索与改造前一致。
2. **rag**：`RagBase` + `SimpleRag` + `MultiDialogueRag` + `RagEvaluate`；用 `scripts/04`、`05`、`06` 验证。
3. **agent**：`AgentTools` + `SearchGraph` + `MedicalAgent`；用 `scripts/07`、`08` 验证。
4. **data**：`annotation.py`。
5. **scripts**：入口改为 `asyncio.run(main())`（随前面各步一起改）。
6. **展示层（最后）**：`api/app.py` 只做调用点适配：把 `await run_sync(x.answer, ...)` 改成 `await x.answer(...)`，在 lifespan 里创建一个共享 `kb` 并注入各组件，流式接口使用包提供的会话锁。服务自身的其他逻辑（鉴权、SQLite 持久化等）不在本次改造范围内。

## 8. 验证方式

- **功能一致**：每一步改造前后对同一批问题跑脚本，对比检索到的文档 `pk` 与回答。
- **并发收益**：写一个小脚本，用 `asyncio.gather` 同时发起 N 个 `SimpleRAG.answer`，对比改造前（线程池）和改造后的总耗时；再用服务做一次压测作为最终展示。
- **不阻塞事件循环**：开启 `loop.set_debug(True)` 并设置 `slow_callback_duration = 0.1`，压测期间日志中不应出现超过 100ms 的慢回调。
