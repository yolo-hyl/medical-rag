# MedicalRag 包异步化改造方案（v2.0.0-anyc）

> 目标：把 `MedicalRag` 包对外暴露的能力（检索、入库、RAG 问答、Agent、标注、评测）改成原生 `async`，让任何上层调用方都能在一个事件循环里高并发地调用它。
> 范围：只改 `src/MedicalRag` 包本身。`software/`（`run_api.py`、前端）和包内的 `MedicalRag/api/`（FastAPI 服务）只是最后用来展示的调用方，不属于这次改造的重点，最后只适配调用点。
> 基线：`v2.0.0` @ `653819e`（已包含 `deploy/` 中间件）。`core/services.py` 没有必要保留，已在本分支删除，同时清理了 `scripts/00_check_services.py`、`tests/test_services.py` 中依赖它的用例，以及 `software/run_api.py` 里的 `preflight()`。
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

- **会话状态串台**：`MultiDialogueRag` 的 `_histories` / `_running_summaries` / `_token_meta_store`，以及 `MedicalAgent.state`，都放在进程内存里：
  - 同一会话的并发请求会交错读写，把历史和摘要写乱；
  - `MultiDialogueRag.avg_tokens_per_char` 是实例级字段，所有会话共用一个值，会话之间会互相影响；
  - `MedicalAgent` 只能一个会话对应一个实例，上层要自己维护 `session_id → MedicalAgent` 字典，而且内存只增不减；
  - 状态绑定在单个进程上，所以服务只能开 `workers=1`。
- **检索参数串台**：`MedicalHybridRetriever` 通过 `self.search_config.query = ...` 修改共享对象，并发时 A 请求可能用上 B 请求的 query。这是进程内多个协程共享同一个对象造成的，和会话无关，改成请求级拷贝就能解决，不需要 Redis。
- **知识库重复创建**：`SimpleRAG`、`MultiDialogueRag`、`IngestionPipeline`、`DBFactory` 各自创建一个 `MedicalHybridKnowledgeBase`，同一进程里有多份 Milvus 连接、Embedding 客户端和 BM25 词表。
- **小 bug**：`_encode_query` 在 Milvus 托管 BM25 时执行 `data = data`，会报变量未定义；`add_documents` 在循环里逐条调用 Embedding API。

---

## 2. 设计要点

1. **IO 密集 → 原生异步**
   - LLM：`invoke` → `ainvoke`（`ChatOpenAI` / `ChatOllama` 原生支持）
   - Embedding：`embed_*` → `aembed_*`
   - Milvus 读写：`AsyncMilvusClient`（现有的 pymilvus 2.5.14 已自带）
   - Redis：`redis.asyncio`（现有依赖 `redis>=5.0` 自带）
   - LangChain 链、LangGraph 图、ToolNode：`invoke` → `ainvoke`
2. **CPU 密集或只有同步 SDK → `asyncio.to_thread`**：BM25 分词（pkuseg）、腾讯云搜索、RAGAS 评测。
3. **直接把原方法改成 `async def`，名字不变**，不保留同步版本。脚本和服务统一用 `await` / `asyncio.run`。
4. **知识库可注入**：各组件的构造函数增加可选参数 `kb`，传了就复用，不传就自己创建（向后兼容），调用方可以全局只用一个实例。删掉 `DBFactory`。
5. **会话状态放进 Redis**（使用 `deploy/` 下已有的 Redis）：多轮对话历史、摘要、token 统计、Agent 跨轮状态全部存到 Redis，组件本身变成无状态。同一会话用 Redis 分布式锁串行执行，不同会话完全并行，多个进程、多个 worker 之间也能共享会话。
6. **检索参数请求级拷贝**，不再修改共享对象。

---

## 3. 改造后的对外 API

| 组件 | 改造前 | 改造后 |
| --- | --- | --- |
| `RedisSessionStore`（新增） | — | 会话存储，见 4.2 |
| `MedicalHybridKnowledgeBase` | `search(req)`、`add_documents(docs)`、`build_index()`、`_create_collection()` | 同名方法全部 `async`；新增 `async close()` |
| `IngestionPipeline(config)` | `run(records) -> bool` | `IngestionPipeline(config, kb=None)`；`async run(records)` |
| `SimpleRAG(config, search_config=None)` | `answer(query, return_document)` | 增加 `kb=None` 参数；`async answer(...)`；`batch_answer` 改为 `asyncio.gather` 并发 |
| `MultiDialogueRag(config, search_config=None)` | `answer(query, return_document, session_id)` | 增加 `kb=None`、`store=None` 参数；`async answer(...)`，会话数据存 Redis |
| `SearchGraph(config, power_model)` | `answer(query)`、`run(state)` | 增加 `kb=None` 参数；`async answer / run` |
| `MedicalAgent(config, power_model)` | `answer(user_input)`，一个实例对应一个会话 | 增加 `kb=None`、`store=None` 参数；`async answer(user_input, session_id)`，**一个实例服务所有会话** |
| `RagasRagEvaluate` | `do_evaluate(...)` | `async do_evaluate(...)` |
| `SimpleAnnotator` / `AnnotationPipeline` | `annotate_single`、`annotate_dataset`、`run` | 全部 `async`，按 `batch_size` 限制并发 |

`store=None` 时按 `config.redis` 自动创建。返回值结构保持不变，上层只需要把 `x.answer(...)` 改成 `await x.answer(...)`；`MedicalAgent` 另外需要传 `session_id`。

---

## 4. 逐模块改动

### 4.1 `core/utils.py`

客户端本身不用改，都原生支持 async。唯一要补的是：配置了 `proxy` 时，异步请求走的是 `http_async_client`。

```python
if config.proxy:
    kwargs["http_client"] = httpx.Client(proxy=config.proxy)            # 顺带修掉现在传 dict 的写法
    kwargs["http_async_client"] = httpx.AsyncClient(proxy=config.proxy)
```

### 4.2 新增 `core/session_store.py`：Redis 会话存储

使用 `redis.asyncio`，连接信息直接复用 `config.redis.url()`（密码来自 `deploy/.env`）。整个文件只有两个类，预计 80 行左右。

```python
class RedisSessionStore:
    def __init__(self, redis_config: RedisConfig):
        self.r = redis.asyncio.Redis.from_url(redis_config.url(), decode_responses=True)
        self.ttl = redis_config.session_ttl

    def history(self, key) -> "RedisChatHistory": ...   # 对话历史
    async def get_json(self, key, default=None): ...     # 其他会话数据
    async def set_json(self, key, value): ...            # 写入时顺带刷新 TTL
    def lock(self, session_id):                          # 同一会话串行化
        return self.r.lock(f"medrag:lock:{session_id}", timeout=120, blocking_timeout=120)
    async def close(self): ...
```

- **对话历史**：`RedisChatHistory` 继承 LangChain 的 `BaseChatMessageHistory`，底层是 Redis List，每个元素是一条消息的 JSON（用 LangChain 自带的 `messages_to_dict` / `messages_from_dict` 序列化）。
  - 实现 `aget_messages` / `aadd_messages` / `aclear`，所以 `RunnableWithMessageHistory` 在 `ainvoke` / `astream_events` 下可以直接用，链的结构不用改；
  - 另加一个 `atrim(start)`（`LTRIM`），给摘要压缩时丢弃旧消息用。
- **key 规划**（统一前缀 `medrag:`，除锁以外都带 TTL）：

  | key | 类型 | 内容 |
  | --- | --- | --- |
  | `medrag:chat:{sid}:history` | List | 多轮 RAG 的对话历史 |
  | `medrag:chat:{sid}:summary` | String | 多轮 RAG 的 running summary |
  | `medrag:chat:{sid}:tokens` | String(JSON) | 多轮 RAG 的 token 统计（`msg_len` / `msg_token_len`） |
  | `medrag:agent:{sid}:state` | String(JSON) | Agent 的跨轮状态 |
  | `medrag:lock:{sid}` | — | 会话锁 |

- **锁**：用 redis-py 自带的 `Lock`。设置 `timeout` 是为了进程崩溃后锁能自动过期，不会把会话永久锁死；`blocking_timeout` 防止请求无限等待。
- **配置**：`RedisConfig` 只加一个字段 `session_ttl: int = 7 * 24 * 3600`，其余沿用现有配置。
- `redis.asyncio` 的连接池是懒创建的，在哪个事件循环里第一次使用就绑定在哪个 loop 上，和 `AsyncMilvusClient` 的约束一样。

### 4.3 `core/KnowledgeBase.py`（核心）

**客户端**：同步 `MilvusClient` 只保留给建表、建索引这类低频 DDL 用（2.5.14 的 `AsyncMilvusClient` 没有 `has_collection`）；检索和写入走 `AsyncMilvusClient`。
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
| `add_documents` | `async`：`aembed_documents(summaries)` 和 `aembed_documents(texts)` 整批调用、同时执行；稀疏编码走 `to_thread`；最后 `await insert_rows(self.aclient, ...)` |
| `_create_collection` / `build_index` | `async` 外壳，内部用 `await asyncio.to_thread(...)` 调同步 client |
| 新增 `close` | `await self._aclient.close()` + `self.client.close()` |

### 4.4 `core/insert.py`

`insert_rows` / `insert_rows_has_id` / `delete_by_ids` 改 `async`，内部 `await client.insert / upsert / delete(...)`，入参换成 `AsyncMilvusClient`。进度条逻辑不变。

### 4.5 `core/HybridRetriever.py`

- 实现 `async _aget_relevant_documents`，删掉同步版。
- 修复检索参数串台，不再修改共享配置：
  ```python
  req = self.search_config.model_copy(update={"query": inputs.get("input", "")})
  documents = await self.knowledge_base.search(req)
  ```

### 4.6 `core/IngestionPipeline.py`

`__init__(config, kb=None)`；`async run`：`await kb._create_collection()` → 分批 `await kb.add_documents(batch)` → `await kb.build_index()`。批次之间保持串行（瓶颈在 Embedding API 限流，单批内部已经是批量请求）。

### 4.7 `core/DBFactory.py`：删除

`AgentTools` 改为构造时接收 `kb`，不再走进程级 `lru_cache` 单例。

### 4.8 `rag/RagBase.py`

- 抽象方法 `answer` 声明为 `async`。
- `batch_answer` 改为 `await asyncio.gather(*(self.answer(q, ...) for q in queries))`。
- 构造函数接收 `kb=None`，统一在基类里执行 `self.knowledge_base = kb or MedicalHybridKnowledgeBase(config)`，两个子类不再各自创建。

### 4.9 `rag/SimpleRag.py`

`answer` → `async`：`result = await self.rag_chain.ainvoke({"input": query})`。
链本身不用改：`ainvoke` 下 `RunnablePassthrough.assign(milvus_result=retriever)` 会自动走 `retriever.ainvoke`；`format_document_str`、`strip_think_and_time` 是纯 CPU 小函数，保持同步。

### 4.10 `rag/MultiDialogueRag.py`

**异步化**：

| 位置 | 改法 |
| --- | --- |
| `_timed_llm_invoke` | `async def`，`await self.llm.ainvoke(...)` |
| `do_retrieve` | `async def`，`await self.self_retriever.ainvoke(...)` |
| `_get_summary` | `async`，`await (summarize_prompt \| self.llm).ainvoke(...)` |
| `_maybe_compress_history` | `async` |
| `_build_document_context` | 仍然同步（纯 CPU），但 running summary 和 `avg_tokens_per_char` 改为从链的输入里取，不再读实例字段 |

**会话状态迁到 Redis**：

| 原来（进程内存） | 改为 |
| --- | --- |
| `self._histories[sid]` | `self.store.history(f"medrag:chat:{sid}:history")`，直接作为 `RunnableWithMessageHistory` 的 `get_session_history` |
| `self._running_summaries[sid]` | `store.get_json / set_json("medrag:chat:{sid}:summary")` |
| `self._token_meta_store[sid]` | `store.get_json / set_json("medrag:chat:{sid}:tokens")` |
| `self.avg_tokens_per_char`（实例级） | 每次按该会话的 token 统计现算，作为局部变量放进链的输入 |
| 压缩时 `hist.messages = hist.messages[cutoff:]` | `await hist.atrim(cutoff)` |

**对外流程**：把一轮对话拆成"准备 → 执行链 → 收尾"三步，`answer` 和上层的流式输出共用：

```python
async def prepare(self, query, session_id) -> dict:   # 压缩历史，读取摘要和 token 统计，拼好链的输入
async def finish(self, result, session_id) -> None:   # 更新 token 统计

async def answer(self, query, return_document=False, session_id="default"):
    async with self.store.lock(session_id):
        inputs = await self.prepare(query, session_id)
        result = await self.rag_chain.ainvoke(inputs, config={"configurable": {"session_id": session_id}})
        await self.finish(result, session_id)
    ...  # 组装返回值，结构不变
```

上层做流式输出（`rag_chain.astream_events`）时，也是 `async with store.lock(sid)` → `prepare` → `astream_events` → `finish`，不再直接访问 `_running_summaries` 这类私有字段。

### 4.11 `agent/SearchGraph.py`

- 节点 `llm_db_search` / `llm_network_search` / `rag` / `judge` 全部改 `async def`，`llm.invoke`、`tool_node.invoke`、`judge_chain.invoke`、`search_chain.invoke` 统一改为 `await ...ainvoke`（`OutputFixingParser` 支持 async）。
- `finish_success` / `finish_fail` / `judge_router` 是纯状态操作，保持同步。
- `answer` / `run` → `async`，`await self.search_graph.ainvoke(...)`。
- LangGraph 能识别 `partial(async_fn, ...)` 为异步节点，图定义不用改。
- 本身是单轮、每次调用都新建初始状态，没有会话状态，不需要 Redis。

### 4.12 `agent/tools/`

- `AgentTools.__init__(app_config, kb)`。
- `database_search` → `async def`，`await self.kb.search(search_config)`。
- `web_search` → `async def`，`await asyncio.to_thread(self.WEBSEARCH_FUNC, query, cnt)`。`TencentSearch.py` 本身不动。
- `calculator` 是纯 CPU，不改。
- 顺手修掉 `raise "未注册网络检索工具"`（raise 字符串会触发 TypeError），改为 `raise RuntimeError(...)`。

### 4.13 `agent/MedicalAgent.py`

**异步化**：

- 节点 `ask_judge` / `extract_background_info` / `check_update_background` / `judge_split_query` / `search_one` / `gather_answer` 改 `async`，`invoke` → `ainvoke`；`search_one` 里 `await search_graph.run(...)`。
  收益：`Send` 拆出来的多个子查询由线程并行变为协程并发。
- 路由函数 `route_entry` / `route_ask_again` / `route_to_subgraphs` 保持同步。

**会话状态迁到 Redis**，实例变成无状态，一个实例服务所有会话：

- 只持久化**跨轮**需要的字段，存到 `medrag:agent:{sid}:state`：

  | 字段 | 序列化方式 |
  | --- | --- |
  | `dialogue_messages` | `messages_to_dict` |
  | `asking_messages`（二维） | 逐轮 `messages_to_dict` |
  | `background_info`、`running_summary` | 字符串 |
  | `multi_summary` | 字符串列表 |
  | `curr_ask_num`、`max_ask_num` | 整数 |

- 单轮内的中间产物不持久化，每轮开始时重置：`curr_input`、`ask_obj`、`sub_query`、`sub_query_results`、`rewritten_query`、`final_answer`、`performance`（原来 `performance` 会存下每一步的完整 AIMessage，只增不减，放 Redis 没有意义）。
- 对外方法：

  ```python
  async def load_state(self, session_id) -> MedicalAgentState   # 读 Redis，没有就返回初始状态，并重置单轮字段
  async def save_state(self, session_id, state) -> None         # 只写跨轮字段

  async def answer(self, user_input, session_id="default"):
      async with self.store.lock(session_id):
          state = await self.load_state(session_id)
          state["curr_input"] = user_input
          state = await self.app.ainvoke(state)
          await self.save_state(session_id, state)
      return state
  ```

  上层做流式输出（`app.astream`）时同样是"加锁 → `load_state` → `astream` → `save_state`"。
- 删除 `self.state` 和 `_reset_state`（后者的内容挪到 `load_state` 的初始值里）。

### 4.14 `rag/RagEvaluate.py`

`do_evaluate` → `async`：
1. 先用 `asyncio.gather` + `Semaphore` 并发获取所有样本的答案（`await self.rag.answer(...)`）；
2. `ragas.evaluate(...)` 内部自带事件循环（`nest_asyncio`），不能在当前 loop 里直接跑，用 `await asyncio.to_thread(evaluate, ...)`。

### 4.15 `data/annotation.py`

- `annotate_single` → `async`：`await self.llm.ainvoke(messages)`，重试逻辑不变。
- `annotate_dataset` → `async`：用 `asyncio.Semaphore(self.batch_size)` 限制并发，`asyncio.gather` 处理整个数据集，替代现在的串行循环。临时文件、最终文件的写入逻辑不变。
- `AnnotationPipeline.run` / `run_annotation` → `async`。

---

## 5. 不改的部分

| 模块 | 原因 |
| --- | --- |
| `config/loader.py` | 只在启动时读一次 YAML 和 `.env`，没必要异步化 |
| `config/models.py` | 仅给 `RedisConfig` 加 `session_ttl` 一个字段 |
| `embed/sparse.py`、`embed/bm25.py` | CPU 密集（pkuseg 分词、多进程建词表），由调用方用 `to_thread` 包装 |
| `prompts/templates.py`、`rag/utils.py`、`agent/utils.py` | 纯字符串 / 纯计算 |
| `agent/tools/TencentSearch.py` | 只有同步 SDK，由 `AgentTools` 用 `to_thread` 包装 |

---

## 6. 依赖变更

不新增依赖。用到的都是现有依赖已有的能力：`pymilvus` 的 `AsyncMilvusClient`、LangChain / LangGraph 的 `ainvoke`、`httpx.AsyncClient`、`redis.asyncio`。

运行多轮 RAG 和 Agent 需要先执行 `deploy/start.sh` 启动 Redis；`SimpleRAG`、`SearchGraph`、入库、检索不依赖 Redis。

---

## 7. 实施顺序

1. **core**：`session_store`（新增）+ `KnowledgeBase` + `insert` + `HybridRetriever` + `IngestionPipeline`，删除 `DBFactory`；用 `scripts/02`、`03` 验证入库、检索与改造前一致，给 `session_store` 补单元测试。
2. **rag**：`RagBase` + `SimpleRag` + `MultiDialogueRag` + `RagEvaluate`；用 `scripts/04`、`05`、`06` 验证。
3. **agent**：`AgentTools` + `SearchGraph` + `MedicalAgent`；用 `scripts/07`、`08` 验证。
4. **data**：`annotation.py`。
5. **scripts**：入口改为 `asyncio.run(main())`（随前面各步一起改）。
6. **展示层（最后）**：只做调用点适配，服务自身的其他逻辑（鉴权、SQLite 持久化等）不在本次改造范围内。
   - `MedicalRag/api/app.py`：
     - 把 `await run_sync(x.answer, ...)` 改成 `await x.answer(...)`；
     - 在 lifespan 里创建一个共享的 `kb` 和 `store`，注入各组件，退出时关闭；
     - 删除 `agent_sessions` 字典和 `get_or_create_agent`，改为全局一个 `MedicalAgent`；
     - 两个流式接口改用包提供的 `store.lock` / `prepare` / `finish`（`load_state` / `save_state`）。
   - `software/run_api.py`：会话状态已经不在进程内存里，`workers=1` 的限制可以放开（是否调整由你决定）。

## 8. 验证方式

- **功能一致**：每一步改造前后对同一批问题跑脚本，对比检索到的文档 `pk` 与回答。
- **会话存储**：
  - 同一 `session_id` 用两个新建的 `MultiDialogueRag` / `MedicalAgent` 实例先后提问，第二个实例能接上第一个实例的历史；
  - 同一 `session_id` 同时发两个请求，历史中两轮问答完整且不交错；
  - 在 Redis 中能看到对应的 key 和 TTL。
- **并发收益**：写一个小脚本，用 `asyncio.gather` 同时发起 N 个 `SimpleRAG.answer`，对比改造前（线程池）和改造后的总耗时；最后再用服务做一次压测作为展示。
- **不阻塞事件循环**：开启 `loop.set_debug(True)` 并设置 `slow_callback_duration = 0.1`，压测期间日志中不应出现超过 100ms 的慢回调。
