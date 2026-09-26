# MedicalRag 包异步化改造方案（v2.0.0-anyc）

> 目标：把 `MedicalRag` 包的核心 RAG 业务能力（检索、入库、单轮 / 多轮 RAG、Agent、标注）改成原生 `async`，让外部业务方能在一个事件循环里高并发地调用；同时把评测从 RAG 实现中拆出来，做成各个库共用的统一评测流程（评测本身不追求并发）。
> 定位：`MedicalRag` 只负责核心 RAG 业务逻辑，不包含任何 API / 服务代码。HTTP 接口、鉴权、SSE 协议等由外部业务自行实现（当前示例在 `software/backend/`），**本方案不涉及 `software/` 的内容**。包需要做的，是给外部提供完整、无需访问私有字段的异步接口（包括流式接口）。
> 基线：`v2.0.0` @ `19ce2de`（`api/` 已移出包，放到 `software/backend/`）。`core/services.py` 没有必要保留，已在本分支删除，并清理了引用它的 `scripts/00_check_services.py`、`tests/test_services.py` 中的用例和 `software/backend/run_api.py` 里的 `preflight()`（不清理会导致导入失败）。
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
| `rag/RagEvaluate.py` | 循环里逐条 `rag.answer`；与 `BasicRAG` 耦合（见下文"评测与具体实现耦合"） |
| `agent/SearchGraph.py`、`agent/MedicalAgent.py` | 每个图节点都是同步 `llm.invoke` / `tool_node.invoke`，图用 `app.invoke` |
| `agent/tools/*` | `database_search` 同步查 Milvus；腾讯云搜索 SDK 只有同步版本 |
| `data/annotation.py` | 逐条 `llm.invoke`，整个数据集串行标注 |

另外还有几个问题，异步化之后会暴露得更明显，这次一起处理：

- **会话状态串台**：`MultiDialogueRag` 的 `_histories` / `_running_summaries` / `_token_meta_store`，以及 `MedicalAgent.state`，都放在进程内存里：
  - 同一会话的并发请求会交错读写，把历史和摘要写乱；
  - `MultiDialogueRag.avg_tokens_per_char` 是实例级字段，所有会话共用一个值，会话之间会互相影响；
  - `MedicalAgent` 只能一个会话对应一个实例，上层要自己维护 `session_id → MedicalAgent` 字典，而且内存只增不减；
  - 状态绑定在单个进程上，外部业务无法多进程 / 多实例部署。
- **缺少对外的流式接口**：包里没有流式方法，原来的 API 层只能直接调用 `rag_chain.astream_events`、`agent.app.astream`，还要读写 `_running_summaries`、`_maybe_compress_history`、`_update_tokens_metadata`、`agent.state` 这些内部实现。API 移出包之后，这些都应该由包自己封装好。
- **检索参数串台**：`MedicalHybridRetriever` 通过 `self.search_config.query = ...` 修改共享对象，并发时 A 请求可能用上 B 请求的 query。这是进程内多个协程共享同一个对象造成的，和会话无关，改成请求级拷贝就能解决，不需要 Redis。
- **知识库重复创建**：`SimpleRAG`、`MultiDialogueRag`、`IngestionPipeline`、`DBFactory` 各自创建一个 `MedicalHybridKnowledgeBase`，同一进程里有多份 Milvus 连接、Embedding 客户端和 BM25 词表。
- **评测与具体实现耦合**：`RagEvaluate` 放在 `rag/` 下，构造时必须传入 `BasicRAG`，上下文固定取 `metadata['document']`，其他库（例如后续的图谱 RAG、GraphRAG-Bench 里的 LightRAG / HippoRAG 等）没法接入同一套流程；`do_sample` 使用不带种子的 `shuffle()`，每次运行抽到的样本都不同，不同库之间的分数不可比。
- **配置全耦合在一个文件里**：
  - 所有配置（Milvus / Redis / PostgreSQL / Neo4j 连接、LLM、Embedding、数据字段映射、多轮对话、Agent）都写在同一个 `app_config.yaml` 里。换一个模型、换一套数据字段、换一个部署环境，都要改同一个文件，也没法只替换其中一部分。
  - 代码层面也是整体耦合：几乎每个组件的构造函数都要求传入整个 `AppConfig`，实际只用其中一两段。还有跨段读取的情况，比如 `SearchGraph` 读的是 `multi_dialogue_rag.console_debug`，`MedicalAgent` 通过 `search_graph.config.agent` 读别人的配置，`AgentTools` 把整个 `AppConfig` 序列化后交给 `DBFactory` 做缓存 key。
  - `AnnotationPipeline` 读取的 `config.data.batch_size` 在 `DataConfig` 里根本不存在，运行会直接报错。
- **小 bug**：`_encode_query` 在 Milvus 托管 BM25 时执行 `data = data`，会报变量未定义；`add_documents` 在循环里逐条调用 Embedding API。

---

## 2. 设计要点

1. **IO 密集 → 原生异步**
   - LLM：`invoke` → `ainvoke`（`ChatOpenAI` / `ChatOllama` 原生支持）
   - Embedding：`embed_*` → `aembed_*`
   - Milvus 读写：`AsyncMilvusClient`（现有的 pymilvus 2.5.14 已自带）
   - Redis：`redis.asyncio`（现有依赖 `redis>=5.0` 自带）
   - LangChain 链、LangGraph 图、ToolNode：`invoke` → `ainvoke`
2. **CPU 密集或只有同步 SDK → `asyncio.to_thread`**：BM25 分词（pkuseg）、腾讯云搜索。
3. **直接把原方法改成 `async def`，名字不变**，不保留同步版本。脚本和外部调用方统一用 `await` / `asyncio.run`。
4. **配置分离**：YAML 按领域拆成多个文件；组件构造函数只接收自己用到的那一段配置，不再接收整个 `AppConfig`。见 4.0。
5. **依赖显式注入**：知识库 `kb`、会话存储 `store` 由调用方创建后传入各组件，组件内部不再自己创建（组件拿不到 Milvus / Redis 的连接配置，也不应该拿）。调用方可以全局只用一个实例。删掉 `DBFactory`。
6. **会话状态放进 Redis**（使用 `deploy/` 下已有的 Redis）：多轮对话历史、摘要、token 统计、Agent 跨轮状态全部存到 Redis，组件本身变成无状态。同一会话用 Redis 分布式锁串行执行，不同会话完全并行，多个进程、多个 worker 之间也能共享会话。
7. **检索参数请求级拷贝**，不再修改共享对象。
8. **流式由包提供**：多轮 RAG 和 Agent 各提供一个 `stream` 异步生成器，内部完成加锁、读写会话、执行链 / 图，外部只负责把事件转换成自己的协议（SSE、WebSocket 等）。
9. **评测独立**：评测拆成独立的 `eval/` 模块，只依赖统一的"答题函数"和样本记录格式，不依赖任何具体的 RAG 实现；逐条串行执行，不做并发。

---

## 3. 改造后的对外接口

| 组件 | 改造前 | 改造后 |
| --- | --- | --- |
| `ConfigLoader` | 读取单个 `app_config.yaml` | 读取配置目录下的多个 YAML，支持按文件覆盖，见 4.0；`ConfigLoader().config` 用法不变 |
| `RedisSessionStore(redis)`（新增） | — | 会话存储，见 4.2 |
| `MedicalHybridKnowledgeBase` | `(app_config)`；`search(req)`、`add_documents(docs)`、`build_index()`、`_create_collection()` | `(milvus, embedding)`；同名方法全部 `async`；新增 `async close()` |
| `IngestionPipeline` | `(config)`；`run(records) -> bool` | `(data, kb)`；`async run(records)` |
| `SimpleRAG` | `(config, search_config=None)`；`answer(query, return_document)` | `(llm, kb, search_config=None)`；`async answer(...)`；`batch_answer` 改为 `asyncio.gather` 并发 |
| `MultiDialogueRag` | `(config, search_config=None)`；`answer(query, return_document, session_id)` | `(llm, dialogue, kb, store, search_config=None)`；`async answer(...)`，会话数据存 Redis；新增 `stream(query, session_id)` |
| `SearchGraph` | `(config, power_model)`；`answer(query)`、`run(state)` | `(llm, agent, kb, power_model)`；`async answer / run` |
| `MedicalAgent` | `(config, power_model)`；`answer(user_input)`，一个实例对应一个会话 | `(llm, agent, kb, store, power_model)`；`async answer(user_input, session_id)`，**一个实例服务所有会话**；新增 `stream(user_input, session_id)` |
| `RagasRagEvaluate`（删除） | `RagasRagEvaluate(rag, dataset, llm, emb).do_evaluate(...)` | 由 `MedicalRag.eval` 替代：`load_samples` → `collect` → `score`，见 4.14 |
| `SimpleAnnotator` / `AnnotationPipeline` | `AnnotationPipeline(config)`；`annotate_single`、`annotate_dataset`、`run` | `AnnotationPipeline(llm, batch_size=10)`；全部 `async`，按 `batch_size` 限制并发 |

表中参数名即对应的配置段：`llm: LLMConfig`、`embedding: EmbeddingConfig`、`milvus: MilvusConfig`、`redis: RedisConfig`、`data: DataConfig`、`dialogue: MultiDialogueRagConfig`、`agent: AgentConfig`。返回值结构保持不变，上层只需要把 `x.answer(...)` 改成 `await x.answer(...)`；`MedicalAgent` 另外需要传 `session_id`。

组装示例：

```python
cfg = ConfigLoader().config
kb = MedicalHybridKnowledgeBase(cfg.milvus, cfg.embedding)
store = RedisSessionStore(cfg.redis)
rag = MultiDialogueRag(cfg.llm, cfg.multi_dialogue_rag, kb, store)
agent = MedicalAgent(cfg.llm, cfg.agent, kb, store, power_model=create_llm_client(cfg.llm))
```

---

## 4. 逐模块改动

### 4.0 `config/`：配置分离

**文件拆分**：删除 `app_config.yaml`，在 `src/MedicalRag/config/` 下按领域拆成 5 个文件。**顶层 key 与原来完全一致**，所以 `AppConfig` 的结构不用改：

| 文件 | 包含的顶层 key | 什么时候改 |
| --- | --- | --- |
| `storage.yaml` | `milvus`、`redis`、`postgres`、`neo4j` | 换部署环境 |
| `models.yaml` | `llm`、`embedding` | 换模型 / 提供商 |
| `data.yaml` | `data` | 换数据集的字段映射 |
| `dialogue.yaml` | `multi_dialogue_rag` | 调整多轮对话策略 |
| `agent.yaml` | `agent` | 调整 Agent 策略 |

**加载方式**：`ConfigLoader(config_dir=None, overrides=())`：

```python
ConfigLoader()                                              # 默认读取包内 config/ 目录下所有 *.yaml
ConfigLoader("my_conf/")                                    # 读取自己的配置目录
ConfigLoader(overrides=["exp/models_qwen.yaml"])            # 默认配置 + 只替换模型配置
```

- 先读 `config_dir` 下所有 `*.yaml`，再按顺序读 `overrides` 里的文件；读到的顶层 key 合并进同一个 dict，**`overrides` 中的文件整段覆盖同名的顶层 key**。这样就能只替换某一个领域的配置，比如评测时换一份 `models.yaml` 对比不同模型。
- 同一个目录内出现重复的顶层 key 直接报错，避免两个文件悄悄互相覆盖。
- 支持环境变量 `MEDRAG_CONFIG_DIR` 指定默认目录。
- 加载时记录每个顶层 key 来自哪个文件。`change(..., save=True)` 只把被修改的那几段写回各自的来源文件，不再把全部配置写进一个文件；`save_path` 改为目录参数，指定后写到该目录下的同名文件。
- 读取 `deploy/.env` 的 `load_deploy_env()` 保持不变。

**代码解耦**：组件只接收自己需要的配置段（见第 3 节的接口表），不再接收整个 `AppConfig`：

| 组件 | 原来从 `AppConfig` 读取 | 改为直接接收 |
| --- | --- | --- |
| `MedicalHybridKnowledgeBase` | `milvus`、`embedding` | `milvus: MilvusConfig`、`embedding: EmbeddingConfig` |
| `IngestionPipeline` | `data`、`embedding.text_sparse` | `data: DataConfig`、`kb`（稀疏词表检查改为通过 `kb` 进行） |
| `BasicRAG` / `SimpleRAG` | `llm`、`milvus.collection_name` | `llm: LLMConfig`、`kb`（`collection_name` 从 `kb` 取） |
| `MultiDialogueRag` | `llm`、`multi_dialogue_rag` | `llm: LLMConfig`、`dialogue: MultiDialogueRagConfig` |
| `SearchGraph` | `llm`、`agent`、`multi_dialogue_rag.console_debug` | `llm: LLMConfig`、`agent: AgentConfig`；`console_debug` 挪到 `AgentConfig` 自己的字段 |
| `MedicalAgent` | `llm`、`search_graph.config.agent` | `llm: LLMConfig`、`agent: AgentConfig` |
| `AgentTools` | 整个 `AppConfig`（用于 `DBFactory` 缓存） | `kb`、`network_search_cnt: int` |
| `AnnotationPipeline` | `llm`、`data.batch_size`（不存在的字段） | `llm: LLMConfig`、`batch_size: int = 10`（顺带修掉这个 bug） |
| `RedisSessionStore` | — | `redis: RedisConfig` |

`AppConfig` 只保留为 `ConfigLoader` 的汇总结果，方便调用方一次拿到所有配置段；包内组件不再依赖它。`SearchGraph.py` 里没有用到的 `ConfigLoader` 导入一并删除。

### 4.1 `core/utils.py`

客户端本身不用改，都原生支持 async。唯一要补的是：配置了 `proxy` 时，异步请求走的是 `http_async_client`。

```python
if config.proxy:
    kwargs["http_client"] = httpx.Client(proxy=config.proxy)            # 顺带修掉现在传 dict 的写法
    kwargs["http_async_client"] = httpx.AsyncClient(proxy=config.proxy)
```

### 4.2 新增 `core/session_store.py`：Redis 会话存储

使用 `redis.asyncio`，构造时只接收 `RedisConfig`，连接信息直接复用 `redis.url()`（密码来自 `deploy/.env`）。整个文件只有两个类，预计 80 行左右。

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

`__init__(data, kb)`；`async run`：`await kb._create_collection()` → 分批 `await kb.add_documents(batch)` → `await kb.build_index()`。批次之间保持串行（瓶颈在 Embedding API 限流，单批内部已经是批量请求）。

### 4.7 `core/DBFactory.py`：删除

`AgentTools` 改为构造时接收 `kb`，不再走进程级 `lru_cache` 单例。

### 4.8 `rag/RagBase.py`

- 抽象方法 `answer` 声明为 `async`。
- `batch_answer` 改为 `await asyncio.gather(*(self.answer(q, ...) for q in queries))`。
- 构造函数改为 `(llm, kb, search_config=None)`，由基类统一保存 `kb`，两个子类不再各自创建知识库；默认检索配置里的 `collection_name` 从 `kb` 取。

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

**对外流程**：一轮对话固定是"加锁 → 准备输入 → 执行链 → 更新 token 统计"。准备和收尾写成两个私有方法，阻塞版和流式版共用：

```python
async def _prepare(self, query, session_id) -> dict:  # 压缩历史，读取摘要和 token 统计，拼好链的输入
async def _finish(self, result, session_id) -> None:  # 更新 token 统计

async def answer(self, query, return_document=False, session_id="default"):
    async with self.store.lock(session_id):
        inputs = await self._prepare(query, session_id)
        result = await self.rag_chain.ainvoke(inputs, config=...)
        await self._finish(result, session_id)
    ...  # 组装返回值，结构不变

async def stream(self, query, session_id="default") -> AsyncIterator[dict]:
    async with self.store.lock(session_id):
        inputs = await self._prepare(query, session_id)
        result = None
        async for event in self.rag_chain.astream_events(inputs, config=..., version="v2"):
            if event["event"] == "on_chain_end" and event["name"] == "rag":
                result = event["data"]["output"]
            yield event
        if result:
            await self._finish(result, session_id)
```

`stream` 原样透传 LangChain 的 `astream_events(v2)` 事件，不在包里定义新的事件格式。链中已有的 `run_name`（`rewritten_query` / `search_documents` / `generate` / `rag`）就是外部识别阶段的依据，写进 docstring。

### 4.11 `agent/SearchGraph.py`

- 节点 `llm_db_search` / `llm_network_search` / `rag` / `judge` 全部改 `async def`，`llm.invoke`、`tool_node.invoke`、`judge_chain.invoke`、`search_chain.invoke` 统一改为 `await ...ainvoke`（`OutputFixingParser` 支持 async）。
- `finish_success` / `finish_fail` / `judge_router` 是纯状态操作，保持同步。
- `answer` / `run` → `async`，`await self.search_graph.ainvoke(...)`。
- LangGraph 能识别 `partial(async_fn, ...)` 为异步节点，图定义不用改。
- 本身是单轮、每次调用都新建初始状态，没有会话状态，不需要 Redis。

### 4.12 `agent/tools/`

- `AgentTools.__init__(kb, network_search_cnt)`。
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
  async def _load_state(self, session_id) -> MedicalAgentState  # 读 Redis，没有就返回初始状态，并重置单轮字段
  async def _save_state(self, session_id, state) -> None        # 只写跨轮字段

  async def answer(self, user_input, session_id="default"):
      async with self.store.lock(session_id):
          state = await self._load_state(session_id)
          state["curr_input"] = user_input
          state = await self.app.ainvoke(state)
          await self._save_state(session_id, state)
      return state
  ```

- 流式版本，原样透传 LangGraph `astream(stream_mode="updates")` 的 `{节点名: 更新}`，外部按节点名（`ask` / `extract_ask_and_reply` / `check_update_background` / `split_query` / `search_one` / `answer`）识别阶段：

  ```python
  async def stream(self, user_input, session_id="default") -> AsyncIterator[dict]:
      async with self.store.lock(session_id):
          state = await self._load_state(session_id)
          state["curr_input"] = user_input
          async for chunk in self.app.astream(state, stream_mode="updates"):
              for updates in chunk.values():
                  merge(state, updates)          # sub_query_results 按 add reducer 追加，其余直接覆盖
              yield chunk
          await self._save_state(session_id, state)
  ```
- 删除 `self.state` 和 `_reset_state`（后者的内容挪到 `load_state` 的初始值里）。`load_state` / `save_state` 改为私有方法 `_load_state` / `_save_state`，外部只需要用 `answer` 和 `stream`。

### 4.14 新增 `eval/`：独立的统一评测模块（替代 `rag/RagEvaluate.py`）

**目标**：不同的库（当前的 Milvus 混合检索 RAG、后续的图谱 RAG、GraphRAG-Bench 里的第三方实现等）走完全相同的评测流程：同一批样本、同一套指标、同一个评测模型，分数才可以横向比较。评测只追求一致、可复现，**不做并发**。

**思路**：把评测拆成"生成答案"和"打分"两步，中间用一份 JSONL 样本记录文件衔接。各个库只负责按统一格式给出答案和上下文，打分部分完全共用。

```
src/MedicalRag/eval/
├── __init__.py
├── runner.py      # 样本定义、采样、收集答案、保存 / 读取、打分；不依赖任何具体 RAG 实现
└── adapters.py    # 把具体实现包装成统一的答题函数（先提供 MedicalRag 的）
```

`runner.py` 只依赖 `ragas`、`datasets`、LangChain 基础类型，**不 import `MedicalRag` 的其他模块**；以后如果要把评测移出包，可以整体搬走。

**统一样本格式**（也是 JSONL 每一行的格式）：

```python
@dataclass
class EvalSample:
    question: str
    reference: str                 # 参考答案
    answer: str = ""               # 被测系统的回答
    contexts: list[str] = field(default_factory=list)   # 被测系统检索到的上下文
```

**统一答题函数**：输入问题，返回 `(answer, contexts)`，同步、异步函数都可以：

```python
AnswerFn = Callable[[str], tuple[str, list[str]] | Awaitable[tuple[str, list[str]]]]
```

**流程函数**：

```python
def load_samples(path, question_field, reference_field, sample_size=None, seed=42) -> list[EvalSample]
    # 读取数据集并采样；固定 seed，保证每个库拿到完全相同的样本

async def collect(samples, answer_fn) -> list[EvalSample]
    # 逐条串行调用 answer_fn，填入 answer / contexts；单条失败记录错误信息后继续

def save_samples(samples, path) / load_saved(path) -> list[EvalSample]
    # JSONL 读写：答案落盘后，可以在不重新生成的情况下重复打分、对比不同的库

def score(samples, llm, embeddings, metrics=None) -> EvaluationResult
    # 用 ragas 打分，默认沿用现在的 4 个指标：
    # AnswerRelevancy / Faithfulness / ContextRecall / ContextPrecision
```

- `score` 是同步函数，在事件循环之外调用（先 `asyncio.run(collect(...))`，再 `score(...)`），避免 ragas 自带的事件循环和调用方冲突。
- 其他库如果本身已经能输出这种格式的 JSONL（比如自带脚本的第三方实现），可以跳过 `collect`，直接 `load_saved` → `score`。

**`adapters.py`**：先只提供 MedicalRag 自己的适配器：

```python
def from_medical_rag(rag, context_field="document") -> AnswerFn:
    async def answer_fn(question):
        r = await rag.answer(question, return_document=True)
        return r["answer"], [d.metadata.get(context_field) or d.page_content for d in r["documents"]]
    return answer_fn
```

多轮 RAG 评测时，每个问题使用独立的 `session_id`，避免题与题之间共享历史。

**删除**：`rag/RagEvaluate.py`。

### 4.15 `data/annotation.py`

- `annotate_single` → `async`：`await self.llm.ainvoke(messages)`，重试逻辑不变。
- `annotate_dataset` → `async`：用 `asyncio.Semaphore(self.batch_size)` 限制并发，`asyncio.gather` 处理整个数据集，替代现在的串行循环。临时文件、最终文件的写入逻辑不变。
- `AnnotationPipeline.run` / `run_annotation` → `async`。

---

## 5. 不改的部分

| 模块 | 原因 |
| --- | --- |
| `config/`（异步化） | 只在启动时读一次 YAML 和 `.env`，没必要异步化；拆分改动见 4.0，`models.py` 只新增 `RedisConfig.session_ttl`、`AgentConfig.console_debug` 两个字段 |
| `embed/sparse.py`、`embed/bm25.py` | CPU 密集（pkuseg 分词、多进程建词表），由调用方用 `to_thread` 包装 |
| `prompts/templates.py`、`rag/utils.py`、`agent/utils.py` | 纯字符串 / 纯计算 |
| `agent/tools/TencentSearch.py` | 只有同步 SDK，由 `AgentTools` 用 `to_thread` 包装 |

---

## 6. 依赖变更

不新增依赖。用到的都是现有依赖已有的能力：`pymilvus` 的 `AsyncMilvusClient`、LangChain / LangGraph 的 `ainvoke`、`httpx.AsyncClient`、`redis.asyncio`。

运行多轮 RAG 和 Agent 需要先执行 `deploy/start.sh` 启动 Redis；`SimpleRAG`、`SearchGraph`、入库、检索不依赖 Redis。

---

## 7. 实施顺序

0. **config**：拆分 YAML、改造 `ConfigLoader`；确认拆分前后加载出的 `AppConfig` 完全相等。
1. **core**：`session_store`（新增）+ `KnowledgeBase` + `insert` + `HybridRetriever` + `IngestionPipeline`，删除 `DBFactory`；用 `scripts/02`、`03` 验证入库、检索与改造前一致，给 `session_store` 补单元测试。
2. **rag**：`RagBase` + `SimpleRag` + `MultiDialogueRag`；用 `scripts/04`、`06` 验证。
3. **agent**：`AgentTools` + `SearchGraph` + `MedicalAgent`；用 `scripts/07`、`08` 验证。
4. **data**：`annotation.py`。
5. **eval**：新增 `eval/`，删除 `rag/RagEvaluate.py`；`scripts/05_eval_rag.py` 改为 `load_samples` → `collect` → `save_samples` → `score`，指标与原 `RagasRagEvaluate` 保持一致。
6. **scripts**：入口改为 `asyncio.run(main())`（随前面各步一起改）。

`software/` 不在本方案范围内。包的接口变化（方法变为 `async`、`MedicalAgent` 改为无状态并新增 `session_id`、新增 `stream`）会导致 `software/backend` 需要跟着适配，这部分留给外部业务实现。

## 8. 验证方式

- **配置拆分**：
  - 拆分后 `ConfigLoader().config` 与拆分前读取 `app_config.yaml` 得到的 `AppConfig` 完全相等；
  - 用 `ConfigLoader(overrides=["xxx/models.yaml"])` 只替换模型配置时，其他配置段保持不变；
  - 同一目录下两个文件包含同一个顶层 key 时，加载报错。
- **功能一致**：每一步改造前后对同一批问题跑脚本，对比检索到的文档 `pk` 与回答。
- **会话存储**：
  - 同一 `session_id` 用两个新建的 `MultiDialogueRag` / `MedicalAgent` 实例先后提问，第二个实例能接上第一个实例的历史；
  - 同一 `session_id` 同时发两个请求，历史中两轮问答完整且不交错；
  - 在 Redis 中能看到对应的 key 和 TTL。
- **流式接口**：`MultiDialogueRag.stream` / `MedicalAgent.stream` 能按顺序产出各阶段事件，结束后 Redis 中的历史和状态与 `answer` 的结果一致。
- **评测一致性**：同一数据集、同一 `seed` 多次 `load_samples` 得到完全相同的样本；对同一份 JSONL 重复 `score`，指标结果一致（在评测模型 `temperature=0` 的前提下）。
- **并发收益**：写一个小脚本，用 `asyncio.gather` 同时发起 N 个 `SimpleRAG.answer`，对比改造前（在线程池里跑同步版）和改造后的总耗时。
- **不阻塞事件循环**：开启 `loop.set_debug(True)` 并设置 `slow_callback_duration = 0.1`，并发测试期间日志中不应出现超过 100ms 的慢回调。
