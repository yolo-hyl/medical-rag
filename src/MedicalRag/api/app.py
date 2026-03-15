from __future__ import annotations

import asyncio
import json
import threading
import time
from contextlib import asynccontextmanager
from copy import deepcopy
from functools import partial
from typing import Any, Dict, List, Literal, Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from langchain_core.documents import Document
from pydantic import BaseModel, Field

from MedicalRag.agent.MedicalAgent import MedicalAgent
from MedicalRag.agent.SearchGraph import SearchGraph
from MedicalRag.config.loader import ConfigLoader
from MedicalRag.config.models import FusionSpec, SearchRequest, SingleSearchRequest
from MedicalRag.core.IngestionPipeline import IngestionPipeline
from MedicalRag.core.utils import create_embedding_client, create_llm_client
from MedicalRag.rag.MultiDialogueRag import MultiDialogueRag
from MedicalRag.rag.SimpleRag import SimpleRAG
from MedicalRag.api import auth as _auth

# ---------------------------------------------------------------------------
# Module-level state — populated during lifespan startup
# ---------------------------------------------------------------------------

state: dict = {}
_agent_lock = threading.Lock()

# Stage names emitted by MultiDialogueRag._setup_chain() via .with_config(run_name=...)
_STAGE_MAP = {
    "rewritten_query": "正在理解您的问题...",
    "search_documents": "正在检索相关文档...",
    "generate": "正在生成回答...",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    _auth.init_db()
    config = ConfigLoader().config
    state["config"] = config
    state["simple_rag"] = SimpleRAG(config)
    state["multi_rag"] = MultiDialogueRag(config)
    state["search_agent"] = SearchGraph(config, power_model=create_llm_client(config.llm))
    state["agent_sessions"] = {}   # session_id -> MedicalAgent
    yield
    state.clear()


app = FastAPI(
    title="Medical RAG API",
    version="1.0.0",
    description="医疗知识问答 RAG 系统后端 API",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def run_sync(fn, *args, **kwargs):
    """Run a blocking function in a thread pool without blocking the event loop."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, partial(fn, *args, **kwargs))


def get_or_create_agent(session_id: str) -> MedicalAgent:
    """Return cached MedicalAgent for the session, creating one if needed."""
    with _agent_lock:
        if session_id not in state["agent_sessions"]:
            config = state["config"]
            state["agent_sessions"][session_id] = MedicalAgent(
                config, power_model=create_llm_client(config.llm)
            )
        return state["agent_sessions"][session_id]


def _doc_to_source(doc: Document) -> dict:
    return {
        "source": doc.metadata.get("source"),
        "source_name": doc.metadata.get("source_name"),
        "distance": doc.metadata.get("distance"),
        "summary": doc.metadata.get("summary"),
        "content_preview": doc.page_content[:200],
    }


def _sse_line(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


def get_optional_user(request: Request) -> Optional[dict]:
    """Extract auth user from X-Auth-Token header. Returns None if missing/invalid."""
    token = request.headers.get("X-Auth-Token", "")
    if not token:
        return None
    user_id = _auth.verify_token(token)
    if not user_id:
        return None
    return {"user_id": user_id}


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class SourceDoc(BaseModel):
    source: Optional[str] = None
    source_name: Optional[str] = None
    distance: Optional[float] = None
    summary: Optional[str] = None
    content_preview: str


# --- Auth ---
class RegisterRequest(BaseModel):
    phone: str = Field(min_length=11, max_length=11)
    password: str = Field(min_length=6)
    role: Literal["doctor", "patient", "admin"]


class LoginRequest(BaseModel):
    phone: str
    password: str


class LoginResponse(BaseModel):
    user_id: str
    token: str
    role: str


# --- Health ---
class HealthResponse(BaseModel):
    status: str
    services: dict


# --- Ingest ---
class IngestRequest(BaseModel):
    records: List[Dict[str, Any]]
    drop_old: bool = Field(default=False, description="是否先清空现有集合")


class IngestResponse(BaseModel):
    success: bool
    records_count: int
    message: str


# --- Search ---
class SearchDocRequest(BaseModel):
    query: str
    limit: int = Field(default=5, ge=1, le=50)
    filter_expr: Optional[str] = Field(default=None, description="Milvus过滤表达式，如 'source == \"qa\"'")
    use_hybrid: bool = Field(default=True, description="True=稠密+稀疏混合检索，False=仅稠密")


class SearchDocResponse(BaseModel):
    query: str
    documents: List[SourceDoc]
    search_time: float


# --- Ask (single-turn RAG) ---
class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: List[SourceDoc]
    search_time: float
    generation_time: float


# --- Chat (multi-turn RAG) ---
class ChatRequest(BaseModel):
    question: str
    session_id: str = Field(default="default", description="多轮对话会话ID")


class ChatResponse(BaseModel):
    session_id: str
    question: str
    answer: str
    sources: List[SourceDoc]
    search_time: float
    rewrite_time: float
    generation_time: float


# --- Eval ---
class EvalRequest(BaseModel):
    eval_file: str = Field(description="服务器端JSONL评测数据集路径")
    sample_size: int = Field(default=10, ge=1)
    query_field: str = Field(default="question", description="数据集中问题字段名")
    reference_field: str = Field(default="answer", description="数据集中参考答案字段名")


class EvalResponse(BaseModel):
    sample_size: int
    metrics: Dict[str, float]


# --- Search Agent (single-turn) ---
class SearchAgentRequest(BaseModel):
    question: str


class SearchAgentResponse(BaseModel):
    question: str
    answer: str
    documents: List[SourceDoc] = Field(default_factory=list)


# --- Medical Agent (multi-turn) ---
class AgentRequest(BaseModel):
    question: str
    session_id: str = Field(default="default", description="智能Agent会话ID")


class AgentResponse(BaseModel):
    session_id: str
    question: str
    needs_clarification: bool
    clarification_questions: List[str]
    answer: Optional[str]


# ---------------------------------------------------------------------------
# Auth Endpoints
# ---------------------------------------------------------------------------

@app.post("/api/auth/register", summary="注册")
async def register(req: RegisterRequest):
    try:
        user_id = await run_sync(_auth.register_user, req.phone, req.password, req.role)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"detail": str(e)})
    return {"user_id": user_id}


@app.post("/api/auth/login", response_model=LoginResponse, summary="登录")
async def login(req: LoginRequest):
    try:
        user_id, token, role = await run_sync(_auth.login_user, req.phone, req.password)
    except ValueError as e:
        return JSONResponse(status_code=401, content={"detail": str(e)})
    return LoginResponse(user_id=user_id, token=token, role=role)


@app.get("/api/auth/me", summary="当前用户信息")
async def me(request: Request):
    token = request.headers.get("X-Auth-Token", "")
    if not token:
        return JSONResponse(status_code=401, content={"detail": "未提供认证令牌"})
    user_id = await run_sync(_auth.verify_token, token)
    if not user_id:
        return JSONResponse(status_code=401, content={"detail": "令牌无效或已过期"})
    info = await run_sync(_auth.get_user_info, user_id)
    if not info:
        return JSONResponse(status_code=404, content={"detail": "用户不存在"})
    return info


# ---------------------------------------------------------------------------
# History Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/history/{service_type}", summary="会话列表")
async def list_sessions(service_type: str, request: Request):
    user = get_optional_user(request)
    if not user:
        return JSONResponse(status_code=401, content={"detail": "需要登录"})
    if service_type not in ("chat", "agent"):
        return JSONResponse(status_code=400, content={"detail": "service_type 必须为 chat 或 agent"})
    sessions = await run_sync(_auth.list_sessions, user["user_id"], service_type)
    return sessions


@app.get("/api/history/{service_type}/{session_id}", summary="会话消息列表")
async def get_session_messages(service_type: str, session_id: str, request: Request):
    user = get_optional_user(request)
    if not user:
        return JSONResponse(status_code=401, content={"detail": "需要登录"})
    try:
        messages = await run_sync(_auth.get_messages, session_id, user["user_id"])
    except PermissionError as e:
        return JSONResponse(status_code=403, content={"detail": str(e)})
    return messages


@app.delete("/api/history/{service_type}/{session_id}", summary="删除会话")
async def delete_session(service_type: str, session_id: str, request: Request):
    user = get_optional_user(request)
    if not user:
        return JSONResponse(status_code=401, content={"detail": "需要登录"})
    ok = await run_sync(_auth.delete_session, session_id, user["user_id"])
    return {"success": ok}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health", response_model=HealthResponse, summary="健康检查")
async def health():
    services = {
        "simple_rag": "simple_rag" in state,
        "multi_rag": "multi_rag" in state,
        "search_agent": "search_agent" in state,
        "agent_sessions": "agent_sessions" in state,
    }
    return HealthResponse(
        status="ok" if all(services.values()) else "degraded",
        services=services,
    )


@app.post("/api/ingest", response_model=IngestResponse, summary="录入数据")
async def ingest(req: IngestRequest):
    """将医疗问答记录写入 Milvus 向量数据库。"""
    def _run():
        cfg = deepcopy(state["config"])
        cfg.milvus.drop_old = req.drop_old
        pipeline = IngestionPipeline(cfg)
        success = pipeline.run(req.records)
        return success

    success = await run_sync(_run)
    return IngestResponse(
        success=success,
        records_count=len(req.records),
        message="数据录入成功" if success else "数据录入失败，请查看服务端日志",
    )


@app.post("/api/search", response_model=SearchDocResponse, summary="检索文档")
async def search_documents(req: SearchDocRequest):
    """在知识库中进行向量检索，返回原始文档列表（不经过LLM生成）。"""
    def _run():
        config = state["config"]
        kb = state["simple_rag"].knowledge_base
        collection_name = config.milvus.collection_name
        output_fields = ["summary", "document", "source", "source_name", "lt_doc_id", "chunk_id", "text"]

        if req.use_hybrid:
            requests = [
                SingleSearchRequest(
                    anns_field="summary_dense",
                    metric_type="COSINE",
                    search_params={"ef": 64},
                    limit=req.limit * 2,
                    expr=req.filter_expr or "",
                ),
                SingleSearchRequest(
                    anns_field="text_sparse",
                    metric_type="IP",
                    search_params={"drop_ratio_search": 0.0},
                    limit=req.limit * 2,
                    expr=req.filter_expr or "",
                ),
            ]
            fuse = FusionSpec(method="rrf", k=60)
        else:
            requests = [
                SingleSearchRequest(
                    anns_field="summary_dense",
                    metric_type="COSINE",
                    search_params={"ef": 64},
                    limit=req.limit,
                    expr=req.filter_expr or "",
                )
            ]
            fuse = None

        search_req = SearchRequest(
            query=req.query,
            collection_name=collection_name,
            requests=requests,
            output_fields=output_fields,
            fuse=fuse,
            limit=req.limit,
        )
        t0 = time.time()
        docs = kb.search(search_req)
        return docs, time.time() - t0

    docs, elapsed = await run_sync(_run)
    return SearchDocResponse(
        query=req.query,
        documents=[SourceDoc(**_doc_to_source(d)) for d in docs],
        search_time=elapsed,
    )


@app.post("/api/ask", response_model=AskResponse, summary="单轮问答")
async def ask(req: AskRequest):
    """单轮医疗问答（SimpleRAG），每次独立，不保留对话历史。"""
    result = await run_sync(
        state["simple_rag"].answer, req.question, return_document=True
    )
    sources = [SourceDoc(**_doc_to_source(d)) for d in result.get("documents", [])]
    return AskResponse(
        question=req.question,
        answer=result["answer"],
        sources=sources,
        search_time=result["search_time"],
        generation_time=result["generation_time"],
    )


@app.post("/api/chat", response_model=ChatResponse, summary="多轮对话（阻塞）")
async def chat(req: ChatRequest, request: Request):
    """多轮对话 RAG，携带 session_id 保留历史上下文。等待全部完成后返回。"""
    result = await run_sync(
        state["multi_rag"].answer,
        req.question,
        return_document=True,
        session_id=req.session_id,
    )
    sources = [SourceDoc(**_doc_to_source(d)) for d in result.get("documents", [])]

    # Persist message if authenticated
    user = get_optional_user(request)
    if user:
        try:
            import json as _json
            rewritten = result.get("llm_rewritten_query", {})
            rewritten_str = rewritten.get("msg", "") if isinstance(rewritten, dict) else ""
            chat_extra = _json.dumps({
                "rewritten_query": rewritten_str,
                "documents": [_doc_to_source(d) for d in result.get("documents", [])],
            }, ensure_ascii=False)
            _auth.upsert_session(req.session_id, user["user_id"], "chat", title=req.question[:20])
            _auth.save_message(req.session_id, "user", req.question)
            _auth.save_message(req.session_id, "assistant", result["answer"], extra_data=chat_extra)
        except Exception:
            pass

    return ChatResponse(
        session_id=req.session_id,
        question=req.question,
        answer=result["answer"],
        sources=sources,
        search_time=result["search_time"],
        rewrite_time=result["rewriten_generate_time"],
        generation_time=result["out_generate_time"],
    )


@app.post("/api/chat/stream", summary="多轮对话（SSE流式）")
async def chat_stream(req: ChatRequest, request: Request):
    """
    多轮对话 SSE 流式响应，使用 LangChain 原生 astream_events。

    事件类型：
    - `progress` — 阶段提示（改写中/检索中/生成中）
    - `rewrite`  — 改写后的查询文本
    - `documents` — 检索到的文档列表（生成前即可推送）
    - `token` — LLM 生成的单个 token（如 LLM 支持流式输出）
    - `answer` — 最终完整答案
    - `done` — 流结束
    """
    user = get_optional_user(request)

    async def event_generator():
        multi_rag = state["multi_rag"]
        multi_rag._maybe_compress_history(req.session_id)
        final_result: dict = {}
        token_buffer: list[str] = []
        accumulated_rewrite: str = ""
        accumulated_docs: list = []

        try:
            async for event in multi_rag.rag_chain.astream_events(
                {
                    "original_input": req.question,
                    "running_summary": multi_rag._running_summaries.get(req.session_id, ""),
                    "session_id": req.session_id,
                },
                config={"configurable": {"session_id": req.session_id}},
                version="v2",
            ):
                ename = event.get("name", "")
                etype = event.get("event", "")

                if etype == "on_chain_start" and ename in _STAGE_MAP:
                    yield _sse_line({"type": "progress", "message": _STAGE_MAP[ename]})

                elif etype == "on_chain_end" and ename == "rewritten_query":
                    output = event["data"].get("output", {})
                    if isinstance(output, dict):
                        rq = output.get("llm_rewritten_query", {})
                        rewritten = rq.get("msg", "") if isinstance(rq, dict) else str(rq)
                    elif isinstance(output, str):
                        rewritten = output
                    else:
                        rewritten = getattr(output, "content", "")
                    if rewritten:
                        accumulated_rewrite = rewritten
                        yield _sse_line({"type": "rewrite", "data": rewritten})

                elif etype == "on_chain_end" and ename == "search_documents":
                    output = event["data"].get("output", {})
                    if isinstance(output, dict):
                        milvus_result = output.get("milvus_result", {})
                        doc_list = milvus_result.get("documents", []) if isinstance(milvus_result, dict) else []
                        if doc_list:
                            accumulated_docs = [_doc_to_source(d) for d in doc_list]
                            yield _sse_line({"type": "documents", "data": accumulated_docs})

                elif etype == "on_retriever_end":
                    raw = event["data"].get("output", {})
                    if isinstance(raw, dict):
                        doc_list = raw.get("documents", [])
                    elif isinstance(raw, list):
                        doc_list = raw
                    else:
                        doc_list = []
                    if doc_list:
                        yield _sse_line({"type": "documents", "data": [_doc_to_source(d) for d in doc_list]})

                elif etype == "on_llm_stream":
                    chunk = event["data"].get("chunk")
                    if chunk and hasattr(chunk, "content") and chunk.content:
                        token_buffer.append(chunk.content)
                        yield _sse_line({"type": "token", "data": chunk.content})

                elif etype == "on_chain_end" and ename == "rag":
                    final_result = event["data"].get("output", {}) or {}

        except Exception as e:
            yield _sse_line({"type": "error", "message": str(e)})

        # Update token metadata after streaming completes
        if final_result:
            try:
                multi_rag._update_tokens_metadata(answer_result=final_result, session_id=req.session_id)
            except Exception:
                pass
            final_answer = final_result.get("answer", "") or "".join(token_buffer)
            yield _sse_line({"type": "answer", "data": final_answer})

            # Persist if authenticated
            if user:
                try:
                    _auth.upsert_session(req.session_id, user["user_id"], "chat", title=req.question[:20])
                    _auth.save_message(req.session_id, "user", req.question)
                    import json as _json
                    extra = _json.dumps({
                        "rewritten_query": accumulated_rewrite,
                        "documents": accumulated_docs,
                    }, ensure_ascii=False)
                    _auth.save_message(req.session_id, "assistant", final_answer, extra_data=extra)
                except Exception:
                    pass

        yield _sse_line({"type": "done"})

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/api/eval", response_model=EvalResponse, summary="RAG评测")
async def evaluate(req: EvalRequest):
    """
    使用 RAGAS 评测 SimpleRAG 系统质量（耗时较长，约数分钟）。
    返回四项指标：answer_relevancy、faithfulness、context_recall、context_precision。
    """
    def _run():
        from datasets import load_dataset
        from MedicalRag.rag.RagEvaluate import RagasRagEvaluate  # lazy: ragas patches asyncio at import
        config = state["config"]
        dataset = load_dataset("json", data_files=req.eval_file)["train"]
        eval_llm = create_llm_client(config.llm)
        eval_embedding = create_embedding_client(config.embedding.summary_dense)
        evaluator = RagasRagEvaluate(
            rag_components=state["simple_rag"],
            eval_datasets=dataset,
            eval_llm=eval_llm,
            embedding=eval_embedding,
        )
        evaluator.do_sample(req.sample_size)
        result = evaluator.do_evaluate(req.query_field, req.reference_field)
        return result

    result = await run_sync(_run)
    metrics = {k: float(v) for k, v in result.to_pandas().mean().items() if k != "user_input"}
    return EvalResponse(sample_size=req.sample_size, metrics=metrics)


@app.post("/api/search-agent", response_model=SearchAgentResponse, summary="单轮智能检索Agent")
async def search_agent(req: SearchAgentRequest):
    """
    单轮智能检索 Agent（SearchGraph）。
    支持多步推理、事实核验、可选联网搜索，不保留对话历史。
    """
    def _run():
        agent: SearchGraph = state["search_agent"]
        if agent.search_graph is None:
            agent.build_search_graph()
        from langchain_core.messages import HumanMessage as _HM
        init_state = {
            "query": req.question,
            "main_messages": [_HM(content=req.question)],
            "other_messages": [],
            "docs": [],
            "summary": "",
            "retry": agent.config.agent.max_attempts,
            "final": "",
        }
        out_state = agent.search_graph.invoke(init_state)
        answer = out_state.get("final", "") or out_state.get("summary", "") or "（空）"
        docs = out_state.get("docs", [])
        return answer, docs

    answer, docs = await run_sync(_run)
    sources = [SourceDoc(**_doc_to_source(d)) for d in docs if hasattr(d, "page_content")]
    return SearchAgentResponse(question=req.question, answer=answer, documents=sources)


@app.post("/api/agent", response_model=AgentResponse, summary="多轮智能医疗Agent（阻塞）")
async def agent_endpoint(req: AgentRequest, request: Request):
    """
    多轮智能医疗 Agent（MedicalAgent）。
    支持主动追问、子查询拆分、并行检索、多轮对话历史。等待全部完成后返回。
    """
    agent = get_or_create_agent(req.session_id)
    result = await run_sync(agent.answer, req.question)
    ask_obj = result.get("ask_obj")
    needs_clarification = bool(ask_obj and ask_obj.need_ask)

    # Persist if answered (not clarification)
    if not needs_clarification:
        user = get_optional_user(request)
        if user:
            try:
                import json as _json
                final_answer = result.get("final_answer", "")
                sub_results = result.get("sub_query_results", [])
                agent_extra = _json.dumps({
                    "rewritten_query": result.get("rewritten_query", ""),
                    "sub_queries": [r.get("query", "") for r in sub_results if r.get("query")],
                    "documents": [
                        _doc_to_source(d)
                        for r in sub_results
                        for d in r.get("docs", [])
                        if hasattr(d, "page_content")
                    ],
                }, ensure_ascii=False)
                _auth.upsert_session(req.session_id, user["user_id"], "agent", title=req.question[:20])
                _auth.save_message(req.session_id, "user", req.question)
                _auth.save_message(req.session_id, "assistant", final_answer, extra_data=agent_extra)
            except Exception:
                pass

    return AgentResponse(
        session_id=req.session_id,
        question=req.question,
        needs_clarification=needs_clarification,
        clarification_questions=ask_obj.questions if needs_clarification else [],
        answer=None if needs_clarification else result.get("final_answer", ""),
    )


@app.post("/api/agent/stream", summary="多轮智能医疗Agent（SSE流式）")
async def agent_stream(req: AgentRequest, request: Request):
    """
    多轮智能医疗 Agent SSE 流式响应，使用 LangGraph 原生 astream。

    事件类型：
    - `progress` — 阶段提示
    - `background` — 用户背景信息（更新或提取后）
    - `rewrite`   — 改写/重写后的检索查询
    - `clarification` — Agent 需要追问（含 questions 字段），此后流结束
    - `sub_queries` — 拆分的子查询列表
    - `documents` — 某子查询检索到的文档
    - `answer` — 最终答案
    - `done` — 流结束
    """
    user = get_optional_user(request)

    async def event_generator():
        agent = get_or_create_agent(req.session_id)
        agent.state["curr_input"] = req.question
        agent.state["sub_query_results"] = []
        accumulated = dict(agent.state)
        final_answer = ""

        try:
            async for update_chunk in agent.app.astream(agent.state, stream_mode="updates"):
                for node_name, updates in update_chunk.items():
                    # Accumulate state (sub_query_results uses LangGraph add reducer)
                    for k, v in updates.items():
                        if k == "sub_query_results" and isinstance(v, list):
                            accumulated[k] = accumulated.get(k, []) + v
                        else:
                            accumulated[k] = v

                    # Emit SSE event based on the completed node
                    event: Optional[dict] = None

                    if node_name == "ask":
                        ask_obj = updates.get("ask_obj")
                        if ask_obj and ask_obj.need_ask:
                            event = {"type": "clarification", "questions": ask_obj.questions}
                        else:
                            event = {"type": "progress", "message": "正在分析问题，准备检索..."}

                    elif node_name == "extract_ask_and_reply":
                        bg = updates.get("background_info", "")
                        if bg:
                            yield _sse_line({"type": "background", "data": bg})
                        event = {"type": "progress", "message": "正在提取用户背景信息..."}

                    elif node_name == "check_update_background":
                        bg = updates.get("background_info", "")
                        if bg:
                            yield _sse_line({"type": "background", "data": bg})
                        event = {"type": "progress", "message": "正在更新用户背景信息..."}

                    elif node_name == "split_query":
                        sq = updates.get("sub_query")
                        if sq:
                            queries = sq.sub_query if sq.need_split else [sq.rewrite_query]
                            event = {"type": "sub_queries", "queries": [q for q in queries if q]}
                        # Also emit rewrite
                        rewritten = updates.get("rewritten_query", "")
                        if not rewritten and sq:
                            rewritten = sq.rewrite_query if not sq.need_split else (sq.sub_query[0] if sq.sub_query else "")
                        if rewritten:
                            yield _sse_line({"type": "rewrite", "data": rewritten})

                    elif node_name == "search_one":
                        for r in updates.get("sub_query_results", []):
                            docs = r.get("docs", [])
                            if docs:
                                yield _sse_line({"type": "documents", "data": [_doc_to_source(d) for d in docs]})

                    elif node_name == "answer":
                        final_answer = updates.get("final_answer", "")
                        event = {"type": "answer", "data": final_answer}

                    if event:
                        yield _sse_line(event)

        except Exception as e:
            yield _sse_line({"type": "error", "message": str(e)})

        agent.state = accumulated

        # Persist if we got a final answer and user is authenticated
        if final_answer and user:
            try:
                _auth.upsert_session(req.session_id, user["user_id"], "agent", title=req.question[:20])
                _auth.save_message(req.session_id, "user", req.question)
                import json as _json
                agent_extra = _json.dumps({
                    "rewritten_query": accumulated.get("rewritten_query", ""),
                    "sub_queries": [
                        q for r in accumulated.get("sub_query_results", [])
                        for q in ([r.get("query", "")] if r.get("query") else [])
                    ],
                    "documents": [
                        _doc_to_source(d)
                        for r in accumulated.get("sub_query_results", [])
                        for d in r.get("docs", [])
                        if hasattr(d, "page_content")
                    ],
                }, ensure_ascii=False)
                _auth.save_message(req.session_id, "assistant", final_answer, extra_data=agent_extra)
            except Exception:
                pass

        yield _sse_line({"type": "done"})

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Global error handler
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc), "type": type(exc).__name__},
    )
