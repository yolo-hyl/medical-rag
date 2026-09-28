"""Agent 轨迹查看页（真模型，不用 mock）。

启动：
    python debug/07_web.py
然后打开 http://127.0.0.1:8100

页面把 MedicalAgent.stream() 产出的 {节点名: 更新} 按顺序显示出来，
用的是 debug/conf 里配的本地 vLLM 与 medrag_debug 集合。
"""
import json
import time
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from common import (
    RAW_RECORDS, apply_websearch_policy, build_kb, clear_session, drop_collection, load_config,
)
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from MedicalRag.agent.MedicalAgent import MedicalAgent
from MedicalRag.core.IngestionPipeline import IngestionPipeline
from MedicalRag.core.session_store import RedisSessionStore
from MedicalRag.core.utils import create_llm_client

PAGE = Path(__file__).parent / "web.html"
ctx = {}


class Ask(BaseModel):
    session_id: str = "web"
    message: str


@asynccontextmanager
async def lifespan(_: FastAPI):
    cfg = load_config()
    # 没有腾讯云凭据就关掉联网检索，只用本地知识库；不拿假数据顶
    agent_cfg, note = apply_websearch_policy(cfg.agent)
    kb = build_kb(cfg)
    if not await IngestionPipeline(cfg.data, kb).run(RAW_RECORDS):
        raise RuntimeError("入库失败，看上面的日志")
    store = RedisSessionStore(cfg.redis)
    agent = MedicalAgent(cfg.llm, agent_cfg, kb, store, power_model=create_llm_client(cfg.llm))
    ctx.update(cfg=cfg, kb=kb, store=store, agent=agent)
    print(f"\n就绪：http://127.0.0.1:8100"
          f"\n  模型 {cfg.llm.model} / 模式 {agent_cfg.mode} / 知识库 {cfg.milvus.collection_name}（{len(RAW_RECORDS)} 条）"
          f"\n  {note}\n")
    yield
    await store.close()
    drop_collection(kb)
    await kb.close()


app = FastAPI(title="Agent 轨迹", lifespan=lifespan)


def describe(node: str, updates: dict) -> list[str]:
    """把一个节点的状态更新翻译成几行人话"""
    if node == "ask":
        ask = updates.get("ask_messages")
        if ask is None:
            return ["判断完成"]
        if ask.ask_decision.ask_signal:
            return [f"需要追问（第 {ask.curr_ask_num} 次）"] + \
                   [f"· {q}" for q in ask.ask_decision.questions]
        return ["信息已充分，不再追问"]

    if node in ("extract_ask_and_reply", "check_update_background"):
        return [f"用户背景：{updates.get('background_info') or '(空)'}"]

    if node == "split_query":
        states = updates.get("retrieval_states") or []
        if not states:
            return ["未产出改写结果"]
        queries = states[-1].rewritten_queries.rewritten_queries
        if len(queries) > 1:
            return [f"拆成 {len(queries)} 个子查询"] + [f"· {q}" for q in queries]
        return [f"不拆分，改写为：{queries[0] if queries else '(空)'}"]

    if node == "retrieve":
        states = updates.get("retrieval_states") or []
        n = len(states[-1].retrieval_outputs) if states else 0
        return [f"检索完成，汇合 {n} 个子查询结果"]

    if node == "search_one":
        lines = []
        for r in updates.get("retrieval_outputs", []):
            lines.append(f"子查询：{r.get('query', '')}")
            lines.append(f"· 检索到 {len(r.get('docs', []))} 篇文档，剩余重试 {r.get('retry')}")
            summary = (r.get("final") or r.get("summary") or "").strip()
            if summary:
                lines.append(f"· 小结：{summary}")
        return lines or ["没有子查询结果"]

    if node == "answer":
        lines = [updates.get("final_answer") or "(空)"]
        if updates.get("summary"):
            lines.append(f"（会话已累积 {len(updates['summary'])} 条摘要）")
        return lines

    return [json.dumps(list(updates.keys()), ensure_ascii=False)]


def sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@app.post("/chat")
async def chat(ask: Ask):
    async def gen():
        t0 = time.time()
        try:
            async for chunk in ctx["agent"].stream(ask.message, session_id=ask.session_id):
                for node, updates in chunk.items():
                    yield sse({
                        "node": node,
                        "elapsed": round(time.time() - t0, 1),
                        "lines": describe(node, updates),
                    })
        except Exception as e:
            yield sse({"node": "错误", "elapsed": round(time.time() - t0, 1), "lines": [repr(e)]})
        yield sse({"done": True, "elapsed": round(time.time() - t0, 1)})

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/reset")
async def reset(ask: Ask):
    """清掉该会话在 Redis 里的状态，从头开始"""
    await clear_session(ctx["store"], ask.session_id)
    return {"ok": True}


@app.get("/")
async def index():
    return HTMLResponse(PAGE.read_text(encoding="utf-8"))


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8100, log_level="warning")
