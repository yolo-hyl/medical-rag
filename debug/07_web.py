"""Agent 轨迹查看页（真模型，不用 mock）。

启动：
    python debug/07_web.py
然后打开 http://127.0.0.1:8100

页面把 MedicalAgent.stream() 产出的 {节点名: 更新} 按顺序显示出来，
用的是 debug/conf 里配的本地 vLLM 与 medrag_debug 集合：集合已存在就直接用，
不存在则新建并写入内置 8 条样例；退出时不删集合。
"""
import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from common import (
    RAW_RECORDS, apply_websearch_policy, build_kb, clear_session, load_config,
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
    # 集合名取 storage.yaml，可用 MEDRAG_WEB_COLLECTION 临时换一个
    name = os.environ.get("MEDRAG_WEB_COLLECTION")
    if name:
        cfg = cfg.model_copy(update={"milvus": cfg.milvus.model_copy(update={"collection_name": name})})
    name = cfg.milvus.collection_name
    # 没有腾讯云凭据就关掉联网检索，只用本地知识库；不拿假数据顶
    agent_cfg, note = apply_websearch_policy(cfg.agent)
    kb = build_kb(cfg)
    # 集合已存在就直接用，不写样例；不存在才建集合并写入内置 8 条。退出时一律不删
    if kb.client.has_collection(name):
        kb.client.load_collection(name)
        rows = kb.client.query(name, output_fields=["count(*)"])[0]["count(*)"]
    else:
        if not await IngestionPipeline(cfg.data, kb).run(RAW_RECORDS):
            raise RuntimeError("入库失败，看上面的日志")
        rows = len(RAW_RECORDS)
    store = RedisSessionStore(cfg.redis)
    agent = MedicalAgent(cfg.llm, agent_cfg, kb, store, power_model=create_llm_client(cfg.llm))
    ctx.update(cfg=cfg, kb=kb, store=store, agent=agent)
    print(f"\n就绪：http://127.0.0.1:8100"
          f"\n  模型 {cfg.llm.model} / 模式 {agent_cfg.mode} / 知识库 {cfg.milvus.collection_name}（{rows} 条）"
          f"\n  {note}\n")
    yield
    await store.close()
    await kb.close()


app = FastAPI(title="Agent 轨迹", lifespan=lifespan)


def clip(text: str, n: int) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= n else text[:n] + "…"


def doc_lines(docs) -> list[str]:
    """每篇文档两行：标题（入库时的问题）+ 分数，正文开头"""
    lines = []
    for i, d in enumerate(docs, 1):
        meta = d.metadata
        score = meta.get("distance")
        score = f"（分数 {score:.3f}）" if isinstance(score, (int, float)) and score != 99999 else ""
        lines.append(f"  [{i}] {clip(meta.get('summary') or meta.get('source_name') or '', 60)}{score}")
        lines.append(f"      {clip(d.page_content, 150)}")
    return lines


def describe(node: str, updates: dict) -> list[str]:
    """把一个节点的状态更新翻译成几行人话"""
    if node == "compress_memory":
        return [f"短期记忆 {len(updates.get('dialogue_messages') or [])} 条消息，"
                f"长期摘要 {len(updates.get('summary') or [])} 段"]

    if node == "ask":
        ask = updates.get("ask_messages")
        if ask is None:
            return ["判断完成"]
        if ask.ask_decision.ask_signal:
            return [f"需要追问（第 {ask.curr_ask_num} 次）"] + \
                   [f"· {q}" for q in ask.ask_decision.questions]
        # 追问满 max_ask_num 次后，ask_judge 不再调 LLM，curr_ask_num 会变成 max_ask_num + 1
        max_ask = ctx["agent"].agent_config.max_ask_num
        if ask.curr_ask_num > max_ask:
            return [f"已达最大追问次数（{max_ask}），不再追问"]
        return ["信息已充分，不再追问"]

    if node == "follow_ask":
        ask = updates.get("ask_messages")
        if ask is not None and ask.curr_ask_num > 0:
            return ["需要补充背景（本问题仅此一次）"] + [f"· {q}" for q in ask.ask_decision.questions]
        return ["背景已充分，不追问"]

    if node in ("extract_ask_and_reply", "check_update_background", "follow_reply"):
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

    # ---- 以下是每个子查询内部检索图的节点，updates 是完整的 SearchMessagesState ----
    if node in ("db_search", "web_search"):
        head = [f"子查询：{updates.get('query', '')}"]
        if node == "web_search":
            judged = [m.content for m in updates.get("other_messages", [])
                      if m.content.startswith(("分析结果", "解析失败"))]
            if judged:
                head.append(f"· 联网判断：{judged[-1]}")
        docs = updates.get("docs") or []
        return head + [f"· 共 {len(docs)} 篇文档"] + doc_lines(docs)

    if node == "rag":
        attempt = "（重新生成）" if updates.get("judge_result") == "retry" else ""
        return [f"子查询：{updates.get('query', '')}",
                f"· 生成{attempt}：{(updates.get('summary') or '(空)').strip()}"]

    if node == "judge":
        verdict = {"pass": "通过", "retry": "不通过，重新生成", "fail": "不通过，重试用完"}
        raw = next((m.content for m in reversed(updates.get("other_messages", []))
                    if m.content.startswith("[JUDGE]=")), "")
        return [f"子查询：{updates.get('query', '')}",
                f"· 评估：{verdict.get(updates.get('judge_result'), updates.get('judge_result'))}"
                f"（模型输出 {raw.removeprefix('[JUDGE]=')[:20] or '(空)'}，剩余重试 {updates.get('retry')}）"]

    if node in ("finish_success", "finish_fail"):
        return [f"子查询：{updates.get('query', '')}", f"· 最终：{updates.get('final', '')}"]

    if node == "search_one":
        return [f"子查询完成：{r.get('query', '')}（{len(r.get('docs', []))} 篇文档）"
                for r in updates.get("retrieval_outputs", [])] or ["没有子查询结果"]

    if node == "answer":
        return [updates.get("final_answer") or "(空)"]

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
