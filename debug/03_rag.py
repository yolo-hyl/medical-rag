"""RAG 全链路验证（真模型）：需要 Milvus + Redis + debug/conf 里配的 vLLM 服务。

验证点：
1. 异步入库；
2. 混合检索命中是否符合语义；
3. 单轮 RAG，各阶段耗时不再是 0（StageTimer 在链外统计）；
4. batch_answer 的并发收益（对比串行）；
5. 多轮对话：会话状态进 Redis，第二轮的改写带上上一轮主题；
6. 换一个新实例能接上同一会话（状态不在进程内存里）；
7. 同一会话并发两轮不交错；
8. 流式：阶段事件 + token 级片段，流式后 token 统计仍写入 Redis。

耗时约 1 分钟（取决于模型速度）。
"""
import asyncio
import time

from common import RAW_RECORDS, build_kb, clear_session, dense_search_config, drop_collection, load_config

from MedicalRag.core.IngestionPipeline import IngestionPipeline
from MedicalRag.core.session_store import RedisSessionStore
from MedicalRag.rag.MultiDialogueRag import MultiDialogueRag
from MedicalRag.rag.SimpleRag import SimpleRAG

SID = "debug-rag"


def one_line(text: str, n: int = 120) -> str:
    return text[:n].replace("\n", " ")


async def main():
    cfg = load_config()
    print(f"集合={cfg.milvus.collection_name} LLM={cfg.llm.model} Embedding={cfg.embedding.text_dense.model}\n")

    kb = build_kb(cfg)
    search = dense_search_config(cfg)
    store = RedisSessionStore(cfg.redis)
    await clear_session(store, SID)

    try:
        # ---- 1) 入库 ----
        t0 = time.time()
        ok = await IngestionPipeline(cfg.data, kb).run(RAW_RECORDS)
        print(f"[1] 入库 {len(RAW_RECORDS)} 条: {ok}（{time.time()-t0:.1f}s）")

        # ---- 2) 混合检索 ----
        docs = await kb.search(search.model_copy(update={"query": "血压高了会不会头疼"}))
        print(f"[2] 检索「血压高了会不会头疼」→ {[d.metadata['summary'] for d in docs]}")

        # ---- 3) 单轮 RAG ----
        rag = SimpleRAG(cfg.llm, kb, search)
        r = await rag.answer("我肚子一阵一阵地痛，该怎么办？", return_document=True)
        print(f"[3] 单轮 RAG（检索 {r['search_time']:.2f}s / 生成 {r['generation_time']:.2f}s）")
        print(f"    {one_line(r['answer'], 160)}")

        # ---- 4) 并发 vs 串行 ----
        qs = ["高血压有什么症状", "感冒和流感区别", "糖尿病吃什么", "长期失眠怎么办"]
        t0 = time.time()
        await rag.batch_answer(qs)
        concur = time.time() - t0
        t0 = time.time()
        for q in qs:
            await rag.answer(q)
        serial = time.time() - t0
        print(f"[4] {len(qs)} 个问题：并发 {concur:.1f}s vs 串行 {serial:.1f}s（加速 {serial/concur:.1f}x）")

        # ---- 5) 多轮对话 ----
        def make_rag():
            return MultiDialogueRag(cfg.llm, cfg.multi_dialogue_rag, kb, store, search)

        mrag = make_rag()
        a1 = await mrag.answer("我最近总是头晕", session_id=SID)
        print(f"[5] 第 1 轮（改写 {a1['rewriten_generate_time']:.2f}s / 生成 {a1['out_generate_time']:.2f}s）")
        print(f"    {one_line(a1['answer'])}")

        rewritten = None
        async for ev in mrag.stream("那需要做什么检查？", session_id=SID):
            if ev["event"] == "on_chain_end" and ev["name"] == "rewritten_query":
                rewritten = ev["data"]["output"]["llm_rewritten_query"]["msg"]
        print(f"    第 2 轮的改写结果: {rewritten!r}")
        print(f"    改写用上了上一轮主题: {'晕' in (rewritten or '')}")

        hist = await store.history(f"medrag:chat:{SID}:history").aget_messages()
        print(f"    Redis 历史 {len(hist)} 条 {[m.content[:8] for m in hist]}")
        print(f"    token 统计: {await store.get_json(f'medrag:chat:{SID}:tokens')}")

        # ---- 6) 新实例接上同一会话 ----
        a3 = await make_rag().answer("我前面问的第一个问题是什么？", session_id=SID)
        print(f"[6] 新实例读 Redis 续接: {one_line(a3['answer'])}")

        # ---- 7) 同一会话并发不交错 ----
        await clear_session(store, SID)
        await asyncio.gather(
            mrag.answer("并发问题一", session_id=SID),
            make_rag().answer("并发问题二", session_id=SID),
        )
        hist = await store.history(f"medrag:chat:{SID}:history").aget_messages()
        print(f"[7] 并发两轮不交错: {[m.content[:8] for m in hist]}")

        # ---- 8) 流式 ----
        stages, rewrite_chunks, answer_chunks = [], [], []
        async for ev in mrag.stream("那平时饮食要注意什么？", session_id=SID):
            if ev["event"] == "on_chain_end" and ev["name"] in (
                    "rewritten_query", "search_documents", "generate", "rag"):
                stages.append(ev["name"])
            elif ev["event"] == "on_chat_model_stream":
                bucket = rewrite_chunks if ev["name"] == "rewrite_llm" else answer_chunks
                bucket.append(ev["data"]["chunk"].content)
        print(f"[8] 流式阶段事件: {stages}")
        print(f"    token 片段：改写 {len(rewrite_chunks)} 个，生成 {len(answer_chunks)} 个")
        print(f"    拼接结果（末 100 字）: {''.join(answer_chunks)[-100:]}".replace("\n", " "))
        tokens = await store.get_json(f"medrag:chat:{SID}:tokens")
        print(f"    流式后 token 统计写入正常（output_tokens 非 0）: "
              f"{all(t > 0 for t in tokens['msg_token_len'])} {tokens['msg_token_len']}")
    finally:
        await clear_session(store, SID)
        await store.close()
        drop_collection(kb)
        await kb.close()


if __name__ == "__main__":
    asyncio.run(main())
