"""假模型快速回归：需要 Milvus + Redis，不占 GPU，10 秒左右跑完。

Embedding 用真服务（便宜），LLM 换成假模型。改完包里的代码先跑这个，
它覆盖的是"链路接得对不对"，不关心回答质量：
    并发收益、会话状态进 Redis、新实例续接、并发不交错、流式、
    摘要压缩、Agent 跨轮状态。
"""
import asyncio
import time

from common import (
    RAW_RECORDS, STRUCTURED_REPLY, FakeLLM, build_kb, clear_session,
    dense_search_config, drop_collection, fake_websearch, load_config,
)

from MedicalRag.agent.MedicalAgent import MedicalAgent
from MedicalRag.core.IngestionPipeline import IngestionPipeline
from MedicalRag.core.session_store import RedisSessionStore
from MedicalRag.rag.MultiDialogueRag import MultiDialogueRag
from MedicalRag.rag.SimpleRag import SimpleRAG

SID = "debug-fake"
OTHER_SID = "debug-fake-other"


async def main():
    cfg = load_config()
    cfg.agent.mode = "fast"          # 假模型不做事实校验回路
    cfg.agent.max_attempts = 1

    kb = build_kb(cfg)
    search = dense_search_config(cfg, limit=5)
    store = RedisSessionStore(cfg.redis)
    await clear_session(store, SID, OTHER_SID)

    plain_llm = FakeLLM(messages=iter([]))
    # Agent 的 ask / split_query 节点要求结构化 JSON 输出
    json_llm = FakeLLM(messages=iter([]), reply=STRUCTURED_REPLY)

    try:
        print(f"[1] 入库: {await IngestionPipeline(cfg.data, kb).run(RAW_RECORDS)}")

        docs = await kb.search(search.model_copy(update={"query": "肚子一阵一阵地痛"}))
        print(f"[2] 检索命中 {len(docs)} 条，最相近: {docs[0].metadata['summary']}"
              f"（distance={docs[0].metadata['distance']:.4f}）")

        # ---- 3) 并发收益：假模型每次固定 0.2s ----
        rag = SimpleRAG(cfg.llm, kb, search)
        rag.llm = plain_llm
        rag._setup_chain()
        t0 = time.time()
        await rag.batch_answer([f"问题{c}" for c in "ABCDE"])
        print(f"[3] batch_answer 5 个问题并发 {time.time()-t0:.2f}s（串行至少 1.0s）")

        # ---- 4~6) 多轮：Redis 状态 / 新实例续接 / 并发不交错 ----
        def make_rag(llm_max_token=None):
            dialogue = cfg.multi_dialogue_rag
            if llm_max_token:
                dialogue = dialogue.model_copy(update={"llm_max_token": llm_max_token})
            m = MultiDialogueRag(cfg.llm, dialogue, kb, store, search)
            m.llm = plain_llm
            m._setup_chain()
            return m

        mrag = make_rag()
        await mrag.answer("我肚子痛", session_id=SID)
        await mrag.answer("需要忌口吗", session_id=SID)
        hist_key = f"medrag:chat:{SID}:history"
        n1 = len(await store.history(hist_key).aget_messages())
        print(f"[4] Redis 历史 {n1} 条，TTL {await store.r.ttl(hist_key)}s，"
              f"token 统计 {await store.get_json(f'medrag:chat:{SID}:tokens')}")

        await make_rag().answer("那能吃辣吗", session_id=SID)
        n2 = len(await store.history(hist_key).aget_messages())
        print(f"[5] 新实例接上历史: {n1} -> {n2} 条")

        await clear_session(store, SID)
        await asyncio.gather(
            mrag.answer("并发问题一", session_id=SID),
            make_rag().answer("并发问题二", session_id=SID),
        )
        print(f"[6] 并发两轮不交错: {[m.content for m in await store.history(hist_key).aget_messages()]}")

        # ---- 7) 流式 ----
        stages = [ev["name"] async for ev in mrag.stream("流式提问", session_id=SID)
                  if ev["event"] == "on_chain_end"
                  and ev["name"] in ("rewritten_query", "search_documents", "generate", "rag")]
        print(f"[7] 流式阶段事件: {stages}")

        # ---- 8) 摘要压缩：把 token 上限压到很低，强制触发 ----
        await clear_session(store, SID)
        squeezed = make_rag(llm_max_token=12)
        for i in range(3):
            await squeezed.answer(f"第{i}轮提问", session_id=SID)
        print(f"[8] 摘要已写入 Redis: {(await store.get_json(f'medrag:chat:{SID}:summary', '(无)'))!r}，"
              f"压缩后历史 {len(await store.history(hist_key).aget_messages())} 条")

        # ---- 9~10) Agent 跨轮状态 ----
        def make_agent():
            a = MedicalAgent(cfg.llm, cfg.agent, kb, store, power_model=json_llm)
            a.normal_llm = json_llm
            a.search_graph.llm = plain_llm   # 检索图里生成的是自然语言回答
            a.search_graph.agent_tools.register_websearch(fake_websearch)
            a.build_graph()
            return a

        await clear_session(store, SID)
        s1 = await make_agent().answer("我肚子痛", session_id=SID)
        saved = await store.get_json(f"medrag:agent:{SID}:state")
        print(f"[9] Agent 第一轮 final_answer={s1['final_answer']!r}，"
              f"Redis 跨轮字段 {sorted(saved.keys())}")

        s2 = await make_agent().answer("那能吃辣吗", session_id=SID)
        print(f"[10] 新实例续接: 对话消息 {len(s1['dialogue_messages'])} -> {len(s2['dialogue_messages'])}，"
              f"单轮 performance {len(s2['performance'])} 条（每轮重置）")

        nodes = [n async for chunk in make_agent().stream("再问一个", session_id=SID) for n in chunk]
        after = await store.get_json(f"medrag:agent:{SID}:state")
        print(f"[11] Agent 流式节点 {nodes}，流式后对话消息 {len(after['dialogue_messages'])} 条")
    finally:
        await clear_session(store, SID, OTHER_SID)
        await store.close()
        drop_collection(kb)
        await kb.close()


if __name__ == "__main__":
    asyncio.run(main())
