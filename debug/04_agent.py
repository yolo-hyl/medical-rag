"""Agent 验证（真模型）：需要 Milvus + Redis + vLLM。

验证点：
1. SearchGraph 单轮跑通（会真的触发 database_search 工具调用）；
2. MedicalAgent 无状态：跨轮字段进 Redis，单轮字段不落盘；
3. 换一个新实例能接上同一会话（追问不会重来）；
4. 不同会话互不影响；
5. 流式透传 LangGraph 的 {节点名: 更新}。

一轮 Agent 在 27B 模型上要 1~3 分钟（会拆子查询、并发检索、校验事实），
所以整个脚本可能跑 5 分钟以上。用 MEDRAG_AGENT_FAST=1 可以把 mode 降成
fast、跳过事实校验回路，快一些。
"""
import asyncio
import os
import time

from common import (
    RAW_RECORDS, apply_websearch_policy, build_kb, clear_session,
    dense_search_config, drop_collection, load_config,
)

from MedicalRag.agent.MedicalAgent import MedicalAgent
from MedicalRag.agent.SearchGraph import SearchGraph
from MedicalRag.core.IngestionPipeline import IngestionPipeline
from MedicalRag.core.session_store import RedisSessionStore
from MedicalRag.core.utils import create_llm_client

SID = "debug-agent"
OTHER_SID = "debug-agent-other"

# 依次喂给 Agent 的用户输入：第一条提问，后面几条回答它的追问。
# 追问次数上限是 AgentConfig.max_ask_num（默认 3），
# 给 5 条留足余量，保证一定能走到最终回答。
REPLIES = [
    "我这两天肚子痛，还拉肚子",
    "大便没有血也不黑，体温 37.8 度，痛在肚子中间偏下",
    "小便次数正常，没吃过药，昨天吃了外面的凉菜",
    "不是水样，一天三四次，还能正常喝水吃饭",
    "没有呕吐，也没有怀孕可能，没有其他症状了",
]


def one_line(text: str, n: int = 200) -> str:
    return (text or "")[:n].replace("\n", " ")


def make_agent(cfg, agent_cfg, kb, store, power_model):
    """Agent 本身无状态，可以随时新建一个实例，会话状态都在 Redis 里"""
    return MedicalAgent(cfg.llm, agent_cfg, kb, store, power_model=power_model)


async def main():
    cfg = load_config()
    if os.environ.get("MEDRAG_AGENT_FAST"):
        cfg.agent.mode = "fast"
    cfg.agent.max_attempts = 1
    # 没有腾讯云凭据就关掉联网检索，只用本地知识库；不拿假数据顶
    agent_cfg, note = apply_websearch_policy(cfg.agent)
    print(f"集合={cfg.milvus.collection_name} LLM={cfg.llm.model} mode={agent_cfg.mode}")
    print(f"{note}\n")

    kb = build_kb(cfg)
    store = RedisSessionStore(cfg.redis)
    await clear_session(store, SID, OTHER_SID)
    power_model = create_llm_client(cfg.llm)

    try:
        # IngestionPipeline.run 会吞掉异常只返回 False，这里必须检查，
        # 否则后面的检索全是空的，看起来像 Agent 的问题
        if not await IngestionPipeline(cfg.data, kb).run(RAW_RECORDS):
            print("入库失败，先看上面的日志；如果是 index not found，"
                  "通常是上一次运行没清理干净、drop 与 create 撞上了，重跑一次即可")
            return

        # ---- 1) SearchGraph 单轮 ----
        # 关掉联网检索单独看本地库：开着的话 llm_network_search 在 remain_doc_index
        # 为空时会清掉本地文档，只剩联网结果（这里联网是桩件，看不出检索效果）
        db_only = agent_cfg.model_copy(update={"network_search_enabled": False})
        graph = SearchGraph(cfg.llm, db_only, kb, power_model)
        t0 = time.time()
        out_state = await graph.run(graph.init_state("头晕目眩可能是什么病？"))
        docs = out_state.get("docs", [])
        print(f"[1] SearchGraph（{time.time()-t0:.0f}s）database_search 取到 {len(docs)} 条"
              f"{[d.metadata.get('summary') for d in docs[:3]]}")
        print(f"      {one_line(out_state.get('final') or out_state.get('summary'))}")

        # ---- 2/3) MedicalAgent 多轮，中途换实例 ----
        agent = make_agent(cfg, agent_cfg, kb, store, power_model)
        state = None
        for i, text in enumerate(REPLIES, 1):
            # 第 2 轮换一个新实例，验证状态确实在 Redis 而不在进程内存
            worker = make_agent(cfg, agent_cfg, kb, store, power_model) if i == 2 else agent
            t0 = time.time()
            state = await worker.answer(text, session_id=SID)
            ask = state["ask_messages"]
            tag = "（新实例）" if i == 2 else ""
            if ask.ask_decision.ask_signal:
                print(f"[2.{i}] 追问中{tag}（{time.time()-t0:.0f}s，curr_ask_num={ask.curr_ask_num}）")
                print(f"      {one_line(ask.asked_messages[-1][-1].content, 150)}")
            else:
                print(f"[2.{i}] 已回答{tag}（{time.time()-t0:.0f}s）")
                print(f"      {one_line(state.get('final_answer'), 300)}")
                break

        saved = await store.get_json(f"medrag:agent:{SID}:state")
        print(f"[3] Redis 只存跨轮字段: {sorted(saved.keys())}")
        print(f"    对话消息 {len(saved['dialogue_messages'])} 条，"
              f"summary {len(saved['summary'])} 条，"
              f"检索轮次 {len(saved['retrieval_states'])} 轮，"
              f"curr_ask_num={saved['ask_messages']['curr_ask_num']}")
        print(f"    background_info={saved['background_info'][:80]!r}")
        print(f"    单次执行字段未落盘: {'performance' not in saved and 'final_answer' not in saved}")

        # ---- 4) 不同会话互不影响 ----
        other = await agent.answer("糖尿病饮食要注意什么", session_id=OTHER_SID)
        other_saved = await store.get_json(f"medrag:agent:{OTHER_SID}:state")
        print(f"[4] 另一个会话独立: 对话消息 {len(other_saved['dialogue_messages'])} 条，"
              f"curr_ask_num={other_saved['ask_messages']['curr_ask_num']}"
              f"（{SID} 仍是 {saved['ask_messages']['curr_ask_num']}）")

        # ---- 5) 流式 ----
        nodes = []
        async for chunk in agent.stream("那饮食上要注意什么？", session_id=SID):
            nodes.extend(chunk.keys())
        after = await store.get_json(f"medrag:agent:{SID}:state")
        print(f"[5] 流式节点: {nodes}")
        print(f"    流式后 Redis 对话消息: {len(after['dialogue_messages'])} 条")
    finally:
        await clear_session(store, SID, OTHER_SID)
        await store.close()
        drop_collection(kb)
        await kb.close()


if __name__ == "__main__":
    asyncio.run(main())
