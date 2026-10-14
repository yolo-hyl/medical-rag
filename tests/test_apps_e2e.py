"""端到端冒烟：真实 Milvus + Redis，模型走本地假 OpenAI 服务

覆盖 M0 迁移后的完整链路：入库 → 混合检索 → SimpleRAG → 多轮 RAG（含流式）→ 检索图（工具调用）→ MedicalAgent 多轮。
假服务的回答是固定的，这里只验证链路通、数据流对，不评价回答质量。
"""
import asyncio
import uuid

import pytest

from medrag.apps.agent import MedicalAgent, SearchGraph
from medrag.apps.dialogue_rag import MultiDialogueRag
from medrag.apps.rag_base import default_search_config
from medrag.apps.simple_rag import SimpleRAG
from medrag.config import (
    AgentConfig, DenseConfig, EmbeddingConfig, LLMConfig, MilvusConfig, MultiDialogueRagConfig, SparseConfig,
    load_deploy_env,
)
from medrag.config.models import DataConfig, RedisConfig
from medrag.ingest import IngestionPipeline
from medrag.llm.langchain_adapter import create_chat_model
from medrag.storage.milvus import MedicalHybridKnowledgeBase
from medrag.storage.redis_session import RedisSessionStore
from medrag.testing.fake_openai import DEFAULT_REPLY

pytestmark = pytest.mark.services

QA = [
    ("肚子一阵一阵地痛怎么办", "阵发性腹痛常见于肠痉挛、肠梗阻早期，建议禁食观察、腹部保暖，持续不缓解需就医。"),
    ("高血压有哪些常见症状", "高血压常见头痛、头晕、心悸、耳鸣，多数早期无明显症状，需定期测量血压。"),
    ("感冒和流感怎么区分", "流感起病急、高热明显、全身酸痛重；普通感冒以鼻塞流涕为主，发热轻。"),
    ("吃完饭就想吐是什么原因", "餐后恶心呕吐常见于胃炎、胃排空延迟、胆囊疾病，需结合腹痛部位判断。"),
    ("糖尿病人饮食要注意什么", "控制总热量，主食粗细搭配，定时定量，限制精制糖和饱和脂肪，配合监测血糖。"),
    ("小孩发烧到几度需要吃退烧药", "腋温超过38.5摄氏度或明显不适可考虑退热药，重点是补液观察精神状态。"),
]


@pytest.fixture(scope="module")
def env(fake_openai):
    load_deploy_env()
    dense = DenseConfig(model="fake-emb", base_url=fake_openai.base_url, dimension=64)
    embedding = EmbeddingConfig(summary_dense=dense, text_dense=dense, text_sparse=SparseConfig(provider="Milvus"))
    milvus = MilvusConfig(collection_name=f"medrag_pytest_{uuid.uuid4().hex[:8]}", drop_old=True, auto_id=False)
    llm = LLMConfig(model="fake-llm", base_url=fake_openai.base_url)
    return {"embedding": embedding, "milvus": milvus, "llm": llm}


def run(coro_fn, env):
    """每个用例在自己的事件循环里新建知识库与会话存储（连接绑定在事件循环上）"""
    async def main():
        kb = MedicalHybridKnowledgeBase(env["milvus"], env["embedding"])
        store = RedisSessionStore(RedisConfig())
        try:
            return await coro_fn(kb, store)
        finally:
            await kb.close()
            await store.close()
    return asyncio.run(main())


def test_01_ingest_and_search(env):
    async def case(kb, store):
        ok = await IngestionPipeline(DataConfig(), kb).run([{"question": q, "answer": a} for q, a in QA])
        assert ok
        req = default_search_config(kb).model_copy(update={"query": "高血压常见症状有哪些"})
        return await kb.search(req)

    docs = run(case, env)
    assert docs and docs[0].metadata["summary"] == "高血压有哪些常见症状"


def test_02_simple_rag(env):
    async def case(kb, store):
        return await SimpleRAG(env["llm"], kb).answer("感冒和流感怎么区分", return_document=True)

    result = run(case, env)
    assert result["answer"] == DEFAULT_REPLY
    assert result["documents"]


def test_03_dialogue_rag_answer_and_stream(env):
    sid = f"pytest-{uuid.uuid4().hex[:8]}"

    async def case(kb, store):
        rag = MultiDialogueRag(env["llm"], MultiDialogueRagConfig(), kb, store)
        first = await rag.answer("小孩发烧怎么办", return_document=True, session_id=sid)
        events = [e async for e in rag.stream("那几度要吃药", session_id=sid)]
        history = await store.list_get(f"medrag:chat:{sid}:history")
        await store.delete(*(f"medrag:chat:{sid}:{k}" for k in ("history", "summary", "tokens")))
        return first, events, history

    first, events, history = run(case, env)
    assert first["answer"] == DEFAULT_REPLY and first["documents"]
    assert any(e["event"] == "on_chat_model_stream" for e in events)
    assert len(history) == 4   # 两轮，每轮一问一答


def test_04_search_graph_calls_db_tool(env):
    async def case(kb, store):
        agent_cfg = AgentConfig(mode="analysis", network_search_enabled=False, max_attempts=1)
        graph = SearchGraph(env["llm"], agent_cfg, kb, power_model=create_chat_model(env["llm"]))
        return await graph.run(graph.init_state("吃完饭就想吐"))

    state = run(case, env)
    assert state["docs"], "database_search 工具应当取到本地资料"
    assert state["final"] == DEFAULT_REPLY


def test_05_medical_agent_two_rounds(env):
    sid = f"pytest-{uuid.uuid4().hex[:8]}"

    async def case(kb, store):
        agent_cfg = AgentConfig(mode="fast", network_search_enabled=False, max_attempts=1)
        agent = MedicalAgent(env["llm"], agent_cfg, kb, store, power_model=create_chat_model(env["llm"]))
        first = await agent.answer("我这两天肚子痛，还拉肚子", session_id=sid)
        second = await agent.answer("需要吃什么药吗", session_id=sid)
        await store.delete(f"medrag:agent:{sid}:state")
        return first, second

    first, second = run(case, env)
    assert first["final_answer"] == DEFAULT_REPLY
    assert second["final_answer"] == DEFAULT_REPLY
    assert len(second["dialogue_messages"]) == 4


def test_99_cleanup(env):
    from pymilvus import MilvusClient
    client = MilvusClient(uri=env["milvus"].uri)
    client.drop_collection(env["milvus"].collection_name)
    client.close()
