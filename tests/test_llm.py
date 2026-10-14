"""模型层：通过假 OpenAI 服务验证分批、顺序、参数透传、思考内容剥离、流式与 usage"""
import asyncio

from medrag.config import DenseConfig, LLMConfig
from medrag.llm import OpenAICompatEmbedding, OpenAICompatLLM, split_think
from medrag.testing.fake_openai import DEFAULT_REPLY, fake_embedding


def test_split_think():
    assert split_think("<think>先想想</think>\n\n答案") == ("先想想", "答案")
    assert split_think("没有思考") == ("", "没有思考")


def test_embedding_batches_keep_order(fake_openai):
    cfg = DenseConfig(model="fake-emb", base_url=fake_openai.base_url, dimension=64, max_batch=3, concurrency=2)
    texts = [f"第{i}段医学文本" for i in range(8)]

    async def run():
        emb = OpenAICompatEmbedding(cfg)
        try:
            return await emb(texts), await emb.embed_query("单条查询")
        finally:
            await emb.aclose()

    start = len(fake_openai.requests)
    vectors, query_vec = asyncio.run(run())
    batches = [body for path, body in fake_openai.requests[start:] if path.endswith("/embeddings")]
    assert [len(b["input"]) for b in batches[:-1]] == [3, 3, 2]
    assert all(b["dimensions"] == 64 for b in batches)
    for text, vec in zip(texts, vectors):
        assert max(abs(a - b) for a, b in zip(vec, fake_embedding(text, 64))) < 1e-6
    assert len(query_vec) == 64


def test_embedding_can_omit_dimensions(fake_openai):
    cfg = DenseConfig(model="fake-emb", base_url=fake_openai.base_url, dimension=32, send_dimensions=False)

    async def run():
        emb = OpenAICompatEmbedding(cfg)
        try:
            return await emb(["文本"])
        finally:
            await emb.aclose()

    start = len(fake_openai.requests)
    asyncio.run(run())
    body = fake_openai.requests[start][1]
    assert "dimensions" not in body


def test_llm_call_and_stream(fake_openai):
    cfg = LLMConfig(model="fake-llm", base_url=fake_openai.base_url, temperature=0.2,
                    extra_body={"enable_thinking": False})
    messages = [{"role": "user", "content": "高血压有哪些症状"}]

    async def run():
        llm = OpenAICompatLLM(cfg)
        try:
            result = await llm(messages)
            deltas = [d async for d in llm.stream(messages)]
            return result, deltas
        finally:
            await llm.aclose()

    start = len(fake_openai.requests)
    result, deltas = asyncio.run(run())
    assert result.text == DEFAULT_REPLY
    assert result.usage.completion_tokens > 0
    body = fake_openai.requests[start][1]
    assert body["temperature"] == 0.2 and body["enable_thinking"] is False
    assert "".join(d.text for d in deltas) == DEFAULT_REPLY
    assert deltas[-1].usage is not None
