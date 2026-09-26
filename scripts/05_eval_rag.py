"""
RAG基础评测：生成答案（异步）→ 落盘 → 打分（同步）

答案落盘后可以在不重新生成的情况下重复打分，也便于和其他库的结果横向比较。
"""
import asyncio
import logging
import os

from langchain_community.embeddings import DashScopeEmbeddings
from langchain_openai import ChatOpenAI

from MedicalRag.config.loader import ConfigLoader
from MedicalRag.core.KnowledgeBase import MedicalHybridKnowledgeBase
from MedicalRag.eval import collect, from_medical_rag, load_samples, save_samples, score
from MedicalRag.rag.SimpleRag import SimpleRAG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

SAMPLE_PATH = "data/eval/medical_rag_samples.jsonl"


async def generate_answers():
    """用被测系统生成答案，写入 JSONL"""
    cfg = ConfigLoader().config
    kb = MedicalHybridKnowledgeBase(cfg.milvus, cfg.embedding)
    rag = SimpleRAG(cfg.llm, kb)
    # 固定 seed：每个被测库拿到完全相同的样本，分数才可比
    samples = load_samples(
        "data/eval/new_qa_200.jsonl",
        question_field="new_question",
        reference_field="answer",
        sample_size=10,  # 根据需要进行快速修改
    )
    try:
        samples = await collect(samples, from_medical_rag(rag))
    finally:
        await kb.close()
    save_samples(samples, SAMPLE_PATH)
    return samples


def main():
    # 1. 生成答案（异步）
    samples = asyncio.run(generate_answers())

    # 2. 打分（同步，在事件循环之外执行，避免和 ragas 自带的事件循环冲突）
    eval_llm = ChatOpenAI(
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model="qwen-plus",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        temperature=0.0,
        extra_body={"enable_thinking": False},
    )
    eval_embedding = DashScopeEmbeddings(
        model="text-embedding-v3",
        dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
    )
    print(score(samples, llm=eval_llm, embeddings=eval_embedding))


if __name__ == "__main__":
    main()
