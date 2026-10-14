"""
RAG基础评测：生成答案（异步）→ 落盘 → 打分（同步）

答案落盘后可以在不重新生成的情况下重复打分，也便于和其他库的结果横向比较。
"""
import asyncio
import logging

from _common import load_config
from medrag.apps.simple_rag import SimpleRAG
from medrag.config import DenseConfig, LLMConfig
from medrag.eval import collect, from_medical_rag, load_samples, save_samples
from medrag.llm.langchain_adapter import create_chat_model, create_embeddings
from medrag.storage.milvus import MedicalHybridKnowledgeBase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

SAMPLE_PATH = "data/eval/medical_rag_samples.jsonl"


async def generate_answers():
    """用被测系统生成答案，写入 JSONL"""
    cfg = load_config()
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
    #    评测模型要固定，不同轮次的分数才可比；ragas 会在 import 时给 asyncio 打补丁，所以在这里才导入
    from medrag.eval.scorers.ragas import score

    dashscope = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    eval_llm = create_chat_model(LLMConfig(
        model="qwen-plus", base_url=dashscope, api_key_env="DASHSCOPE_API_KEY",
        temperature=0.0, extra_body={"enable_thinking": False},
    ))
    eval_embedding = create_embeddings(DenseConfig(
        model="text-embedding-v3", base_url=dashscope, api_key_env="DASHSCOPE_API_KEY", dimension=1024,
    ))
    print(score(samples, llm=eval_llm, embeddings=eval_embedding))


if __name__ == "__main__":
    main()
