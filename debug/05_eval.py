"""统一评测流程验证（真模型）：需要 Milvus + vLLM，不需要 Redis。

验证点：
1. 固定 seed 采样可复现；
2. collect 逐条生成答案，单条失败记录错误后继续；
3. JSONL 落盘与读回一致；
4. score 打分跑通；
5. 对同一份 JSONL 重复打分（不重新生成答案）。

data/eval/new_qa_200.jsonl 是 Git LFS 指针（未拉取），所以这里用 common.QA
临时写一份数据集。拉下来真数据后把 DATASET 换成那个路径即可。

耗时约 4~6 分钟，主要花在 ragas 打分上（12 次指标计算，每次都要调 LLM）。
"""
import asyncio
import json
import os
import tempfile

from common import QA, build_kb, dense_search_config, drop_collection, load_config, RAW_RECORDS

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from MedicalRag.core.IngestionPipeline import IngestionPipeline
from MedicalRag.eval import collect, from_medical_rag, load_samples, load_saved, save_samples, score
from MedicalRag.rag.SimpleRag import SimpleRAG

OUT_DIR = tempfile.mkdtemp(prefix="medrag_eval_")
DATASET = os.path.join(OUT_DIR, "eval_qa.jsonl")
SAMPLES = os.path.join(OUT_DIR, "samples.jsonl")
SAMPLE_SIZE = 3


def write_dataset():
    with open(DATASET, "w", encoding="utf-8") as f:
        for q, a in QA:
            f.write(json.dumps({"new_question": q, "answer": a}, ensure_ascii=False) + "\n")


async def generate_answers(cfg):
    """入库 + 用被测系统生成答案"""
    kb = build_kb(cfg)
    try:
        if not await IngestionPipeline(cfg.data, kb).run(RAW_RECORDS):
            raise RuntimeError("入库失败，先看上面的日志")
        rag = SimpleRAG(cfg.llm, kb, dense_search_config(cfg))
        samples = load_samples(DATASET, "new_question", "answer", sample_size=SAMPLE_SIZE)
        print(f"2) 采样 {len(samples)} 条: {[s.question for s in samples]}")
        return await collect(samples, from_medical_rag(rag))
    finally:
        drop_collection(kb)
        await kb.close()


def main():
    cfg = load_config()
    # 用独立集合，避免和 03/04 的脚本同时跑时互相 drop
    cfg.milvus.collection_name = "medrag_debug_eval"
    write_dataset()

    # 1) 同 seed 可复现
    a = load_samples(DATASET, "new_question", "answer", sample_size=SAMPLE_SIZE)
    b = load_samples(DATASET, "new_question", "answer", sample_size=SAMPLE_SIZE)
    c = load_samples(DATASET, "new_question", "answer", sample_size=SAMPLE_SIZE, seed=7)
    print(f"1) 同 seed 采样一致: {[s.question for s in a] == [s.question for s in b]}，"
          f"换 seed 会不同: {[s.question for s in a] != [s.question for s in c]}")

    # 2/3) 生成答案（异步）并落盘
    samples = asyncio.run(generate_answers(cfg))
    save_samples(samples, SAMPLES)
    print(f"3) JSONL 往返一致: {load_saved(SAMPLES) == samples}（{SAMPLES}）")
    for s in samples:
        flag = f" [失败: {s.error}]" if s.error else ""
        print(f"   问: {s.question} | 上下文 {len(s.contexts)} 条{flag}")
        print(f"     答: {s.answer[:70]}".replace("\n", " "))

    # 4/5) 打分：同步函数，必须在事件循环之外调用，否则和 ragas 自带的循环冲突
    eval_llm = ChatOpenAI(
        model=cfg.llm.model, base_url=cfg.llm.base_url,
        api_key=os.environ["VLLM_API_KEY"], temperature=0.0,
    )
    eval_emb = OpenAIEmbeddings(
        model=cfg.embedding.text_dense.model, base_url=cfg.embedding.text_dense.base_url,
        api_key=os.environ["VLLM_API_KEY"], check_embedding_ctx_length=False,
    )
    eval_emb.dimensions = None   # 同 common.build_kb 的原因

    print("4) 第一次打分:", score(samples, llm=eval_llm, embeddings=eval_emb))
    print("5) 对同一份 JSONL 重复打分:", score(load_saved(SAMPLES), llm=eval_llm, embeddings=eval_emb))
    print("   注：即使 temperature=0，answer_relevancy / faithfulness 仍会小幅浮动"
          "（这两个指标要靠 LLM 生成中间产物）；检索类指标是稳定的。")


if __name__ == "__main__":
    main()
