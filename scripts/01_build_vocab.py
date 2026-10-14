"""构建自管理 BM25 的领域词表（需要安装 extras: zh）。词表默认保存在 medrag/retrieval/sparse/vocab_files/"""
import time

from datasets import load_dataset

from medrag.retrieval.sparse.vocab import BM25Vectorizer, Vocabulary


if __name__ == "__main__":
    vocab = Vocabulary()
    vectorizer = BM25Vectorizer(vocab, domain_model="medicine")

    qa_rows = load_dataset("json", data_files="data/qa_50000.jsonl", split="train")

    t0 = time.time()
    # # 并行分词 -> 建库/统计
    total_len = 0
    for toks in vectorizer.tokenize_parallel(qa_rows['text'], workers=8, chunksize=128):
        vocab.add_document(toks)
        total_len += len(toks)
    t1 = time.time()
    vocab.freeze()
    vocab.save("vocab.pkl.gz")
    print(f"词表构建及保存成功！")
    print(f"构建用时：{t1 - t0}")
    print(f"总token数量：{total_len}")