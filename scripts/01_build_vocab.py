from typing import List, Dict
from datasets import load_dataset
from MedicalRag.embed.sparse import Vocabulary, BM25Vectorizer
import time


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