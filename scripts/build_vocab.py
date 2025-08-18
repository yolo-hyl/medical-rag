from typing import List, Dict
from datasets import load_dataset
from MedicalRag.data.processor.sparse import Vocabulary, BM25Vectorizer
import time

if __name__ == "__main__":
    vocab = Vocabulary()
    vectorizer = BM25Vectorizer(vocab, domain_model="medicine")

    rows = load_dataset("json", data_files="/home/weihua/medical-rag/raw_data/tf-data/qa/*.json", split="train")
    docs = [f"{x['question']}\n\n{x['answer']}" for x in rows]

    t0 = time.time()
    # # 并行分词 -> 建库/统计
    total_len = 0
    for toks in vectorizer.tokenize_parallel(docs, workers=8, chunksize=128):
        vocab.add_document(toks)
        total_len += len(toks)
    vocab.freeze()
    vocab.save("vocab.pkl.gz")
