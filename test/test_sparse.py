import unittest
import math
from typing import List, Dict

from MedicalRag.data.processor.sparse import Vocabulary, BM25Vectorizer

if __name__ == "__main__":
    import time, json
    # vocab = Vocabulary()
    # vectorizer = BM25Vectorizer(vocab, domain_model="medicine")

    with open('raw_data/tf-data/qa/qa.temp_4070.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    docs = [f"{x['question']}？{x['answer']}" for x in data]

    t0 = time.time()
    # # Pass 1: 并行分词 -> 建库/统计
    total_len = 0
    vocab = Vocabulary.load("vocab.pkl.gz")
    vectorizer = BM25Vectorizer(vocab, domain_model="medicine")
    for toks in vectorizer.tokenize_parallel(docs, workers=8, chunksize=128):
        # vocab.add_document(toks)
        total_len += len(toks)
    # vocab.freeze()
    # vocab.save("vocab.pkl.gz")
    
    
    avgdl = total_len / max(1, vocab.N)

    # Pass 2: 再分词一次（并行）-> 向量化（不更新词表）
    vectors = []
    for toks in vectorizer.tokenize_parallel(docs, workers=8, chunksize=128):
        vec = vectorizer.build_sparse_vec_from_tokens(toks, avgdl, update_vocab=False)
        vectors.append(vec)

    print(vectors[0])
    print(f"elapsed: {time.time()-t0:.2f}s")
