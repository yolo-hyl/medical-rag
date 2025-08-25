from typing import List, Dict
from datasets import load_dataset
from MedicalRag.knowledge.sparse import Vocabulary, BM25Vectorizer
import time
from datasets import concatenate_datasets


if __name__ == "__main__":
    vocab = Vocabulary()
    vectorizer = BM25Vectorizer(vocab, domain_model="medicine")

    qa_rows = load_dataset("json", data_files="/home/weihua/medical-rag/raw_data/raw/train/sample/qa_50000.jsonl", split="train")
    # graph_rows = load_dataset("json", data_files="/home/weihua/medical-rag/raw_data/raw/train/graph_train_datasets.jsonl", split="train")
    
    # qa = qa.select_columns(["question", "answer", "qa_text"])
    # graph = graph.select_columns(["question", "answer", "qa_text"])
    # merged = concatenate_datasets([qa, graph])
    # merged.to_json("/home/weihua/medical-rag/raw_data/raw/train/merged_dataset.json")

    t0 = time.time()
    # # 并行分词 -> 建库/统计
    total_len = 0
    for toks in vectorizer.tokenize_parallel(qa_rows['text'], workers=8, chunksize=128):
        vocab.add_document(toks)
        total_len += len(toks)
    vocab.freeze()
    vocab.save("vocab.pkl.gz")