from typing import List, Dict
from datasets import load_dataset
from MedicalRag.data.processor.sparse import Vocabulary, BM25Vectorizer
import time
from datasets import concatenate_datasets
def extract_qa_first(example):
    # 取第一个数组的第一个元素
    example["question"] = example["questions"][0][0] if example["questions"] and example["questions"][0] else None
    example["answer"] = example["answers"][0] if example["answers"][0] else None
    example["qa_text"] = f"{example['question']}\n\n{example['answer']}"
    return example

def extract_gragh_first(example):
    example["question"] = example["questions"][0] if example["questions"][0] else None
    example["answer"] = example["answers"][0] if example["answers"][0] else None
    example["qa_text"] = f"{example['question']}\n\n{example['answer']}"
    return example

if __name__ == "__main__":
    vocab = Vocabulary()
    vectorizer = BM25Vectorizer(vocab, domain_model="medicine")

    qa_rows = load_dataset("json", data_files="/home/weihua/medical-rag/raw_data/raw/train/qa_train_datasets.jsonl", split="train")
    graph_rows = load_dataset("json", data_files="/home/weihua/medical-rag/raw_data/raw/train/graph_train_datasets.jsonl", split="train")
    
    qa = qa_rows.map(extract_qa_first)
    graph = graph_rows.map(extract_gragh_first)
    
    qa = qa.select_columns(["question", "answer", "qa_text"])
    graph = graph.select_columns(["question", "answer", "qa_text"])
    merged = concatenate_datasets([qa, graph])
    merged.to_json("/home/weihua/medical-rag/raw_data/raw/train/merged_dataset.json")

    t0 = time.time()
    # # 并行分词 -> 建库/统计
    total_len = 0
    for toks in vectorizer.tokenize_parallel(merged['qa_text'], workers=8, chunksize=128):
        vocab.add_document(toks)
        total_len += len(toks)
    vocab.freeze()
    vocab.save("vocab.pkl.gz")
