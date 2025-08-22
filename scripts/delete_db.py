from pymilvus import MilvusClient
from MedicalRag.config.default_cfg import load_cfg
from MedicalRag.core.vectorstore.milvus_client import MilvusConn
from MedicalRag.core.vectorstore.milvus_write import insert_rows
import logging
from MedicalRag.data.processor.sparse import Vocabulary, BM25Vectorizer
from MedicalRag.core.embeddings.EmbeddingClient import FastEmbeddings
from datasets import load_dataset
import traceback
from tqdm import tqdm

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
# 关闭 httpx 的日志
logging.getLogger("httpx").setLevel(logging.WARNING)


def extract_qa_first(example):
    # 取第一个数组的第一个元素
    example["question"] = example["questions"][0][0] if example["questions"] and example["questions"][0] else None
    example["answer"] = example["answers"][0] if example["answers"][0] else None
    example["text"] = f"{example['question']}\n\n{example['answer']}"
    return example

def extract_gragh_first(example):
    example["question"] = example["questions"][0] if example["questions"][0] else None
    example["answer"] = example["answers"][0] if example["answers"][0] else None
    example["text"] = f"{example['question']}\n\n{example['answer']}"
    if len(example["question"]) > 60000:  # 中文1字符占3字节
        example["question"] = example["question"][0:60000]
    if len(example["answer"]) > 60000:
        example["answer"] = example["answer"][0:60000]
    if len(example["text"]) > 60000:
        example["text"] = example["text"][0:60000]
    return example


def main():
    ########  加载配置  ############
    cfg = load_cfg("scripts/examples/default.yaml")  

    ########  连接与集合  ############
    conn = MilvusConn(cfg)
    assert conn.healthy(), "Milvus not healthy"
    client: MilvusClient = conn.get_client()
    
    ########  加载数据集  ############
    # rows = load_dataset("json", data_files="/home/weihua/medical-rag/raw_data/tf-data/qa/*.json", split="train")
    data = load_dataset("json", data_files="/home/weihua/medical-rag/raw_data/raw/train/graph_train_datasets.jsonl", split="train")
    data = data.map(extract_gragh_first)
    
    data = data.select_columns(["question", "answer", "text"])
    ########  加载数据集  ############
    delete_num = 0
    for i in tqdm(range(len(data))):
        question = data[i]['question']
        q = client.query(
            collection_name="medical_knowledge", 
            filter=f"question == '{question}'"
        )
        if len(q) == 0:
            continue
        else:
            d = client.delete(
                collection_name="medical_knowledge", 
                filter=f"question == '{data[i]['question']}'"
            )
            delete_num += d['delete_count']

    print(f"删除数据 {delete_num} 条")

if __name__ == "__main__":
    main()
