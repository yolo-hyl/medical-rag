from pymilvus import MilvusClient
from MedicalRag.config.default_cfg import load_cfg
from MedicalRag.core.vectorstore.milvus_client import MilvusConn
from MedicalRag.core.vectorstore.milvus_write import insert_rows
import logging
from MedicalRag.data.processor.sparse import Vocabulary, BM25Vectorizer
from MedicalRag.core.llm.EmbeddingClient import FastEmbeddings
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


def main():
    ########  加载配置  ############
    cfg = load_cfg("scripts/default.yaml")  

    ########  连接与集合  ############
    conn = MilvusConn(cfg)
    assert conn.healthy(), "Milvus not healthy"
    client: MilvusClient = conn.get_client()

    ########  加载自定义的词表和BM25算法配置  ############
    vocab = Vocabulary.load(cfg.embedding.sparse_bm25.vocab_path)
    vectorizer = BM25Vectorizer(vocab, domain_model=cfg.embedding.sparse_bm25.domain_model)
    ########  加载自定义的词表和BM25算法配置  ############
    
    ########  加载数据集  ############
    # rows = load_dataset("json", data_files="/home/weihua/medical-rag/raw_data/tf-data/qa/*.json", split="train")
    data = load_dataset("json", data_files="/home/weihua/medical-rag/raw_data/raw/train/qa_train_datasets.jsonl", split="train")
    data = data.map(extract_qa_first)
    
    data = data.select_columns(["question", "answer", "text"])
    ########  加载数据集  ############
    
    
    ########  组装数据  ############# 
    emb = FastEmbeddings(cfg.embedding.dense)
    try:
        dense_text = emb.embed_documents(data['text'], show_progress=True, batch_size=32)
        avgdl = max(1.0, vocab.sum_dl / max(1, vocab.N))
        data_rows = []
        for i in tqdm(range(len(data)), desc="tokenize"):
            text_tokens = vectorizer.tokenize(data['text'][i])
            sp_text = vectorizer.build_sparse_vec_from_tokens(text_tokens, avgdl, update_vocab=False)
            if cfg.embedding.sparse_bm25.prune_empty_sparse and not sp_text:
                sp_text = cfg.embedding.sparse_bm25.empty_sparse_fallback
            data_rows.append({
                "question": data["question"][i],
                "answer": data["answer"][i],
                "text": data["text"][i],
                "doc_id": "",
                "chunk_id": -1,
                "dense_vec_text":  dense_text[i],
                "sparse_vec_text": sp_text,
                "departments": [],
                "categories": [],
                "source": "qa",
            })
        ########  开始插入数据  ############# 
        insert_rows(client, cfg, data_rows, show_progress=True)
    except Exception as e:
        print(e)
        traceback.print_exc()
    finally:
        emb.close()

if __name__ == "__main__":
    main()
