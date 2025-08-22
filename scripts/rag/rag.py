from MedicalRag.search.RAG import RAG
from MedicalRag.core.llm.HttpClient import OllamaClient
from MedicalRag.config.default_cfg import DenseEmbedCfg
from MedicalRag.core.embeddings.EmbeddingClient import FastEmbeddings
from MedicalRag.pipeline.query.query_pipeline import QueryPipeline
from langchain_ollama import ChatOllama
import re

def remove_think_blocks(text: str) -> str:
    # 删除 <think> ... </think> 之间的所有内容（含标签本身）
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

def main():
    llm = ChatOllama(
        model="qwen3:32b",        # 你在 Ollama 里拉取的模型名
        base_url="http://127.0.0.1:11434",  # 如用远程/自定义地址，可指定
        temperature=0.3,                   # 常用推理参数
        num_predict=8192,                      # 最大支持长度
    )
    search_dict = {
        "milvus": {
            "client": {
                "uri": "http://localhost:19530",
                "token": "root:Milvus",
                "db_name": "",
                "timeout_ms": 30000,
                "tls": False,
                "tls_verify": False
            },
            "collection": {
                "name": "medical_knowledge",
                "description": "RAG medical knowledge base",
                "load_on_start": True,
                "properties": ""
            }
        },
        "search": {
            "default_limit": 10,
            "output_fields": ["question", "answer", "sparse_vec_text"],
            "pagination": {"page_size": 10, "max_pages": 100},
            "rrf": {"enabled": False, "k": 100},
            "channels": [
                # {
                #     "name": "dense_doc",
                #     "field": "dense_vec_text",
                #     "enabled": True,
                #     "kind": "dense_query",
                #     "metric_type": "COSINE",
                #     "limit": 10,
                #     "params": { "ef": 256 },
                #     "expr": "",
                #     "weight": 0.4
                # },
                {
                    "name": "sparse_vec_text",
                    "field": "sparse_vec_text",
                    "enabled": True,
                    "kind": "sparse_query",
                    "metric_type": "IP",
                    "limit": 10,
                    "params": {"drop_ratio_search": 0.0},
                    "expr": "",
                    "weight": 0.3
                },
                {
                    "name": "dense_q",
                    "field": "dense_vec_summary",
                    "enabled": True,
                    "kind": "dense_query",
                    "metric_type": "COSINE",
                    "limit": 10,
                    "params": { "ef": 512 },
                    "expr": "",
                    "weight": 0.6
                }
            ]
        },
        "embedding": {
            "dense": {
                "provider": "ollama",
                "model": "bge-m3:latest",
                "base_url": "http://localhost:11434",
                "dim": 1024,
                "normalize": False,
                "prefixes": {"query": "", "document": ""}
            },
            "sparse_bm25": {
                "vocab_path": "vocab.pkl.gz",
                "domain_model": "medicine",
                "prune_empty_sparse": True,
                "empty_sparse_fallback": {"0": 0.0},
                "k1": 1.5,
                "b": 0.75
            }
        }
    }
    pipeline = QueryPipeline.create_from_config_dict(search_dict)
    rag = RAG(llm, pipeline)
    result = rag.answer(
        "我好像得了根尖周病，我该怎么办？", 
        retrieval_top_k=10, 
        return_retrieved_text=True,
        is_thinking=True
    )
    print(result[0])
    print("===============")
    print(result[1])
    
if __name__ == "__main__":
    main()