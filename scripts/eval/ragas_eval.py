from ragas import evaluate
from ragas.metrics import AnswerRelevancy, Faithfulness, ContextRecall, ContextPrecision
import pandas as pd
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama, OllamaEmbeddings
from datasets import load_dataset
from MedicalRag.pipeline.query.query_pipeline import QueryPipeline
from MedicalRag.search.RAG import RAG
from ragas import EvaluationDataset
from tqdm import tqdm
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig
import os

def main():
    data = load_dataset("json", data_files="/home/weihua/medical-rag/raw_data/raw/train/sample/eval/new_qa_200.jsonl", split="train")
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
            "output_fields": ["question", "answer", "sparse_vec_text", "text"],
            "pagination": {"page_size": 10, "max_pages": 100},
            "rrf": {"enabled": False, "k": 100},
            "channels": [
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
    answer_llm = ChatOllama(
        base_url = "http://172.16.40.51:11434",
        model = "qwen3:32b",
        temperature = 0.8,
        num_predict = 8192
    )
    rag = RAG(answer_llm, pipeline)
    
    user_input_list = []  # 用户查询
    reference_list = []  # 参考答案
    retrieved_contexts_list = []  # 查询到的答案
    response_list = []  # RAG大模型的响应
    
    for i in tqdm(range(len(data))):
        user_input_list.append(data["new_question"][i])
        reference_list.append(data["answer"][i])
        llm_response, retrieved_contexts = rag.answer(
            data["new_question"][i], 
            retrieval_top_k=10, 
            return_retrieved_text=True,
            is_thinking=True,
            output_fields=["answer", "text"]
        )
        response_list.append(llm_response)
        retrieved_contexts_list.append([item["text"] for item in retrieved_contexts])
    df = pd.DataFrame(
        {
            "user_input": user_input_list,
            "retrieved_contexts": retrieved_contexts_list,
            "response": response_list,
            "reference": reference_list,
        }
    )
    rag_results = EvaluationDataset.from_pandas(df)
    local_embeddings = LangchainEmbeddingsWrapper(
        OllamaEmbeddings(
            base_url="http://172.16.40.51:11434",
            model="bge-m3:latest"
        )
    )
    
    llm = ChatOpenAI(
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model="qwen-plus",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        temperature=0.0,
        extra_body={
            "enable_thinking": False
        }
    )
    
    evaluator_llm = LangchainLLMWrapper(llm)
    results = evaluate(
        dataset=rag_results,
        metrics=[
            AnswerRelevancy(llm=evaluator_llm, embeddings=local_embeddings),
            Faithfulness(llm=evaluator_llm),
            ContextRecall(llm=evaluator_llm),
            ContextPrecision(llm=evaluator_llm),
        ],
        run_config=RunConfig(
            max_workers=1,
            timeout=900
        )
    )
    print(results)

if __name__ == "__main__":
    main()