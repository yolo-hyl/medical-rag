"""
RAG基础评测
"""
import logging
from MedicalRag.config.loader import ConfigLoader
from MedicalRag.rag.SimpleRag import SimpleRAG
from langchain_openai import ChatOpenAI, OpenAIEmbeddings  
import os
from langchain_community.embeddings import DashScopeEmbeddings
from MedicalRag.rag.RagEvaluate import RagasRagEvaluate
from datasets import load_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

def main():
    # 加载配置
    config_manager = ConfigLoader()
    # 创建基础RAG系统
    rag = SimpleRAG(config_manager.config)
    eval_data = load_dataset("json", data_files="data/eval/new_qa_200.jsonl", split="train")
    qwen_llm = ChatOpenAI(
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model="qwen-plus",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        temperature=0.0,
        extra_body={
            "enable_thinking": False
        }
    )
    qwen_embedding = DashScopeEmbeddings(
        model="text-embedding-v3",
        dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
    )
    eval = RagasRagEvaluate(rag_components=rag, eval_datasets=eval_data, eval_llm=qwen_llm, embedding=qwen_embedding)
    eval.do_sample(1)
    print(eval.do_evaluate(datasets_query_field_name="new_question", datasets_reference_field_name="answer"))

if __name__ == "__main__":
    main()
