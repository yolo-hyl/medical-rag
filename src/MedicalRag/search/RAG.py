from tqdm import tqdm
from openai import OpenAI
from pymilvus import MilvusClient
from langchain_core.language_models import BaseChatModel
from ..pipeline.query.query_pipeline import QueryPipeline
from ..core.embeddings.EmbeddingClient import FastEmbeddings
from ..data.processor.sparse import Vocabulary, BM25Vectorizer
from typing import List, Dict, Any, Optional, Union
from ..config.prompts import RAG_PROMPT, RAG_SYS_PROMPT
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
import re

def remove_think_blocks(text: str) -> str:
    # 删除 <think> ... </think> 之间的所有内容（含标签本身）
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

class RAG:
    """
    RAG (Retrieval-Augmented Generation) class built upon OpenAI and Milvus.
    """

    def __init__(self, llm_client: BaseChatModel, search_pipeline: QueryPipeline):
        self.llm_client = llm_client
        assert isinstance(llm_client, BaseChatModel), f"请传入BaseChatModel实例，当前实例{llm_client.__bases__[0].__name__}"
        self.search_pipeline = search_pipeline
        self.search_pipeline.setup()
        
    def retrieve(
        self, 
        question: str, 
        expr_vars: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = 3
    ) -> List[str]:
        """
        Retrieve the most similar text data to the given question.
        """
        results = self.search_pipeline.search_single(
            query=question,
            expr_vars=expr_vars,
            limit=top_k
        )
        return results
    
    def build_messages(self, query, retrieved_texts):
        retrieved_docs = "\n".join(f"Q: {retrieve_text['question']}\nA: {retrieve_text['answer']}"for retrieve_text in retrieved_texts)
        messages = [
            SystemMessage(content=RAG_SYS_PROMPT),
            HumanMessage(content=RAG_PROMPT.format(query=query, retrieved_docs=retrieved_docs))
        ]
        return messages
    
    def answer(
        self,
        question: str,
        retrieval_top_k: int = 3,
        return_retrieved_text: bool = False,
        output_fields: list[str] = "",
        is_thinking: bool = True
    ):
        """
        Answer the given question with the retrieved knowledge.
        """
        retrieved_texts = self.retrieve(question, top_k=retrieval_top_k)
        if len(retrieved_texts) == 0:
            return "检索结果为空", []
        result = self.llm_client.invoke(
            self.build_messages(query=question, retrieved_texts=retrieved_texts)
        )
        if is_thinking:
            result = remove_think_blocks(result.content)
        res_retrieved_texts = []
        assert all([(output_field in retrieved_texts[0]) for output_field in output_fields]), "需要输出的属性不存在"
        for item in retrieved_texts:
            res_retrieved_texts.append({k:item[k] for k in item if k in output_fields})
        return result, res_retrieved_texts if return_retrieved_text else None
