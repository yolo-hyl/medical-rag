"""
Basic retrieval‑augmented generation (RAG) for MedicalRag.

This module defines the ``BasicRAG`` class which wires together a
knowledge base (Milvus hybrid vector store), a prompt template and a
language model into a LangChain retrieval chain.  It exposes a
simple ``answer`` method that accepts a query and returns either
just the answer or the answer along with the supporting context.

Only the basic RAG mode is implemented here.  If you wish to add
agentic behaviour, such as web search or tool use, extend this class
or create a new one in a separate module.  Keeping the base class
simple makes it easy to understand and maintain.
"""

from __future__ import annotations

import logging
from typing import List, Dict, Any, Optional, Union

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from ..config.models import *
from ..config.models import AppConfig
from ..core.KnowledgeBase import MedicalHybridKnowledgeBase
from ..prompts.templates import get_prompt_template


logger = logging.getLogger(__name__)


class BasicRAG:
    """A simple RAG system built using LangChain components."""

    def __init__(self, milvus_config: MilvusConfig, embedding_config: EmbeddingConfig, llm_config: LLMConfig) -> None:
        self.knowledge_base = MedicalHybridKnowledgeBase(milvus_config=milvus_config, embedding_config=embedding_config, llm_config=llm_config)
        # Initialise the collection if it does not exist
        try:
            self.knowledge_base.initialize_collection(drop_old=milvus_config.drop_old)
        except Exception as e:
            logger.warning(f"Failed to initialise collection on BasicRAG init: {e}")
        # Set up prompt
        template = get_prompt_template("basic_rag")
        if isinstance(template, dict):
            self.prompt = ChatPromptTemplate.from_messages([
                ("system", template["system"]),
                ("human", template["user"]),
            ])
        else:
            self.prompt = ChatPromptTemplate.from_template(template)
        # Set up retrieval chain
        self._setup_chain()

    def _setup_chain(self) -> None:
        """Construct the retrieval and document combination chain."""
        # Compose documents into a single string using the prompt
        document_chain = create_stuff_documents_chain(
            self.knowledge_base.llm,
            self.prompt,
        )
        # Build a retriever from the knowledge base
        retriever = self.knowledge_base.as_retriever()
        # Combine into a RAG chain
        self.rag_chain = create_retrieval_chain(retriever, document_chain)

    def answer(self, query: str, return_context: bool = False, filters: Optional[Dict[str, Any]] = None) -> Union[str, Dict[str, Any]]:
        """Answer a question using the RAG pipeline.

        Parameters
        ----------
        query:
            The user's natural language question.
        return_context:
            If True, return both the answer and the retrieved documents.
        filters:
            Optional filter expression to restrict the search.  The
            expression syntax should follow Milvus filter syntax.

        Returns
        -------
        str or dict
            If ``return_context`` is False, only the answer is returned.
            Otherwise a dictionary containing the answer, the query
            and a list of context documents is returned.
        """
        logger.info(f"Handling query: {query}")
        # Update retriever with optional filters
        if filters:
            # Rebuild retriever on demand
            retriever = self.knowledge_base.vectorstore.as_retriever(
                search_kwargs={"filter": filters, "k": self.config.search.top_k}
            )
            # Recreate chain with new retriever
            document_chain = create_stuff_documents_chain(
                self.knowledge_base.llm,
                self.prompt,
            )
            self.rag_chain = create_retrieval_chain(retriever, document_chain)
        try:
            result = self.rag_chain.invoke({"input": query})
            answer = result.get("answer", "抱歉，根据提供的资料无法回答您的问题。")
            if return_context:
                context_docs = result.get("context", [])
                formatted_context = [
                    {"content": doc.page_content, "metadata": doc.metadata}
                    for doc in context_docs
                ]
                return {"answer": answer, "query": query, "context": formatted_context}
            return answer
        except Exception as e:
            logger.error(f"RAG processing failed: {e}")
            error_msg = "抱歉，处理您的问题时出现错误。请稍后再试。"
            if return_context:
                return {"answer": error_msg, "query": query, "context": []}
            return error_msg