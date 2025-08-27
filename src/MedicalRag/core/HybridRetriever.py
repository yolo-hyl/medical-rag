import logging
from typing import List, Dict, Any, Optional, Union, Callable
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from ..config.models import *
from .KnowledgeBase import MedicalHybridKnowledgeBase
import time

logger = logging.getLogger(__name__)

class MedicalHybridRetriever(BaseRetriever):
    """医疗混合检索器 - LangChain标准接口"""
    
    knowledge_base: MedicalHybridKnowledgeBase
    search_config: SearchRequest
    
    def __init__(self, knowledge_base: MedicalHybridKnowledgeBase, search_config: SearchRequest):
        super().__init__(knowledge_base=knowledge_base, search_config=search_config)
    
    def _get_relevant_documents(
        self, 
        inputs: dict
    ) -> List[Document]:
        """检索逻辑"""
        self.search_config.query = inputs.get("input", "")
        start_time = time.time()
        documents = self.knowledge_base.search(self.search_config)
        search_time = time.time() - start_time
        return {"documents": documents, "search_time": search_time}

