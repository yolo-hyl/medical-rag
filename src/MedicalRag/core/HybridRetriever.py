import logging
from typing import List, Dict, Any, Optional, Union, Callable
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from ..config.models import *
from .KnowledgeBase import MedicalHybridKnowledgeBase

logger = logging.getLogger(__name__)

class MedicalHybridRetriever(BaseRetriever):
    """医疗混合检索器 - LangChain标准接口"""
    
    def __init__(self, knowledge_base: MedicalHybridKnowledgeBase, search_config: SearchRequest):
        super().__init__()
        self.knowledge_base = knowledge_base
        self._search_config = search_config
    
    @property
    def search_config(self):
        return self._search_config
    
    @search_config.setter
    def search_config(self, new_search_config: SearchRequest):
        """更新搜索配置"""
        self._search_config = new_search_config
    
    def _get_relevant_documents(
        self, 
        query: str
    ) -> List[Document]:
        """检索逻辑"""
        self._search_config.query = query
        return self.knowledge_base.search(self._search_config.query)
