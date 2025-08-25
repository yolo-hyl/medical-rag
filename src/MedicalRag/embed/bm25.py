# src/medical_rag/knowledge/bm25.py
"""
BM25稀疏向量处理（保留原项目实现）
"""
import math
from typing import List, Dict, Iterable, Iterator
import pkuseg
from multiprocessing import Pool, cpu_count
import os, gzip, pickle
from pathlib import Path
import logging
from .sparse import Vocabulary, BM25Vectorizer

logger = logging.getLogger(__name__)

# 为了兼容langchain的稀疏嵌入接口，创建一个适配器
from langchain_core.embeddings import Embeddings

class BM25SparseEmbedding(Embeddings):
    """BM25稀疏嵌入适配器"""
    
    def __init__(self, vocab: Vocabulary, vectorizer: BM25Vectorizer):
        self.vocab = vocab
        self.vectorizer = vectorizer
    
    def embed_documents(self, texts: List[str]) -> List[Dict[int, float]]:
        """嵌入文档列表"""
        avgdl = max(1.0, self.vocab.sum_dl / max(1, self.vocab.N)) if self.vocab.N > 0 else 1.0
        return self.vectorizer.vectorize_texts(texts, avgdl)
    
    def embed_query(self, text: str) -> Dict[int, float]:
        """嵌入查询"""
        avgdl = max(1.0, self.vocab.sum_dl / max(1, self.vocab.N)) if self.vocab.N > 0 else 1.0
        return self.vectorizer.vectorize_texts([text], avgdl)[0]