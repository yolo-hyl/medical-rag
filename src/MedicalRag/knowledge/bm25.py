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

# 直接复用原项目的停用词过滤逻辑
from .sparse import (
    Vocabulary, BM25Vectorizer, 
    filter_stopwords, _init_seg_worker, _cut_worker
)

logger = logging.getLogger(__name__)

class SimpleBM25Manager:
    """简化的BM25管理器，兼容langchain"""
    
    def __init__(self, vocab_path: str = "vocab.pkl.gz", domain_model: str = "medicine"):
        self.vocab_path = vocab_path
        self.domain_model = domain_model
        self._vocab = None
        self._vectorizer = None
    
    @property
    def vocab(self) -> Vocabulary:
        """获取词表"""
        if self._vocab is None:
            vocab_file = Path(self.vocab_path)
            if vocab_file.exists():
                self._vocab = Vocabulary.load(str(vocab_file))
                logger.info(f"加载词表: {vocab_file}, 包含 {len(self._vocab.token2id)} 个词汇")
            else:
                self._vocab = Vocabulary()
                logger.info("创建新词表")
        return self._vocab
    
    @property
    def vectorizer(self) -> BM25Vectorizer:
        """获取向量化器"""
        if self._vectorizer is None:
            # 创建配置对象
            from ..config.models import SparseConfig
            config = SparseConfig(
                vocab_path=self.vocab_path,
                domain_model=self.domain_model
            )
            self._vectorizer = BM25Vectorizer(self.vocab, config)
        return self._vectorizer
    
    def build_vocab_from_texts(self, texts: List[str]) -> None:
        """从文本构建词表"""
        logger.info(f"构建词表，文本数量: {len(texts)}")
        
        # 使用原项目的并行分词
        for tokens in self.vectorizer.tokenize_parallel(texts):
            self.vocab.add_document(tokens)
        
        self.vocab.freeze()
        self.vocab.save(self.vocab_path)
        logger.info(f"词表构建完成，保存到: {self.vocab_path}")
    
    def vectorize_texts(self, texts: List[str]) -> List[Dict[int, float]]:
        """向量化文本"""
        avgdl = max(1.0, self.vocab.sum_dl / max(1, self.vocab.N)) if self.vocab.N > 0 else 1.0
        vectors = []
        
        for text in texts:
            vec = self.vectorizer.vectorize_text(text, avgdl)
            vectors.append(vec)
        
        return vectors
    
    def vectorize_single(self, text: str) -> Dict[int, float]:
        """向量化单个文本"""
        avgdl = max(1.0, self.vocab.sum_dl / max(1, self.vocab.N)) if self.vocab.N > 0 else 1.0
        return self.vectorizer.vectorize_text(text, avgdl)

# 为了兼容langchain的稀疏嵌入接口，创建一个适配器
from langchain_core.embeddings import Embeddings

class BM25SparseEmbedding(Embeddings):
    """BM25稀疏嵌入适配器"""
    
    def __init__(self, bm25_manager: SimpleBM25Manager):
        self.bm25_manager = bm25_manager
    
    def embed_documents(self, texts: List[str]) -> List[Dict[int, float]]:
        """嵌入文档列表"""
        return self.bm25_manager.vectorize_texts(texts)
    
    def embed_query(self, text: str) -> Dict[int, float]:
        """嵌入查询"""
        return self.bm25_manager.vectorize_single(text)