"""把自管理词表的 BM25 包装成和稠密向量一致的调用方式（embed_documents / embed_query）

v2 里它继承 LangChain 的 Embeddings；核心层去掉 LangChain 后只保留这两个方法。
分词是 CPU 密集的，异步调用方应通过 asyncio.to_thread 执行。
"""
from typing import Dict, List

from .vocab import BM25Vectorizer, Vocabulary


class BM25SparseEmbedding:
    def __init__(self, vocab: Vocabulary, vectorizer: BM25Vectorizer):
        self.vocab = vocab
        self.vectorizer = vectorizer

    def _avgdl(self) -> float:
        return max(1.0, self.vocab.sum_dl / max(1, self.vocab.N)) if self.vocab.N > 0 else 1.0

    def embed_documents(self, texts: List[str]) -> List[Dict[int, float]]:
        return self.vectorizer.vectorize_texts(texts, self._avgdl())

    def embed_query(self, text: str) -> Dict[int, float]:
        return self.vectorizer.vectorize_texts([text], self._avgdl())[0]
