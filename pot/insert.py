# ingest_v26.py
from typing import Dict, Any, List
from pymilvus import MilvusClient
from MedicalRag.data.processor.sparse import Vocabulary, BM25Vectorizer  # 你的代码
import json
from langchain_ollama import OllamaEmbeddings

class _EmbedAdapter:
    def __init__(self, impl: OllamaEmbeddings, prefix: str | None = None):
        self.impl = impl
        self.prefix = prefix  # 为 Nomic 添加 task 前缀

    def _add_prefix(self, texts: List[str]) -> List[str]:
        if not self.prefix:
            return texts
        return [f"{self.prefix} {t}" for t in texts]

    def embed(self, texts: List[str]) -> List[List[float]]:
        # LangChain 的 OllamaEmbeddings 接口是 embed_documents
        return self.impl.embed_documents(self._add_prefix(texts))

    def embed_query(self, text: str) -> List[float]:
        return self.impl.embed_query(f"{self.prefix} {text}" if self.prefix else text)
    
def build_embedder(kind: str = "ollama", **kwargs):
    """
    统一构造：支持 'ollama'。kwargs 里可放:
      - model="bge-m3:latest"
      - base_url="http://localhost:11434"
    """
    if kind == "ollama":
        # 文档用 search_document，查询用 search_query —— 下方 prepare_rows 分别实例化
        model = kwargs.get("model", "bge-m3:latest")
        base_url = kwargs.get("base_url", "http://localhost:11434")
        return OllamaEmbeddings(model=model, base_url=base_url)
    raise ValueError(f"unknown embedder kind: {kind}")


def prepare_rows(
    rows: List[Dict[str, Any]],
    vocab: Vocabulary,
    vectorizer: BM25Vectorizer,
    embedder_kind: str = "hf",
    embedder_kwargs: Dict[str, Any] = None
):
    # 计算 avgdl（你已有 sum_dl/N；也可用当前批 tokens 汇总）
    avgdl = max(1.0, vocab.sum_dl / max(1, vocab.N))

    # 稠密向量（一次性 batch）
    qa_texts = [f"{r['question']} \n\n {r['answer']}" for r in rows]
    _impl_doc = build_embedder(embedder_kind, **(embedder_kwargs or {}))
    qa_dense_model = _EmbedAdapter(_impl_doc, prefix="search_document:")
    qa_dense_vecs = qa_dense_model.embed(qa_texts)

    q_texts = [r['question'] for r in rows]
    _impl_q = build_embedder(embedder_kind, **(embedder_kwargs or {}))
    q_dense_model = _EmbedAdapter(_impl_q, prefix="search_query:")
    q_dense_vecs = q_dense_model.embed(q_texts)

    batch = []
    for i, r in enumerate(rows):
        dept = str((r.get("departments") or r.get("department") or ["-1"])[0])  # 选第一个当主科室id

        # 稀疏向量（用自定义的BM25）
        qa = vectorizer.tokenize(f"{r['question']} \n\n {r['answer']}")
        q = vectorizer.tokenize(r['question'])
        sparse_vec_qa = vectorizer.build_sparse_vec_from_tokens(qa, avgdl, update_vocab=False)  # {tid: weight}
        sparse_vec_q = vectorizer.build_sparse_vec_from_tokens(q, avgdl, update_vocab=False)

        row = {
            "id": r["id"],
            "dept_pk": dept,
            "question": r["question"],
            "answer": r["answer"],
            "qa_text": f"{r['question']} \n\n {r['answer']}",
            "sparse_vec_q": sparse_vec_q,
            "dense_vec_qa": qa_dense_vecs[i],
            "dense_vec_q": q_dense_vecs[i],
            "sparse_vec_qa": sparse_vec_qa,      # ← 直接塞字典即可
            # 其他元数据走动态字段（$meta）
            "categories": r.get("categories"),
            "departments": r.get("departments"),
            "reasoning": r.get("reasoning"),
            "source_name": r.get("source_name") if r.get("source_name") else 'huatuo_qa',
        }
        batch.append(row)
    return batch

def insert_rows(client, data_rows: List[Dict[str, Any]]):
    # 批量插入
    res = client.insert(collection_name=COL, data=data_rows)
    client.load_collection(COL)
    return res

if __name__ == '__main__':
    URI = "http://localhost:19530"
    TOKEN = "root:Milvus"
    COL = "qa_knowledge"

    client = MilvusClient(uri=URI, token=TOKEN)
    vocab = Vocabulary.load("vocab.pkl.gz")
    vectorizer = BM25Vectorizer(vocab, domain_model="medicine")
    rows = json.load(open("raw_data/tf-data/qa.temp_10.json", "r", encoding="utf-8"))

    data_rows = prepare_rows(
        rows, vocab, vectorizer,
        embedder_kind="ollama",
        embedder_kwargs={"model": "bge-m3:latest", "base_url": "http://172.16.40.51:11434"}
    )
    insert_rows(client, data_rows)