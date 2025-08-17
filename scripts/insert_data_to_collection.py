# examples/main_demo.py
from pymilvus import MilvusClient
from MedicalRag.config.default_cfg import load_cfg
from MedicalRag.core.vectorstore.milvus_client import MilvusConn
from MedicalRag.core.vectorstore.milvus_schema import ensure_collection
from MedicalRag.core.vectorstore.milvus_index import build_index_params
from MedicalRag.core.vectorstore.milvus_write import insert_rows
from MedicalRag.core.vectorstore.milvus_hybrid import HybridRetriever
import logging
from MedicalRag.data.processor.sparse import Vocabulary, BM25Vectorizer
from langchain_ollama import OllamaEmbeddings
import json

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def build_embedder(cfg):
    d = cfg.embedding.dense
    if d.provider == "ollama":
        return OllamaEmbeddings(model=d.model, base_url=d.base_url)
    raise ValueError(f"unsupported provider: {d.provider}")

def main():
    cfg = load_cfg("config/rag.yaml")

    # --- 连接与集合 ---
    conn = MilvusConn(cfg)
    assert conn.healthy(), "Milvus not healthy"
    client: MilvusClient = conn.get_client()

    # 索引参数 + 集合
    index_params = build_index_params(client, cfg)
    ensure_collection(client, cfg, index_params)

    # --- 写入 ---
    vocab = Vocabulary.load(cfg.embedding.sparse_bm25.vocab_path)
    vectorizer = BM25Vectorizer(vocab, domain_model=cfg.embedding.sparse_bm25.domain_model)
    rows = json.load(open(cfg.ingest.input_path, "r", encoding="utf-8"))

    # 组装数据
    emb = build_embedder(cfg)
    pre = cfg.embedding.dense.prefixes
    q_texts = [r["question"] for r in rows]
    qa_texts = [f'{r["question"]}\n\n{r["answer"]}' for r in rows]

    dense_q = emb.embed_documents([f'{pre["query"]} {t}' if pre["query"] else t for t in q_texts])
    dense_qa = emb.embed_documents([f'{pre["document"]} {t}' if pre["document"] else t for t in qa_texts])

    avgdl = max(1.0, vocab.sum_dl / max(1, vocab.N))
    data_rows = []
    for i, r in enumerate(rows):
        qa_tokens = vectorizer.tokenize(qa_texts[i])
        q_tokens = vectorizer.tokenize(q_texts[i])
        sp_qa = vectorizer.build_sparse_vec_from_tokens(qa_tokens, avgdl, update_vocab=False)
        sp_q  = vectorizer.build_sparse_vec_from_tokens(q_tokens,  avgdl, update_vocab=False)
        if cfg.embedding.sparse_bm25.prune_empty_sparse and not sp_q:
            sp_q = cfg.embedding.sparse_bm25.empty_sparse_fallback
        dept = str((r.get("departments") or r.get("department") or ["-1"])[0])

        data_rows.append({
            "dept_pk": dept,
            "question": r["question"],
            "answer": r["answer"],
            "qa_text": qa_texts[i],
            "dense_vec_q":  dense_q[i],
            "dense_vec_qa": dense_qa[i],
            "sparse_vec_q": sp_q,
            "sparse_vec_qa": sp_qa,
            "departments": r.get("departments"),
            "categories": r.get("categories"),
            "reasoning": r.get("reasoning"),
            "source_name": r.get("source_name") or cfg.ingest.source_name_default,
        })

    insert_rows(client, cfg, data_rows)

    # --- 检索（示例） ---
    retriever = HybridRetriever(client, cfg, emb, vectorizer)

    # 例如：顶层模板 'dept_pk in ["{dept}"]'
    # 且 sparse_q 通道模板为 'dept_pk in ["{dept}"] and source_name == "{src}"'
    queries=["梅毒", "巨肠症是什么东西？"]
    res = retriever.search(
        queries=queries,
        expr_vars={"src": "huatuo_qa"},
        page=1
    )

    for i, hits in enumerate(res):
        print(f"\n=== Query[{i}] {queries[i]} ===")
        for h in hits:
            print(h.id, h.distance, h.get("question"))

if __name__ == "__main__":
    main()
