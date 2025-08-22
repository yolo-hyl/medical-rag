from typing import List, Dict
from pymilvus import MilvusClient, AnnSearchRequest, RRFRanker
from langchain_ollama import OllamaEmbeddings
from MedicalRag.data.processor.sparse import Vocabulary, BM25Vectorizer

URI = "http://localhost:19530"
COL = "qa_knowledge"

# 1) 准备批量 query 文本
queries: List[str] = [
    "梅毒",
    "巨肠症是什么东西？",
    "最广泛的性病是什么？"
]

# 2) 批量生成向量（注意前缀与入库一致）
emb = OllamaEmbeddings(model="bge-m3:latest", base_url="http://172.16.40.51:11434")

def add_prefix(texts: List[str], p: str|None):
    return [f"{p} {t}" if p else t for t in texts]

dense_q_list  = emb.embed_documents(add_prefix(queries, "search_query:"))        # N x 1024
dense_qa_list = emb.embed_documents(add_prefix(queries, "search_document:"))     # N x 1024

# 稀疏：用你自己的 BM25 计算 N 条
vocab = Vocabulary.load("vocab.pkl.gz")
vectorizer = BM25Vectorizer(vocab, domain_model="medicine")
avgdl = max(1.0, vocab.sum_dl / max(1, vocab.N))
sparse_q_list: List[Dict[int, float]] = []
for q in queries:
    toks = vectorizer.tokenize(q)
    t = []
    for tok in toks:
        if tok in ['什么', '？', '是', '如何']:
            continue
        t.append(tok)
    sp = vectorizer.build_sparse_vec_from_tokens(toks, avgdl, update_vocab=False)
    # Milvus 不允许空稀疏，空的话降级为仅稠密
    if not sp:
        sp = {0: 0.0}  # 或者直接跳过该通道
    sparse_q_list.append(sp)

# 3) 构造三个 AnnSearchRequest（每个 data 都是长度 N 的列表）
req_dense_q = AnnSearchRequest(
    data=dense_q_list,           # ← N 条
    anns_field="dense_vec_q",
    param={"metric_type": "COSINE", "params": {"ef": 256}},
    limit=5,
    expr=""                      # 批量时同一个 expr 作用到所有 query
)

req_dense_qa = AnnSearchRequest(
    data=dense_qa_list,          # ← N 条
    anns_field="dense_vec_qa",
    param={"metric_type": "COSINE", "params": {"ef": 256}},
    limit=5,
    expr=""
)

req_sparse_q = AnnSearchRequest(
    data=sparse_q_list,          # ← N 条（每条是 {token_id: weight}）
    anns_field="sparse_vec_q",
    param={"metric_type": "IP", "params": {"drop_ratio_search": 0.0}},
    limit=5,
    expr=""
)

client = MilvusClient(uri=URI, token="root:Milvus")
client.load_collection(COL)

# 4) RRF 重排（对不同量纲更稳）
ranker = RRFRanker(k=100)

res = client.hybrid_search(
    collection_name=COL,
    reqs=[req_dense_qa, req_sparse_q],
    ranker=ranker,
    limit=10,
    output_fields=["question","answer","departments","categories","source_name"]
)

# 5) 读取结果：外层长度 N；res[i] 是第 i 条 query 的 TopK
for i, hits in enumerate(res):
    print(f"\n=== Query[{i}] {queries[i]} ===")
    for h in hits:
        print(h.id, h.distance, h.get("question"))
