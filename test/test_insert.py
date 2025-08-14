from pymilvus import MilvusClient, DataType, Function, FunctionType

URI = "http://localhost:19530"
TOKEN = "root:Milvus"
COL = "qa_knowledge_v26"

client = MilvusClient(uri=URI, token=TOKEN)

# 1) Schema
schema = client.create_schema(auto_id=False, enable_dynamic_field=True)

schema.add_field("id", datatype=DataType.VARCHAR, is_primary=True, max_length=64)
schema.add_field("dept_pk", datatype=DataType.VARCHAR, max_length=32, is_partition_key=True)  # 分区键
schema.add_field("question", datatype=DataType.VARCHAR, max_length=65535, enable_analyzer=True)
schema.add_field("answer",   datatype=DataType.VARCHAR, max_length=65535, enable_analyzer=True)
schema.add_field("qa_text",  datatype=DataType.VARCHAR, max_length=65535, enable_analyzer=True)

# 稠密/稀疏向量字段
schema.add_field("dense_vec_qa", datatype=DataType.FLOAT_VECTOR, dim=1536)
schema.add_field("sparse_vec_qa", datatype=DataType.SPARSE_FLOAT_VECTOR)
# 如需额外字段：
schema.add_field("dense_vec_q",  datatype=DataType.FLOAT_VECTOR, dim=1536, nullable=True)
schema.add_field("dense_vec_a",  datatype=DataType.FLOAT_VECTOR, dim=1536, nullable=True)
schema.add_field("sparse_vec_a", datatype=DataType.SPARSE_FLOAT_VECTOR, nullable=True)

# 2) 使用 Milvus 内建 BM25：把 qa_text -> sparse_vec_qa 自动生成（推荐）
bm25_fn = Function(
    name="qa_bm25",
    function_type=FunctionType.BM25,
    input_field_names=["qa_text"],
    output_field_names=["sparse_vec_qa"],
    # 可选：中文分词器也可额外通过 analyzer/multi_analyzer 细化
)
schema.add_function(bm25_fn)

# 3) 索引（dense 选 HNSW；超大规模可改 IVF_PQ / GPU_IVF_PQ；sparse 用 SPARSE_INVERTED_INDEX+BM25）
index_params = client.prepare_index_params()
index_params.add_index(
    field_name="dense_vec_qa",
    index_type="HNSW",      # 或 IVF_PQ / GPU_IVF_PQ
    metric_type="IP",       # 与所用嵌入一致
    params={"M": 32, "efConstruction": 200}  # HNSW 关键参数
)
index_params.add_index(
    field_name="sparse_vec_qa",
    index_type="SPARSE_INVERTED_INDEX",
    metric_type="BM25",     # 内建 BM25 时使用 BM25 metric
    params={"inverted_index_algo": "DAAT_WAND"}  # 2.6 新推荐算法
)

# 4) 创建集合（可指定分区数量，启用 Partition Key Isolation 仅 HNSW 生效）
client.create_collection(
    collection_name=COL,
    schema=schema,
    index_params=index_params,
    num_partitions=128,
    properties={"partitionkey.isolation": True},  # 仅 HNSW 支持
)

import os
from typing import Literal

Provider = Literal["openai", "hf"]
PROVIDER: Provider = os.getenv("EMBED_PROVIDER", "openai")

if PROVIDER == "openai":
    from langchain_openai import OpenAIEmbeddings
    dense = OpenAIEmbeddings(model="text-embedding-3-large")
else:
    # 本地/自托管：适合中文如 BAAI/bge-m3 或 bge-large-zh-v1.5
    from langchain_huggingface import HuggingFaceEmbeddings
    dense = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")  # 需要 sentence-transformers


import math, pkuseg
from collections import Counter, defaultdict

seg = pkuseg.pkuseg(model_name='medicine')  # 医学领域
vocab, df = {}, defaultdict(int)

def build_bm25_corpus_stats(docs):
    # docs: list[str]
    tokenized = []
    for d in docs:
        toks = seg.cut(d)
        tokenized.append(toks)
        for t in set(toks):
            df[t] += 1
    for t in df:
        if t not in vocab:
            vocab[t] = len(vocab) + 1  # 0 号留空
    N = len(docs)
    idf = {t: math.log((N - df[t] + 0.5) / (df[t] + 0.5) + 1) for t in df}
    return tokenized, idf

def bm25_sparse_vector(text, idf, k1=1.2, b=0.75, avgdl=300):
    toks = seg.cut(text)
    tf = Counter(toks)
    dl = len(toks)
    vec = {}
    for t, f in tf.items():
        if t not in vocab: 
            continue
        score = idf.get(t, 0.0) * ((f*(k1+1)) / (f + k1*(1 - b + b*dl/avgdl)))
        if score > 0:
            vec[vocab[t]] = float(score)
    return vec  # {token_id: weight}

# 构造 sparse_vec_a 之类字段
row = {
  "id":"...", "dept_pk":"3",
  "question": "...", "answer": "...", "qa_text":"...",
  "dense_vec_qa": dense.embed_documents(["..."])[0],
  "sparse_vec_a": bm25_sparse_vector("仅针对答案字段构建bm25", idf)  # 可选
}
client.insert(collection_name=COL, data=[row])
