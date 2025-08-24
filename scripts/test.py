# -*- coding: utf-8 -*-
from langchain_milvus import Milvus, BM25BuiltInFunction
from langchain_ollama.embeddings import OllamaEmbeddings
from pymilvus import MilvusClient, DataType, Collection, connections, FunctionType, Function

URI = "http://localhost:19530"
OLLAMA_BASE_URL = "http://localhost:11434"
MODEL = "bge-m3:latest"   # 1024 维

COL = "kb_multi_fields_demo"

# ========== 1) 预建集合：显式列 ==========
client = MilvusClient(uri=URI, token="root:Milvus")

# 若已存在且你想重建，先删除；如果不想删，就把下面 drop 注释掉
if client.has_collection(collection_name=COL):
    client.drop_collection(collection_name=COL)

schema = MilvusClient.create_schema(
    auto_id=True,
    enable_dynamic_field=False,   # 关闭动态字段：source/question/answer 走显式列
)
schema.add_field(field_name="pk",           datatype=DataType.INT64, is_primary=True)
schema.add_field(field_name="text",         datatype=DataType.VARCHAR, max_length=4096, enable_analyzer=True)
schema.add_field(field_name="source",       datatype=DataType.VARCHAR, max_length=256)
schema.add_field(field_name="question",     datatype=DataType.VARCHAR, max_length=1024)
schema.add_field(field_name="answer",       datatype=DataType.VARCHAR, max_length=4096)
schema.add_field(field_name="vec_question", datatype=DataType.FLOAT_VECTOR, dim=1024)
schema.add_field(field_name="vec_text",     datatype=DataType.FLOAT_VECTOR, dim=1024)
schema.add_field(field_name="sparse",       datatype=DataType.SPARSE_FLOAT_VECTOR)
bm25_fn = Function(
    name="bm25_text_to_sparse",
    function_type=FunctionType.BM25,
    input_field_names=["text"],
    output_field_names=["sparse"],
)
schema.add_function(bm25_fn)
client.create_collection(collection_name=COL, schema=schema)

# （可选）你也可以此时给 sparse 建默认索引，或导完再建：
# client.create_index(
#     collection_name=COL,
#     field_name="sparse",
#     index_params={"index_type": "AUTOINDEX", "metric_type": "BM25"},
# )

# ========== 2) 初始化向量与 BM25 ==========
q_emb = OllamaEmbeddings(model=MODEL, base_url=OLLAMA_BASE_URL)
d_emb = OllamaEmbeddings(model=MODEL, base_url=OLLAMA_BASE_URL)

# 2) BM25 由 Milvus 端生成稀疏向量（不要把它放进 embedding 列表里）
bm25 = BM25BuiltInFunction(
    input_field_names="text",
    output_field_names="sparse",
    analyzer_params={
        "tokenizer": "jieba",
        "filter": [{"type": "stop", "stop_words": []}],  # 这里放你的停用词列表
    },
)

# 3) 初始化向量库：为每个字段分别给 index/search 参数（顺序与 vector_field 对齐）
vs = Milvus(
    embedding_function=[q_emb, d_emb],
    builtin_function=bm25,
    vector_field=["vec_question", "vec_text", "sparse"],   # [dense1, dense2, bm25]
    text_field="text",
    collection_name="kb_multi_fields_demo",
    connection_args={"uri": URI},
    consistency_level="Strong",
    drop_old=False,
    auto_id=True,
    enable_dynamic_field=False,  # 和建表一致，避免进 $meta

    # 关键：逐字段索引参数
    index_params=[
        {"index_type": "HNSW", "metric_type": "IP",   "params": {"M": 16, "efConstruction": 200}},  # vec_question
        {"index_type": "HNSW", "metric_type": "IP",   "params": {"M": 16, "efConstruction": 200}},  # vec_text
        {"index_type": "AUTOINDEX", "metric_type": "BM25", "params": {}},                            # sparse (BM25)
    ],
    # 关键：逐字段检索参数
    search_params=[
        {"metric_type": "IP",   "params": {"ef": 64}},  # vec_question
        {"metric_type": "IP",   "params": {"ef": 64}},  # vec_text
        {"metric_type": "BM25", "params": {}},          # sparse
    ],
)


# ========== 3) 写入一条数据 ==========
record = {
    "question": "冠心病的治疗方法？",
    "answer": "冠心病治疗包括：1.药物治疗：抗血小板、降脂药等；2.介入治疗：支架植入术；3.外科治疗：搭桥手术；4.生活方式改变。"
}
text = f"问题：{record['question']} 回答：{record['answer']}"

q_vec = q_emb.embed_documents([record["question"]])[0]
d_vec = d_emb.embed_documents([text])[0]

pk = 1

vs.add_embeddings(
    texts=[text],
    embeddings=[[q_vec, d_vec]],
    metadatas=[{
        "question": record["question"],  # 显式列
        "answer": record["answer"],      # 显式列
        "source": "demo"                # 显式列
    }],
    ids=None
)

# ========== 4) 检索 ==========
query = "怎么治疗冠心病？"
query_vec = q_emb.embed_documents([query])[0]  # 使用 q_emb 来嵌入问题
docs = vs.similarity_search(
    query, k=5,
    ranker_type="weighted",
    ranker_params={"weights": [0.0, 0.0, 0.0]},
)
for i, d in enumerate(docs, 1):
    print(i, getattr(d, "score", None), d.page_content, d.metadata)
    pk = d.metadata["pk"]

# ========== 5) 校验：schema、index、向量一致性 ==========
connections.connect(uri=URI)
col = Collection(COL)

schema_ok = []
from pymilvus import DataType as DT  # 为了简写
for f in col.schema.fields:
    if f.dtype in (DT.FLOAT_VECTOR, DT.BINARY_VECTOR):
        schema_ok.append((f.name, "DENSE", getattr(f, "dim", None)))
    elif f.dtype == DT.SPARSE_FLOAT_VECTOR:
        schema_ok.append((f.name, "SPARSE", None))
    else:
        schema_ok.append((f.name, str(f.dtype), getattr(f, "max_length", None)))
print("schema_ok =", schema_ok)

for idx in col.indexes:
    print("index:", idx.field_name, idx.to_dict().get("index_param", {}))

# 查询显式列（注意：不要在 output_fields 里写 "$meta"）
rows = col.query(
    expr=f'pk == {pk}',
    output_fields=["vec_question", "vec_text", "text", "question", "answer", "source"]
)
row = rows[0]

import numpy as np
dq = np.linalg.norm(np.array(row["vec_question"], dtype="float32") - np.array(q_vec, dtype="float32"))
dt = np.linalg.norm(np.array(row["vec_text"], dtype="float32") - np.array(d_vec, dtype="float32"))
print("Δ(vec_question)=", dq, "Δ(vec_text)=", dt)
# 经验上 Δ 接近 0（<1e-5），如果你没有重算嵌入的话
