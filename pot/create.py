from pymilvus import MilvusClient, DataType, Function, FunctionType

URI = "http://localhost:19530"
TOKEN = "root:Milvus"
COL = "qa_knowledge"

client = MilvusClient(uri=URI, token=TOKEN)

DENSE_DIM = 768  # ← 改成你实际嵌入维度

schema = client.create_schema(auto_id=False, enable_dynamic_field=True)

# 主键 + 分区键
schema.add_field("id", datatype=DataType.VARCHAR, is_primary=True, max_length=64)
schema.add_field("dept_pk", datatype=DataType.VARCHAR, max_length=32, is_partition_key=True)

# 原文：可加 analyzer 以便未来切 BM25 内置
schema.add_field("question", datatype=DataType.VARCHAR, max_length=65535, enable_analyzer=True)
schema.add_field("answer",   datatype=DataType.VARCHAR, max_length=65535, enable_analyzer=True)
schema.add_field("qa_text",  datatype=DataType.VARCHAR, max_length=65535, enable_analyzer=True)

# 向量字段：稠密 + 稀疏
schema.add_field("dense_vec_q",  datatype=DataType.FLOAT_VECTOR,  dim=DENSE_DIM)
schema.add_field("dense_vec_qa",  datatype=DataType.FLOAT_VECTOR,  dim=DENSE_DIM)
schema.add_field("sparse_vec_q", datatype=DataType.SPARSE_FLOAT_VECTOR)
schema.add_field("sparse_vec_qa", datatype=DataType.SPARSE_FLOAT_VECTOR)
schema.add_field(
    "departments", 
    datatype=DataType.ARRAY, 
    element_type=DataType.INT64, 
    max_length=32,
    max_capacity=6
)
schema.add_field(
    "categories",
    datatype=DataType.ARRAY, 
    element_type=DataType.INT64, 
    max_length=32,
    max_capacity=8
)
schema.add_field(
    "source_name",
    datatype=DataType.VARCHAR,
    max_length=65535
)


# 索引：HNSW + 稀疏倒排
index_params = client.prepare_index_params()
index_params.add_index(
    field_name="dense_vec_qa",
    index_type="HNSW",
    metric_type="COSINE",                  # 与嵌入一致（COSINE/IP均可，取决于你的向量）
    params={"M": 32, "efConstruction": 200},
)
index_params.add_index(
    field_name="dense_vec_q",
    index_type="HNSW",
    metric_type="COSINE",                  # 与嵌入一致（COSINE/IP均可，取决于你的向量）
    params={"M": 32, "efConstruction": 200},
)
index_params.add_index(
    field_name="sparse_vec_qa",
    index_type="SPARSE_INVERTED_INDEX",
    metric_type="IP",                  # 手动BM25用IP；若使用内置BM25，metric_type=BM25
    params={"inverted_index_algo": "DAAT_MAXSCORE"},
)
index_params.add_index(
    field_name="sparse_vec_q",
    index_type="SPARSE_INVERTED_INDEX",
    metric_type="IP",                  # 手动BM25用IP；若使用内置BM25，metric_type=BM25
    params={"inverted_index_algo": "DAAT_MAXSCORE"},
)
if client.has_collection(collection_name=COL):
    print(f"Collection {COL} 已存在，正在删除...")
    client.drop_collection(collection_name=COL)
    print("删除完成")
client.create_collection(
    collection_name=COL,
    schema=schema,
    index_params=index_params,
    num_partitions=128,
    properties={"partitionkey.isolation": True},  # 仅 HNSW 有效
)
