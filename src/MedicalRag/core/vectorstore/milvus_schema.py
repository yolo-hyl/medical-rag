"""
集合与字段定义、JSON字段/dynamic field/PK/PK分区
"""
# milvus/milvus_schema.py
from __future__ import annotations
from typing import Dict
from pymilvus import MilvusClient, DataType
from ...config.default_cfg import AppCfg, FieldCfg

# 映射 DataType
_DT_MAP: Dict[str, DataType] = {
    "BOOL": DataType.BOOL, "INT8": DataType.INT8, "INT16": DataType.INT16,
    "INT32": DataType.INT32, "INT64": DataType.INT64, "FLOAT": DataType.FLOAT,
    "DOUBLE": DataType.DOUBLE, "VARCHAR": DataType.VARCHAR, "JSON": DataType.JSON,
    "ARRAY": DataType.ARRAY, "FLOAT_VECTOR": DataType.FLOAT_VECTOR,
    "BINARY_VECTOR": DataType.BINARY_VECTOR, "SPARSE_FLOAT_VECTOR": DataType.SPARSE_FLOAT_VECTOR,
}

def _add_field(schema, f: FieldCfg):
    dt = _DT_MAP[f.dtype]
    kw = {}
    if f.is_primary: kw["is_primary"] = True
    if f.is_partition_key: kw["is_partition_key"] = True
    if f.enable_analyzer: kw["enable_analyzer"] = True
    if f.max_length is not None: kw["max_length"] = f.max_length
    if f.dim is not None: kw["dim"] = f.dim
    if f.dtype == "ARRAY":
        if f.element_type is None:
            raise ValueError(f"ARRAY field '{f.name}' requires element_type")
        kw["element_type"] = _DT_MAP[f.element_type]
        if f.max_capacity is not None:
            kw["max_capacity"] = f.max_capacity
        if f.max_length is not None:
            kw["max_length"] = f.max_length
    schema.add_field(f.name, datatype=dt, **kw)

def build_schema(client: MilvusClient, cfg: AppCfg):
    sc = client.create_schema(
        auto_id=cfg.milvus.schema_.auto_id,
        enable_dynamic_field=cfg.milvus.schema_.enable_dynamic_field
    )
    for f in cfg.milvus.schema_.fields:
        _add_field(sc, f)
    return sc

def ensure_collection(client: MilvusClient, cfg: AppCfg, index_params):
    coll = cfg.milvus.collection
    name = coll.name
    if coll.recreate_if_exists and client.has_collection(name):
        client.drop_collection(name)
    if not client.has_collection(name):
        schema = build_schema(client, cfg)
        client.create_collection(
            collection_name=name,
            schema=schema,
            index_params=index_params,
            num_partitions=coll.num_partitions,
            properties=coll.properties or {}
        )
    if coll.load_on_start:
        client.load_collection(name)
