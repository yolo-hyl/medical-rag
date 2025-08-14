"""
批量插入、upsert、delete、compaction
"""
from __future__ import annotations
from typing import List, Dict, Any, Iterable
from pymilvus import MilvusClient
from ...config.milvus_cfg import AppCfg

def _chunks(seq: Iterable[Any], size: int):
    buf = []
    for x in seq:
        buf.append(x)
        if len(buf) >= size:
            yield buf
            buf = []
    if buf:
        yield buf

def insert_rows(client: MilvusClient, cfg: AppCfg, rows: List[Dict[str, Any]]):
    name = cfg.milvus.collection.name
    for batch in _chunks(rows, cfg.milvus.write.batch_size):
        _ = client.insert(collection_name=name, data=batch)
    if cfg.milvus.write.auto_load_after_write:
        client.load_collection(name)

def upsert_rows(client: MilvusClient, cfg: AppCfg, rows: List[Dict[str, Any]]):
    name = cfg.milvus.collection.name
    if getattr(client, "upsert", None):
        for batch in _chunks(rows, cfg.milvus.write.batch_size):
            _ = client.upsert(collection_name=name, data=batch)
    else:
        # 兼容老版本：简单实现为 insert（需确保 PK 唯一）
        insert_rows(client, cfg, rows)
    if cfg.milvus.write.auto_load_after_write:
        client.load_collection(name)

def delete_by_ids(client: MilvusClient, cfg: AppCfg, ids: List[str]):
    name = cfg.milvus.collection.name
    id_field = cfg.milvus.write.id_field
    id_list = ", ".join(f'"{i}"' for i in ids)
    expr = f'{id_field} in [{id_list}]'
    client.delete(collection_name=name, expr=expr)

def compact(client: MilvusClient, cfg: AppCfg):
    if not cfg.milvus.write.compaction.enabled:
        return
    client.compact(collection_name=cfg.milvus.collection.name)
