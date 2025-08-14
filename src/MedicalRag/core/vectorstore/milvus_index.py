"""
索引创建与参数管理（HNSW/IVF_PQ）
"""
# milvus/milvus_index.py
from __future__ import annotations
from pymilvus import MilvusClient
from ...config.milvus_cfg import AppCfg

def build_index_params(client: MilvusClient, cfg: AppCfg):
    ip = client.prepare_index_params()
    for field, icfg in (cfg.milvus.index.by_field or {}).items():
        ip.add_index(
            field_name=field,
            index_type=icfg.index_type,
            metric_type=icfg.metric_type,
            params=icfg.params or {}
        )
    return ip

def create_or_update_indexes(client: MilvusClient, cfg: AppCfg):
    if not cfg.milvus.index.build_on_create:
        ip = build_index_params(client, cfg)
        client.create_index(
            collection_name=cfg.milvus.collection.name,
            index_params=ip
        )
