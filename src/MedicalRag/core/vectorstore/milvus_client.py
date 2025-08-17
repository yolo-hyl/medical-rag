"""
连接/健康检查/集合管理（pymilvus）
"""
from __future__ import annotations
from typing import Optional
from pymilvus import MilvusClient
from ...config.default_cfg import AppCfg
import time

class MilvusConn:
    """
    负责：初始化连接、健康检查、按需创建/加载集合。
    """
    def __init__(self, cfg: AppCfg):
        self.cfg = cfg
        self.client = MilvusClient(
            uri=cfg.milvus.client.uri,
            token=cfg.milvus.client.token
        )

    # --- 健康检查 ---
    def healthy(self, timeout_sec: int = 10) -> bool:
        ddl = time.time() + timeout_sec
        while time.time() < ddl:
            try:
                _ = self.client.list_collections()
                return True
            except Exception:
                time.sleep(0.2)
        return False

    # --- 集合管理 ---
    def has_collection(self) -> bool:
        return self.client.has_collection(self.cfg.milvus.collection.name)

    def drop_collection(self):
        name = self.cfg.milvus.collection.name
        if self.client.has_collection(name):
            self.client.drop_collection(name)

    def load_collection(self):
        self.client.load_collection(self.cfg.milvus.collection.name)

    def get_client(self) -> MilvusClient:
        return self.client
