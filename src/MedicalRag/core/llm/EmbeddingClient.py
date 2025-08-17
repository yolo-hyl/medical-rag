from typing import Dict, Any, List, Optional, Tuple
import asyncio
import logging
import math
import os
import json
import httpx
import asyncio
from typing import List, Dict, Any, Optional
from MedicalRag.core.base.BaseClient import LLMClient  # 你原本的接口
from ...config.default_cfg import DenseEmbedCfg
from .HttpClient import OllamaClient

def _run_async(coro):
    """
    简单的同步跑协程工具：脚本环境用 asyncio.run。
    （如果你在已有事件循环中使用，比如 Notebook，需要换成更安全的实现）
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # 如果未来需要兼容 Notebook，可引入 nest_asyncio；脚本场景不会走到这里
        raise RuntimeError("An event loop is already running; call the async API directly.")
    return asyncio.run(coro)

class FastEmbeddings:
    """
    适配器：把你的 OllamaClient 封装成类似 LangChain Embeddings 的接口。
    支持并发 batch_embedding，用于大规模加速。
    """
    def __init__(self, cfg: DenseEmbedCfg):
        self._config: Dict[str, Any] = {
            "model_name": cfg.model,
            "ollama": {
                "base_url": cfg.base_url,
                "timeout": cfg.timeout,
                "max_concurrent": cfg.max_concurrent,
                "proxy": cfg.proxy,
                "verify": cfg.verify,
                "max_retries": cfg.max_retries,
                "backoff_base": cfg.backoff_base,
                "backoff_cap": cfg.backoff_cap,
            }
        }
        if cfg.provider == "ollama":
            self._client = OllamaClient(self._config)
        elif cfg.provider == "openai":
            raise "尚未实现"
        elif cfg.provider == "local":
            raise "尚未实现"
        self._closed = False
        self.prefixes = cfg.prefixes

    # --- public API（与 LangChain Embeddings 类似） ---
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        同步方法，内部跑一次 asyncio（脚本场景 OK）。
        返回与原来 OllamaEmbeddings 相同的结构：List[List[float]]
        """
        if self._closed:
            raise RuntimeError("FastOllamaEmbeddings already closed.")
        return _run_async(self._client.batch_embedding(texts, prefix=self.prefixes["document"]))

    def embed_query(self, text: str) -> List[float]:
        """可选：如果你的 HybridRetriever 里用了 embed_query，这里也兼容。"""
        if self._closed:
            raise RuntimeError("FastOllamaEmbeddings already closed.")
        vecs = _run_async(self._client.batch_embedding([text], prefix=self.prefixes["query"]))
        return vecs[0]

    def close(self):
        if not self._closed:
            _run_async(self._client.aclose())
            self._closed = True

    def __del__(self):
        # 防遗忘收尾（尽量别依赖 __del__，还是在主流程里显式 close）
        try:
            self.close()
        except Exception:
            pass
        