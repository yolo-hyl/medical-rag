# 修改 src/MedicalRag/core/llm/EmbeddingClient.py

import asyncio
import logging
from typing import List, Dict, Any, Optional
from ...config.default_cfg import DenseEmbedCfg
from .HttpClient import OllamaClient

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
        self.prefixes = cfg.prefixes
        self._client = None
        self._closed = False

    def _get_or_create_client(self):
        """延迟创建客户端"""
        if self._client is None:
            if self._config["ollama"]["base_url"]:
                self._client = OllamaClient(self._config)
            else:
                raise ValueError("不支持的嵌入提供商")
        return self._client

    def _run_async_safe(self, coro):
        """
        安全的异步执行，每次都创建新的客户端实例避免连接复用问题
        """
        async def _safe_exec():
            # 创建临时客户端
            temp_client = OllamaClient(self._config)
            try:
                # 执行任务
                if hasattr(coro, '__name__') and coro.__name__ == 'batch_embedding':
                    # 这是 batch_embedding 调用
                    return await temp_client.batch_embedding(*coro.args, **coro.kwargs)
                else:
                    # 通用协程
                    return await coro
            finally:
                # 确保客户端关闭
                await temp_client.aclose()
        
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            raise RuntimeError("An event loop is already running; call the async API directly.")
        
        return asyncio.run(_safe_exec())

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        同步方法，每次调用都使用独立的客户端实例
        """
        if self._closed:
            raise RuntimeError("FastEmbeddings already closed.")
        
        # 创建协程任务信息
        class CoroutineTask:
            def __init__(self, texts, prefix):
                self.args = (texts, prefix)
                self.kwargs = {}
                self.__name__ = 'batch_embedding'
        
        task = CoroutineTask(texts, self.prefixes["document"])
        return self._run_async_safe(task)

    def embed_query(self, text: str) -> List[float]:
        """单个查询嵌入"""
        if self._closed:
            raise RuntimeError("FastEmbeddings already closed.")
        
        class CoroutineTask:
            def __init__(self, texts, prefix):
                self.args = (texts, prefix)
                self.kwargs = {}
                self.__name__ = 'batch_embedding'
        
        task = CoroutineTask([text], self.prefixes["query"])
        vecs = self._run_async_safe(task)
        return vecs[0]

    def close(self):
        """关闭资源"""
        self._closed = True
        # 注意：由于我们每次都创建新客户端，这里不需要关闭持久客户端

    def __del__(self):
        """防遗忘收尾"""
        try:
            self.close()
        except Exception:
            pass