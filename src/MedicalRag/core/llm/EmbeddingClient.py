import asyncio
import logging
from typing import List, Dict, Any, Optional, Callable, Iterable, Tuple
from ...config.default_cfg import DenseEmbedCfg
from .HttpClient import OllamaClient

# 尝试引入 tqdm，用于进度条展示；未安装则自动降级为无进度条
try:
    # 在脚本/终端用 tqdm，在 notebook 用 tqdm.auto 也没问题
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


def _iter_chunks(arr: List[str], size: int) -> Iterable[Tuple[int, List[str]]]:
    """把列表切成块，返回 (起始下标, 子列表)"""
    n = len(arr)
    if size <= 0:
        size = n
    for i in range(0, n, size):
        yield i, arr[i:i + size]


class FastEmbeddings:
    """
    适配器：把你的 OllamaClient 封装成类似 LangChain Embeddings 的接口。
    支持并发 batch_embedding，用于大规模加速。
    现在新增：可选进度条/回调
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

    def _make_task(self, texts: List[str], prefix: str):
        """内部工具：构造成与 _run_async_safe 配套的“伪协程”任务"""
        class CoroutineTask:
            def __init__(self, texts, prefix):
                self.args = (texts, prefix)
                self.kwargs = {}
                self.__name__ = 'batch_embedding'
        return CoroutineTask(texts, prefix)

    def embed_documents(
        self,
        texts: List[str],
        *,
        show_progress: bool = False,
        batch_size: Optional[int] = None,
        progress_fn: Optional[Callable[[int, int], None]] = None,
    ) -> List[List[float]]:
        """
        同步方法，每次调用都使用独立的客户端实例。
        现在支持进度条/回调。默认不显示。
        参数：
          - show_progress: 是否用 tqdm 展示进度
          - batch_size: 每批次文本数量（默认 64）
          - progress_fn: 自定义回调，签名 progress_fn(done:int, total:int)
        """
        if self._closed:
            raise RuntimeError("FastEmbeddings already closed.")

        total = len(texts)
        if total == 0:
            return []

        bs = batch_size or 64
        done = 0
        all_vecs: List[List[float]] = []

        pbar = None
        if show_progress and tqdm is not None:
            pbar = tqdm(total=total, desc="Embedding documents", unit="text")

        try:
            for _, chunk in _iter_chunks(texts, bs):
                task = self._make_task(chunk, self.prefixes["document"])
                vecs = self._run_async_safe(task)
                all_vecs.extend(vecs)

                # 更新进度
                done += len(chunk)
                if pbar is not None:
                    pbar.update(len(chunk))
                if progress_fn:
                    try:
                        progress_fn(done, total)
                    except Exception as e:
                        logging.warning("progress_fn error: %s", e)
        finally:
            if pbar is not None:
                pbar.close()

        return all_vecs

    def embed_query(
        self,
        text: str,
        *,
        show_progress: bool = False,
        progress_fn: Optional[Callable[[int, int], None]] = None,
    ) -> List[float]:
        """
        单个查询嵌入；支持轻量进度显示（总量=1）。
        """
        if self._closed:
            raise RuntimeError("FastEmbeddings already closed.")

        pbar = None
        if show_progress and tqdm is not None:
            pbar = tqdm(total=1, desc="Embedding query", unit="text")

        try:
            task = self._make_task([text], self.prefixes["query"])
            vecs = self._run_async_safe(task)
            if pbar is not None:
                pbar.update(1)
            if progress_fn:
                try:
                    progress_fn(1, 1)
                except Exception as e:
                    logging.warning("progress_fn error: %s", e)
        finally:
            if pbar is not None:
                pbar.close()

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
