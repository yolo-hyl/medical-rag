"""
批量插入、upsert、delete、compaction（带可选进度条）
"""
from __future__ import annotations
from typing import List, Dict, Any, Iterable, Callable, Optional
from pymilvus import MilvusClient
from ...config.default_cfg import AppCfg

# 尝试引入 tqdm；未安装则降级为无进度条
try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None


def _chunks(seq: Iterable[Any], size: int):
    buf = []
    for x in seq:
        buf.append(x)
        if len(buf) >= max(1, size):
            yield buf
            buf = []
    if buf:
        yield buf


def _maybe_update_progress(pbar, progress_fn, inc: int, done_total: list[int]):
    """
    统一进度更新：
    - pbar：tqdm 实例或 None
    - progress_fn：回调或 None，签名 (done:int, total:int)
    - inc：本次新增完成数量
    - done_total：长度为2的列表 [done, total]，以可变引用传入以便原地累加
    """
    done_total[0] += inc
    if pbar is not None:
        pbar.update(inc)
    if progress_fn:
        try:
            progress_fn(done_total[0], done_total[1])
        except Exception as _:
            # 回调不应影响主流程：忽略异常
            pass


def insert_rows(
    client: MilvusClient,
    cfg: AppCfg,
    rows: List[Dict[str, Any]],
    *,
    show_progress: bool = False,
    progress_fn: Optional[Callable[[int, int], None]] = None,
):
    name = cfg.milvus.collection.name
    bs = cfg.milvus.write.batch_size
    total = len(rows)
    done_total = [0, total]

    pbar = None
    if show_progress and tqdm is not None:
        pbar = tqdm(total=total, desc="Milvus insert", unit="row")

    try:
        for batch in _chunks(rows, bs):
            _ = client.insert(collection_name=name, data=batch)
            _maybe_update_progress(pbar, progress_fn, len(batch), done_total)
    finally:
        if pbar is not None:
            pbar.close()

    if cfg.milvus.write.auto_load_after_write:
        client.load_collection(name)


def upsert_rows(
    client: MilvusClient,
    cfg: AppCfg,
    rows: List[Dict[str, Any]],
    *,
    show_progress: bool = False,
    progress_fn: Optional[Callable[[int, int], None]] = None,
):
    name = cfg.milvus.collection.name
    bs = cfg.milvus.write.batch_size
    total = len(rows)
    done_total = [0, total]

    pbar = None
    if show_progress and tqdm is not None:
        pbar = tqdm(total=total, desc="Milvus upsert", unit="row")

    try:
        if getattr(client, "upsert", None):
            for batch in _chunks(rows, bs):
                _ = client.upsert(collection_name=name, data=batch)  # type: ignore[attr-defined]
                _maybe_update_progress(pbar, progress_fn, len(batch), done_total)
        else:
            # 兼容老版本：降级为 insert，但仍保留进度
            for batch in _chunks(rows, bs):
                _ = client.insert(collection_name=name, data=batch)
                _maybe_update_progress(pbar, progress_fn, len(batch), done_total)
    finally:
        if pbar is not None:
            pbar.close()

    if cfg.milvus.write.auto_load_after_write:
        client.load_collection(name)


def delete_by_ids(
    client: MilvusClient,
    cfg: AppCfg,
    ids: List[str],
    *,
    show_progress: bool = False,
    progress_fn: Optional[Callable[[int, int], None]] = None,
):
    """
    分批删除并显示进度。批大小复用 write.batch_size；
    如需单独控制，可在 cfg 中新增 delete_batch_size 并优先使用。
    """
    name = cfg.milvus.collection.name
    id_field = cfg.milvus.write.id_field
    bs = getattr(getattr(cfg.milvus, "write", object()), "delete_batch_size", None) or cfg.milvus.write.batch_size

    total = len(ids)
    done_total = [0, total]

    pbar = None
    if show_progress and tqdm is not None:
        pbar = tqdm(total=total, desc="Milvus delete", unit="id")

    try:
        for batch in _chunks(ids, bs):
            id_list = ", ".join(f'"{i}"' for i in batch)
            expr = f'{id_field} in [{id_list}]'
            client.delete(collection_name=name, expr=expr)
            _maybe_update_progress(pbar, progress_fn, len(batch), done_total)
    finally:
        if pbar is not None:
            pbar.close()


def compact(
    client: MilvusClient,
    cfg: AppCfg,
    *,
    show_progress: bool = False,
    progress_fn: Optional[Callable[[int, int], None]] = None,
):
    """
    触发 compaction。当前 pymilvus 的同步接口无法细粒度获知进度；
    因此这里只提供“一步完成式”的进度：0 -> 1。
    如你的 Milvus 版本支持 compaction 状态查询，可在此处扩展轮询并细化进度。
    """
    if not cfg.milvus.write.compaction.enabled:
        return

    pbar = None
    if show_progress and tqdm is not None:
        pbar = tqdm(total=1, desc="Milvus compaction", unit="step")

    try:
        client.compact(collection_name=cfg.milvus.collection.name)
        if pbar is not None:
            pbar.update(1)
        if progress_fn:
            try:
                progress_fn(1, 1)
            except Exception:
                pass
    finally:
        if pbar is not None:
            pbar.close()
