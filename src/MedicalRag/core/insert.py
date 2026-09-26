"""
批量插入、upsert、delete 工具类（带可选进度条）
"""
from __future__ import annotations
from typing import List, Dict, Any, Iterable, Callable, Optional
from pymilvus import AsyncMilvusClient
from tqdm import tqdm


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


async def insert_rows(
    client: AsyncMilvusClient,
    collection_name: str,
    rows: List[Dict[str, Any]],
    show_progress: bool = False,
    progress_fn: Optional[Callable[[int, int], None]] = None,
):
    bs = 20
    total = len(rows)
    done_total = [0, total]
    if "pk" in rows[0]:
        await insert_rows_has_id(
            client,collection_name,
            rows,show_progress
        )
        return
    pbar = None
    if show_progress and tqdm is not None:
        pbar = tqdm(total=total, desc="Milvus insert", unit="row")

    try:
        for batch in _chunks(rows, bs):
            _ = await client.insert(collection_name=collection_name, data=batch)
            _maybe_update_progress(pbar, progress_fn, len(batch), done_total)
    finally:
        if pbar is not None:
            pbar.close()
            


async def insert_rows_has_id(
    client: AsyncMilvusClient,
    collection_name: str,
    rows: List[Dict[str, Any]],
    show_progress: bool = False,
    progress_fn: Optional[Callable[[int, int], None]] = None,
):
    bs = 20
    total = len(rows)
    done_total = [0, total]

    pbar = None
    if show_progress and tqdm is not None:
        pbar = tqdm(total=total, desc="Milvus upsert", unit="row")

    try:
        for batch in _chunks(rows, bs):
            _ = await client.upsert(collection_name=collection_name, data=batch)
            _maybe_update_progress(pbar, progress_fn, len(batch), done_total)
    finally:
        if pbar is not None:
            pbar.close()


async def delete_by_ids(
    client: AsyncMilvusClient,
    collection_name: str,
    ids: List[str],
    id_field: str,
    *,
    show_progress: bool = False,
    progress_fn: Optional[Callable[[int, int], None]] = None,
):
    """
    分批删除并显示进度。批大小复用 write.batch_size；
    如需单独控制，可在 cfg 中新增 delete_batch_size 并优先使用。
    """
    bs = 1

    total = len(ids)
    done_total = [0, total]

    pbar = None
    if show_progress and tqdm is not None:
        pbar = tqdm(total=total, desc="Milvus delete", unit="id")

    try:
        for batch in _chunks(ids, bs):
            id_list = ", ".join(f'"{i}"' for i in batch)
            expr = f'{id_field} in [{id_list}]'
            await client.delete(collection_name=collection_name, filter=expr)
            _maybe_update_progress(pbar, progress_fn, len(batch), done_total)
    finally:
        if pbar is not None:
            pbar.close()