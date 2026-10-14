"""Milvus 分批写入 / 删除（带可选进度条与进度回调）"""
from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional

from pymilvus import AsyncMilvusClient
from tqdm import tqdm

ProgressFn = Callable[[int, int], None]   # (已完成, 总数)

DEFAULT_WRITE_BATCH = 20
DEFAULT_DELETE_BATCH = 500


def _chunks(seq: Iterable[Any], size: int):
    buf = []
    for x in seq:
        buf.append(x)
        if len(buf) >= max(1, size):
            yield buf
            buf = []
    if buf:
        yield buf


class _Progress:
    def __init__(self, total: int, desc: str, unit: str, show: bool, progress_fn: Optional[ProgressFn]):
        self.done, self.total = 0, total
        self.bar = tqdm(total=total, desc=desc, unit=unit) if show else None
        self.progress_fn = progress_fn

    def update(self, inc: int) -> None:
        self.done += inc
        if self.bar is not None:
            self.bar.update(inc)
        if self.progress_fn:
            try:
                self.progress_fn(self.done, self.total)
            except Exception:
                pass  # 回调是给调用方看进度的，不能影响写入

    def close(self) -> None:
        if self.bar is not None:
            self.bar.close()


async def insert_rows(
    client: AsyncMilvusClient,
    collection_name: str,
    rows: List[Dict[str, Any]],
    *,
    batch_size: int = DEFAULT_WRITE_BATCH,
    show_progress: bool = False,
    progress_fn: Optional[ProgressFn] = None,
) -> None:
    """带主键（pk）的行走 upsert，重复写入会覆盖；没有主键的行（auto_id）走 insert"""
    if not rows:
        return
    upsert = "pk" in rows[0]
    progress = _Progress(len(rows), "Milvus upsert" if upsert else "Milvus insert", "row", show_progress, progress_fn)
    try:
        for batch in _chunks(rows, batch_size):
            if upsert:
                await client.upsert(collection_name=collection_name, data=batch)
            else:
                await client.insert(collection_name=collection_name, data=batch)
            progress.update(len(batch))
    finally:
        progress.close()


async def delete_by_ids(
    client: AsyncMilvusClient,
    collection_name: str,
    ids: List[str],
    id_field: str,
    *,
    batch_size: int = DEFAULT_DELETE_BATCH,
    show_progress: bool = False,
    progress_fn: Optional[ProgressFn] = None,
) -> None:
    """按主键分批删除。v2 每批只删 1 条，大批量删除时请求数等于 id 数，这里改为可配置的批大小"""
    progress = _Progress(len(ids), "Milvus delete", "id", show_progress, progress_fn)
    try:
        for batch in _chunks(ids, batch_size):
            id_list = ", ".join(f'"{i}"' for i in batch)
            await client.delete(collection_name=collection_name, filter=f"{id_field} in [{id_list}]")
            progress.update(len(batch))
    finally:
        progress.close()
