"""关系型存储：配置了 postgres.host 用 PostgreSQL，否则降级为 SQLite（D18）

只封装两边都支持的最小接口，给 M1 的 KnowledgeStore（文档与状态、Claim、概念、治理事件、任务）使用：

    db = SqlDatabase(cfg.postgres)
    await db.execute("CREATE TABLE IF NOT EXISTS doc_status (doc_id TEXT PRIMARY KEY, state TEXT)")
    await db.execute("INSERT INTO doc_status VALUES (?, ?)", ("d1", "pending"))
    rows = await db.fetchall("SELECT * FROM doc_status WHERE state = ?", ("pending",))   # list[dict]
    await db.close()

约定：
- SQL 统一用 `?` 占位符，PostgreSQL 下自动换成 `%s`；因此 SQL 文本里不要出现字面量的 `?`，参数一律走占位符。
- 只用两种数据库都支持的语法（CREATE TABLE IF NOT EXISTS、INSERT ... ON CONFLICT ... DO UPDATE 等）。
- 每条语句自动提交。
SQLite 只有同步驱动，按"同步 IO 走 asyncio.to_thread"的约定在线程里执行，并用锁串行化同一连接上的访问。
"""
from __future__ import annotations

import asyncio
import sqlite3
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence

from ..config.models import PostgresConfig
from .backends import sqlite_path

Params = Sequence[Any]


class SqlDatabase:
    def __init__(self, cfg: PostgresConfig):
        self.cfg = cfg
        self.backend: Literal["postgres", "sqlite"] = "sqlite" if cfg.is_sqlite else "postgres"
        self._pg = None                      # psycopg.AsyncConnection，懒创建（必须在事件循环里）
        self._sqlite: Optional[sqlite3.Connection] = None
        self._lock = asyncio.Lock()

    # ---------- 连接 ----------
    async def _pg_conn(self):
        if self._pg is None:
            import psycopg
            from psycopg.rows import dict_row
            self._pg = await psycopg.AsyncConnection.connect(self.cfg.dsn(), autocommit=True, row_factory=dict_row)
        return self._pg

    def _sqlite_conn(self) -> sqlite3.Connection:
        if self._sqlite is None:
            # isolation_level=None 即自动提交；check_same_thread=False 因为会在 to_thread 的不同线程里使用
            conn = sqlite3.connect(sqlite_path(self.cfg), isolation_level=None, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            self._sqlite = conn
        return self._sqlite

    @staticmethod
    def _to_pg(sql: str) -> str:
        return sql.replace("%", "%%").replace("?", "%s")

    # ---------- 执行 ----------
    async def execute(self, sql: str, params: Params = ()) -> int:
        """执行一条语句，返回受影响的行数"""
        if self.backend == "postgres":
            conn = await self._pg_conn()
            async with conn.cursor() as cur:
                await cur.execute(self._to_pg(sql), params)
                return cur.rowcount
        async with self._lock:
            return await asyncio.to_thread(lambda: self._sqlite_conn().execute(sql, params).rowcount)

    async def executemany(self, sql: str, seq_of_params: Iterable[Params]) -> None:
        seq = list(seq_of_params)
        if self.backend == "postgres":
            conn = await self._pg_conn()
            async with conn.cursor() as cur:
                await cur.executemany(self._to_pg(sql), seq)
            return
        async with self._lock:
            await asyncio.to_thread(lambda: self._sqlite_conn().executemany(sql, seq))

    async def fetchall(self, sql: str, params: Params = ()) -> List[Dict[str, Any]]:
        if self.backend == "postgres":
            conn = await self._pg_conn()
            async with conn.cursor() as cur:
                await cur.execute(self._to_pg(sql), params)
                return list(await cur.fetchall())
        async with self._lock:
            rows = await asyncio.to_thread(lambda: self._sqlite_conn().execute(sql, params).fetchall())
        return [dict(r) for r in rows]

    async def close(self) -> None:
        if self._pg is not None:
            await self._pg.close()
            self._pg = None
        if self._sqlite is not None:
            conn, self._sqlite = self._sqlite, None
            await asyncio.to_thread(conn.close)
