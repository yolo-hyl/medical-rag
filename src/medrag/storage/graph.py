"""图数据库：配置了 neo4j.uri 用 Neo4j，否则降级为 FalkorDBLite

只封装执行 Cypher 这一件事，给知识层的图投影（storage/graph_projection.py）使用：

    graph = GraphDatabase(cfg.neo4j)
    await graph.query("MERGE (d:Disease {name: $name})", {"name": "高血压"})
    rows = await graph.query("MATCH (d:Disease) RETURN d.name AS name")   # list[dict]
    await graph.close()

约定：
- 参数一律用 `$name` 形式传入，不要拼接字符串。
- 返回值统一转成纯 Python 类型：每行一个 dict（键是 RETURN 的列名），节点 / 关系只保留属性。
- FalkorDB 支持的是 openCypher 的子集，个别语法（如部分 APOC 过程、某些子查询写法）与 Neo4j 不同；
  图投影只用两边都支持的语法，并在两个后端上各跑一遍测试。
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from ..config.models import Neo4jConfig
from .backends import falkordb_path


def _to_plain(value: Any) -> Any:
    if isinstance(value, list):
        return [_to_plain(v) for v in value]
    if isinstance(value, dict):
        return {k: _to_plain(v) for k, v in value.items()}
    # FalkorDB 的 Node / Edge 对象带 properties 属性，转成属性字典，与 Neo4j record.data() 的行为一致
    props = getattr(value, "properties", None)
    if isinstance(props, dict):
        return {k: _to_plain(v) for k, v in props.items()}
    return value


class GraphDatabase:
    def __init__(self, cfg: Neo4jConfig):
        self.cfg = cfg
        self.backend: Literal["neo4j", "falkordb"] = "falkordb" if cfg.is_lite else "neo4j"
        self._driver = None    # neo4j.AsyncDriver
        self._falkor = None    # redislite.async_falkordb_client.AsyncFalkorDB
        self._graph = None

    async def _falkor_graph(self):
        if self._graph is None:
            from redislite.async_falkordb_client import AsyncFalkorDB
            self._falkor = AsyncFalkorDB(falkordb_path(self.cfg))
            self._graph = self._falkor.select_graph(self.cfg.graph_name)
        return self._graph

    def _neo4j_driver(self):
        if self._driver is None:
            from neo4j import AsyncGraphDatabase
            self._driver = AsyncGraphDatabase.driver(self.cfg.uri, auth=(self.cfg.user, self.cfg.password()))
        return self._driver

    async def query(self, cypher: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        params = params or {}
        if self.backend == "neo4j":
            async with self._neo4j_driver().session(database=self.cfg.database) as session:
                result = await session.run(cypher, params)
                return [_to_plain(row) for row in await result.data()]
        graph = await self._falkor_graph()
        result = await graph.query(cypher, params)
        columns = [h[1] for h in (result.header or [])]
        return [{c: _to_plain(v) for c, v in zip(columns, row)} for row in result.result_set]

    async def close(self) -> None:
        if self._driver is not None:
            await self._driver.close()
            self._driver = None
        if self._falkor is not None:
            client = self._falkor.client
            self._falkor = self._graph = None
            # redislite 的异步 close 会跳过关停嵌入式服务这一步，数据不会写到 falkordb_path（实测进程退出后文件不存在）。
            # 所以先显式 SAVE 落盘，再让嵌入式服务退出；服务退出时会断开连接，这里的连接错误是预期的。
            await client.execute_command("SAVE")
            try:
                await client.shutdown(nosave=True, now=True, force=True)
            except Exception:
                pass
            await client.close()
