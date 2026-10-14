import ast
import asyncio
import json
import logging
import operator as op
from typing import Callable, List

from langchain_core.documents import Document
from langchain_core.tools import tool

from ....config.models import SearchRequest
from ....storage.milvus import MedicalHybridKnowledgeBase

logger = logging.getLogger(__name__)

WebSearchFunc = Callable[[str, int], List[Document]]


class AgentTools:
    """Agent 可调用的工具。

    本地检索工具只接收查询文本和返回条数（D10）：检索字段、距离指标、过滤表达式、collection
    都由调用方配置的 search_config 决定，不交给 LLM。v2 让 LLM 直接填 SearchRequest，
    常出现 collection 填错、给 Milvus 托管的 BM25 配 IP 指标等错误，取不到数据。
    """

    def __init__(
        self,
        kb: MedicalHybridKnowledgeBase,
        search_config: SearchRequest,
        network_search_cnt: int,
        max_top_k: int = 10,
    ) -> None:
        self.kb = kb
        self.search_config = search_config
        self.network_search_cnt = network_search_cnt
        self.max_top_k = max_top_k
        self.websearch_func: WebSearchFunc | None = None

    def register_websearch(self, func: WebSearchFunc) -> None:
        self.websearch_func = func

    def make_web_search_tool(self):
        if self.websearch_func is None:
            raise RuntimeError("未注册网络检索工具")

        cnt = self.network_search_cnt
        func = self.websearch_func

        @tool("web_search")
        async def web_search(query: str) -> str:
            """使用输入文本进行联网搜索"""
            # 腾讯云搜索只有同步 SDK，放到线程里执行，避免阻塞事件循环
            results: List[Document] = await asyncio.to_thread(func, query, cnt)
            return json.dumps([d.model_dump() for d in results], ensure_ascii=False)

        return web_search

    def make_database_search_tool(self):
        base = self.search_config
        max_top_k = self.max_top_k

        @tool("database_search")
        async def database_search(query: str, top_k: int = 5) -> str:
            """在本地医学知识库中检索与查询相关的资料。

            Args:
                query: 独立、意图明确、便于检索的查询文本
                top_k: 返回的资料条数，1~10
            """
            req = base.model_copy(update={"query": query, "limit": max(1, min(top_k, max_top_k))})
            results = await self.kb.search(req=req)
            return json.dumps([d.model_dump() for d in results], ensure_ascii=False)

        return database_search

    def make_calculator_tool(self):
        operators = {
            ast.Add: op.add,
            ast.Sub: op.sub,
            ast.Mult: op.mul,
            ast.Div: op.truediv,
            ast.USub: op.neg,
        }

        def eval_expr(expr: str) -> float:
            """只支持四则运算的安全求值，不执行任意代码"""
            def _eval(node):
                if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                    return node.value
                if isinstance(node, ast.BinOp):
                    return operators[type(node.op)](_eval(node.left), _eval(node.right))
                if isinstance(node, ast.UnaryOp):
                    return operators[type(node.op)](_eval(node.operand))
                raise ValueError("不支持的表达式")

            return _eval(ast.parse(expr, mode="eval").body)

        @tool("calculator")
        def calculator(expression: str) -> str:
            """计算器：输入一个算术表达式字符串，返回结果。支持 + - * /"""
            try:
                return str(eval_expr(expression))
            except Exception as e:
                return f"计算出错: {e}"

        return calculator
