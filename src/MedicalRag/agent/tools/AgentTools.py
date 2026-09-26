import ast
import asyncio
import json
import logging
import operator as op
from typing import List

from langchain.tools import tool
from langchain_core.documents import Document

from ...config.models import SearchRequest
from ...core.KnowledgeBase import MedicalHybridKnowledgeBase

logger = logging.getLogger(__name__)

class AgentTools:
    def __init__(self, kb: MedicalHybridKnowledgeBase, network_search_cnt: int) -> None:
        self.kb = kb
        self.network_search_cnt = network_search_cnt
        self.WEBSEARCH_FUNC = None

    def register_websearch(self, func):
        self.WEBSEARCH_FUNC = func

    def make_web_search_tool(self):
        if self.WEBSEARCH_FUNC is None:
            raise RuntimeError("未注册网络检索工具")

        cnt = self.network_search_cnt

        @tool("web_search")
        async def web_search(query: str) -> str:
            """ 使用输入文本进行联网搜索 """
            # 腾讯云搜索只有同步 SDK，放到线程里执行，避免阻塞事件循环
            results: List[Document] = await asyncio.to_thread(self.WEBSEARCH_FUNC, query, cnt)
            return json.dumps([d.model_dump() for d in results], ensure_ascii=False)

        return web_search

    def make_database_search_tool(self):

        @tool("database_search")
        async def database_search(search_config: SearchRequest) -> str:
            """ 输入检索参数,使用向量数据库进行本地检索 """
            results = await self.kb.search(req=search_config)
            return json.dumps([d.model_dump() for d in results], ensure_ascii=False)

        return database_search

    def make_calculator_tool(self):
        # 支持的运算符
        operators = {
            ast.Add: op.add,
            ast.Sub: op.sub,
            ast.Mult: op.mul,
            ast.Div: op.truediv,
            ast.USub: op.neg,
        }

        def eval_expr(expr: str) -> float:
            """安全地计算表达式"""
            def _eval(node):
                if isinstance(node, ast.Num):  # 处理常数
                    return node.n
                elif isinstance(node, ast.BinOp):  # 二元运算
                    return operators[type(node.op)](_eval(node.left), _eval(node.right))
                elif isinstance(node, ast.UnaryOp):  # 负号等
                    return operators[type(node.op)](_eval(node.operand))
                else:
                    raise ValueError("不支持的表达式")

            node = ast.parse(expr, mode="eval").body
            return _eval(node)

        @tool("calculator")
        def calculator(expression: str) -> str:
            """计算器：输入一个算术表达式字符串，返回结果。支持 + - * /"""
            try:
                result = eval_expr(expression)
                return str(result)
            except Exception as e:
                return f"计算出错: {e}"

        return calculator
