from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, END
from MedicalRag.config.loader import ConfigLoader
from MedicalRag.agent.tools import AgentTools
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage, ToolMessage
from typing import TypedDict, List
from langchain_community.chat_models.tongyi import ChatTongyi
from typing import Any, Dict, List, Optional, Union
from langchain_core.documents import Document
import re, json
from langchain_core.language_models.chat_models import BaseChatModel
from MedicalRag.prompts.templates import get_prompt_template
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnablePassthrough, RunnableParallel, RunnableLambda, RunnableMap
)
from functools import partial
from .tools import tencent_cloud_search
import logging
from ..core.utils import create_llm_client
from ..config.models import AppConfig
from copy import deepcopy

logger = logging.getLogger(__name__)

def del_think(text):
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()

def json_to_list_document(text):
    return [Document(**d) for d in json.loads(text)]

def format_document_str(documents: List[Document]) -> str:
    parts = []
    for i, d in enumerate(reversed(documents)):
        if i >= 6:  # 做一个简单的截断
            break
        parts.append(f"## 文档{i+1}：\n{d.page_content}\n")
    return "".join(parts)
    

# ===================== 状态结构 =====================
class SearchMessagesState(TypedDict, total=False):
    query: str
    main_messages: List[Union[HumanMessage, AIMessage]]
    other_messages: List[BaseMessage]
    docs: List[Document]
    summary: str
    retry: int              # 剩余可重试次数
    final: str              # 最终输出
    judge_result: str
    
    
class NetworkSearchResult(BaseModel):
    need_search: bool = Field(description="是否需要进行网络搜索")
    search_query: str = Field(description="网络搜索查询词", default="")
    remain_doc_index: List[int] = Field(description="保留的文档索引列表", default=[])


def _should_call_tool(last_ai: BaseMessage) -> bool:
    """ 判断上一步是否触发了工具 """
    return bool(getattr(last_ai, "tool_calls", None))

def llm_db_search(
    state: SearchMessagesState, 
    llm: BaseChatModel,
    db_tool_node: ToolNode,
    show_debug: bool
) -> SearchMessagesState:
    """ DB 检索节点 可能出发db_tool """
    query = state["query"]
    db_ai = llm.invoke([
        SystemMessage(content=get_prompt_template("call_db")["system"]),
        HumanMessage(content=get_prompt_template("call_db")["user"].format(query=query))
    ])
    state["other_messages"].append(db_ai)
    if _should_call_tool(db_ai):
        if show_debug:
            logger.info(f"开始db检索，检索参数：{db_ai.additional_kwargs['tool_calls'][0]['function']['arguments']}")
        tool_msgs : ToolMessage = db_tool_node.invoke([db_ai])
        state["other_messages"].append(tool_msgs)
        state["docs"].extend(json_to_list_document(tool_msgs[0].content))
        if show_debug:
            if len(state["docs"]) >= 2:
                logger.info(f"部分示例（共{len(state['docs'])}条）：\n\n{state['docs'][0].page_content[:200]}...\n\n{state['docs'][1].page_content[:200]}...")
            else:
                logger.info(f"仅检索一条数据：\n\n{state['docs'][0].page_content[:200]}")
    return state


def llm_network_search(
    state: SearchMessagesState,
    judge_llm: BaseChatModel,
    network_search_llm: BaseChatModel,
    network_tool_node: ToolNode,
    show_debug: bool
) -> SearchMessagesState:
    """ 联网检索节点  可能触发web_tool """
    if show_debug:
        logger.info(f"检查是否缺失资料需要网络搜索...")
    # 创建Pydantic解析器
    parser = PydanticOutputParser(pydantic_object=NetworkSearchResult)
    # 可选：创建容错解析器，能自动修复格式错误
    fixing_parser = OutputFixingParser.from_llm(parser=parser, llm=judge_llm)  # 使用不绑定工具的LLM
    
    # 获取格式指令并转义大括号
    format_instructions = parser.get_format_instructions().replace("{", "{{").replace("}", "}}")
    
    # 构建判断消息模板
    judge_messages = ChatPromptTemplate.from_messages([
        ("system", get_prompt_template("web_router")["system"].format(format_instructions=format_instructions)), 
        ("human", get_prompt_template("web_router")["user"])
    ])
    
    # 构建工具调用消息模板
    calling_messages = ChatPromptTemplate.from_messages([
        ("system", get_prompt_template("call_web")["system"]),
        ("human", get_prompt_template("call_web")["user"])
    ])
    
    # 步骤1：判断是否需要搜索
    judge_chain = judge_messages | judge_llm | RunnableLambda(lambda x: del_think(x.content)) | fixing_parser  # 使用不绑定工具的LLM
    
    try:
        # 执行判断链，直接得到解析后的结果
        result: NetworkSearchResult = judge_chain.invoke({
            "query": state['query'],
            "docs": format_document_str(state.get('docs', []))
        })
        if show_debug:
            logger.info(f"判断结果: {'需要网络检索' if result.need_search else '不需要网络检索'}, 检索文本：{result.search_query}")
        
        # 保存判断消息到状态（用于调试）
        judge_ai_content = f"分析结果: {result.model_dump()}"
        judge_ai = AIMessage(content=judge_ai_content)
        state["other_messages"].append(judge_ai)
        
    except Exception as e:
        logger.error(f"JSON解析错误: {e}")
        # 默认值
        result = NetworkSearchResult(need_search=False, search_query="", remain_doc_index=[])
        judge_ai = AIMessage(content=f"解析失败，使用默认值: {result.model_dump()}")
        state["other_messages"].append(judge_ai)
    
    # 步骤2：如果需要搜索，执行工具调用
    if result.need_search and result.search_query.strip():
        
        # 创建搜索链
        search_chain = calling_messages | network_search_llm
        
        # 执行搜索
        search_ai = search_chain.invoke({"search_query": result.search_query})
        state["other_messages"].append(search_ai)
        
        # 检查是否有工具调用
        if _should_call_tool(search_ai):
            tool_msgs: ToolMessage = network_tool_node.invoke([search_ai])
            state["other_messages"].append(tool_msgs)
            
            # 更新文档
            remain_doc = result.remain_doc_index
            if remain_doc:
                # 过滤有效索引，避免越界
                valid_indices = [i-1 for i in remain_doc if 0 < i <= len(state.get("docs", []))]
                state["docs"] = [state["docs"][i] for i in valid_indices]
            else:
                state["docs"] = []  # 如果没有指定保留文档，清空原文档
                
            # 添加新搜索到的文档
            state["docs"].extend(json_to_list_document(tool_msgs[0].content))
            if show_debug:
                logger.info(f"网络检索完毕")
    else:
        if show_debug:
            logger.info(f"信息完整，无需网络搜索...")
    
    return state


def rag(
    state: SearchMessagesState,
    llm: BaseChatModel,
    show_debug: bool
) -> SearchMessagesState:
    if show_debug:
        logger.info(f"开始RAG...")
    sys = get_prompt_template("basic_rag")["system"]
    user = get_prompt_template("basic_rag")["user"]

    prompt = [
        SystemMessage(content=sys),
        HumanMessage(content=user.format(
            all_document_str=format_document_str(state.get("docs", [])),
            input=state["query"]
        ))
    ]
    
    rag_ai = llm.invoke(prompt)
    rag_ai.content = del_think(rag_ai.content)
    if not isinstance(state["main_messages"][-1], AIMessage):
        # 上一轮rag生成合格
        state["main_messages"].append(rag_ai)
    else:
        # 上一轮rag生成不合格，删除上一轮的信息
        state["main_messages"].pop()
        state["main_messages"].append(rag_ai)
    state["summary"] = rag_ai.content
    return state


def judge(
    state: SearchMessagesState,
    llm: BaseChatModel,
    show_debug: bool
) -> SearchMessagesState:
    """判断节点：负责判断和修改状态"""
    if show_debug:
        logger.info(f"开始评估...")
    judge_ai = llm.invoke([
        SystemMessage(content=get_prompt_template("judge_rag")["system"]),
        HumanMessage(content=get_prompt_template("judge_rag")["user"].format(
            format_document_str=format_document_str(state.get('docs', [])),
            query=state['query'],
            summary=state.get('summary', '')
        ))
    ])
    result = del_think(judge_ai.content or "").strip().lower()
    if show_debug:
        logger.info(f"评估结果{result[:20]}")
    state["other_messages"].append(AIMessage(content=f"[JUDGE]={result}"))
    
    # 在这里修改状态
    if 'y' in result: 
        state["judge_result"] = "pass"
    else:
        retries_left = int(state.get("retry", 0)) 
        if retries_left > 0: 
            state["retry"] = retries_left - 1  # 状态修改会被保存
            state["judge_result"] = "retry"
        else: 
            state["judge_result"] = "fail"
    
    return state


class SearchGraph:
    def __init__(self, config: AppConfig, power_model: BaseChatModel, websearch_func=tencent_cloud_search) -> None:
        self.config = config
        self.agent_tools = AgentTools(self.config)
        self.agent_tools.register_websearch(websearch_func)
        self.db_search_tool = self.agent_tools.make_database_search_tool()
        self.network_search_tool = self.agent_tools.make_web_search_tool()
        
        self.db_search_llm = deepcopy(power_model).bind_tools([self.db_search_tool])
        self.network_search_llm = deepcopy(power_model).bind_tools([self.network_search_tool])
        self.llm = create_llm_client(self.config.llm)

        self.db_tool_node = ToolNode([self.db_search_tool])
        self.network_tool_node = ToolNode([self.network_search_tool])
        self.search_graph = None

    # ---------- 具有重试回路的图式构建 ----------
    def build_search_graph(self):
        """
        构建图
        """
        def judge_router(state: SearchMessagesState) -> str:
            """简单的路由函数：只读取状态，不修改"""
            return state.get("judge_result", "fail")

        def finish_success(state: SearchMessagesState) -> SearchMessagesState:
            """ 结束节点：成功输出 """
            state["final"] = (state.get("summary", "") or "").strip() or "（空）"
            return state

        def finish_fail(state: SearchMessagesState) -> SearchMessagesState:
            """ 结束节点：失败警告输出 """
            base = (state.get("summary", "") or "").strip() or "（空）"
            state["final"] = base + "\n\n（内容可能不属实）"
            return state
        
        g = StateGraph(SearchMessagesState)

        # 原子节点
        db_search_node = partial(
            llm_db_search,
            llm=self.db_search_llm,
            db_tool_node=self.db_tool_node,
            show_debug=self.config.multi_dialogue_rag.console_debug
        )
        g.add_node("db_search", db_search_node)
        network_search_node = partial(
            llm_network_search,
            judge_llm=self.llm,
            network_search_llm=self.network_search_llm,
            network_tool_node=self.network_tool_node,
            show_debug=self.config.multi_dialogue_rag.console_debug
        )
        g.add_node("web_search", network_search_node)
        rag_node = partial(
            rag,
            llm=self.llm,
            show_debug=self.config.multi_dialogue_rag.console_debug
        )
        g.add_node("rag", rag_node)
        g.add_node("finish_success", finish_success)
        g.add_node("finish_fail", finish_fail)
        judge_node = partial(
            judge,
            llm=self.llm,
            show_debug=self.config.multi_dialogue_rag.console_debug
        )
        g.add_node("judge", judge_node)
        # 入口
        g.set_entry_point("db_search")

        # db_search -> web_search
        if self.config.agent.network_search_enabled:
            g.add_edge("db_search", "web_search")
            g.add_edge("web_search", "rag")
        else:
            g.add_edge("db_search", "rag")
        
        # rag -> judge_router（条件分支）
        # judge -> 条件路由
        if self.config.agent.mode == "analysis":
            g.add_edge("rag", "judge")
            g.add_conditional_edges(
                "judge",  # 从判断节点出发
                judge_router,  # 纯路由函数
                {
                    "pass": "finish_success",
                    "retry": "rag",     
                    "fail": "finish_fail",
                }
            )
            # 结束
            g.add_edge("finish_success", END)
            g.add_edge("finish_fail", END)
        elif self.config.agent.mode == "fast":
            g.add_edge("rag", END)

        self.search_graph = g.compile()

    # ---------- 对外：跑整张图，返回最终输出 ----------
    def answer(self, query: str) -> str:
        if self.search_graph is None:
            self.build_search_graph()
        init_state: SearchMessagesState = {
            "query": query,
            "main_messages": [HumanMessage(content=query)],
            "other_messages": [],
            "docs": [],
            "summary": "",
            "retry": self.config.agent.max_attempts,
            "final": ""
        }
        # 执行图
        out_state: SearchMessagesState = self.search_graph.invoke(init_state)
        return out_state.get("final", "") or out_state.get("summary", "") or "（空）"
    
    def run(self, init_state: SearchMessagesState) -> SearchMessagesState:
        if self.search_graph is None:
            self.build_search_graph()
        out_state: SearchMessagesState = self.search_graph.invoke(init_state)
        return out_state