from pydantic import BaseModel, Field
from typing import Literal, TypedDict
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import StateGraph, START
from langgraph.constants import END
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
import logging
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
import re
from ..config.models import SearchRequest, AppConfig
from ..core.utils import create_llm_client
from operator import add
from typing import Annotated, List
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

class DialogueState(TypedDict):
    # 多轮对话中始终维护的一个状态
    messages: BaseMessage[list]

class InformationState(TypedDict):
    """ 智能体用以反复询问患者的信息状态,以便能够掌握全部信息用以拆解问题 """
    what: str  # 症状
    when: str  # 发生时间
    how_severe: str  # 严重程度
    who: str  # 患者基本信息
    maybe_why: str  # 诱发因素
    what_else: str  # 伴随症状
    background: str  # 既往史
    medication: str   # 用药情况
    level: int  # 信息完善等级 1 ~ 5

class Information(BaseModel):
    what: str = Field(default="", description="用户的症状")
    when: str = Field(default="", description="症状的发生时间") 
    how_severe: str = Field(default="", description="用户症状的严重程度")
    who: str = Field(default="", description="用户的基本信息")
    maybe_why: str = Field(default="", description="用户症状的可能诱发因素,用户可能自己都不知道")
    what_else: str = Field(default="", description="用户症状的伴随症状") 
    background: str = Field(default="", description="用户既往史")
    medication: str = Field(default="", description="用户的用药情况")
    level: int = Field(default=0, ge=0, le=5, description="用户的用药情况") # 信息完善等级 1 ~ 5
    
    
class SubQueryDefine(BaseModel):
    symptom: List[str] = Field(default=[""], description="症状匹配子查询")
    diagnosis: List[str] = Field(default=[""], description="诊断病症子查询")
    treatment: List[str] = Field(default=[""], description="治疗病症子查询")
    
class SearchState(TypedDict):
    messages: BaseMessage[list]

