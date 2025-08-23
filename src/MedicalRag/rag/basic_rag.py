# src/medical_rag/rag/basic_rag.py
"""
RAG功能模块（使用langchain标准组件）
"""
import logging
from typing import List, Dict, Any, Optional, Union
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from ..config.models import AppConfig
from ..core.components import KnowledgeBase
from ..prompts.templates import get_prompt_template

logger = logging.getLogger(__name__)

class BasicRAG:
    """基础RAG系统"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.knowledge_base = KnowledgeBase(config)
        self.retriever = self.knowledge_base.as_retriever()
        
        # 获取提示模板
        template = get_prompt_template("basic_rag")
        if isinstance(template, dict):
            self.prompt = ChatPromptTemplate.from_messages([
                ("system", template["system"]),
                ("human", template["user"])
            ])
        else:
            self.prompt = ChatPromptTemplate.from_template(template)
        
        # 创建RAG链
        self._setup_chain()
    
    def _setup_chain(self):
        """设置RAG链"""
        # 创建文档组合链
        document_chain = create_stuff_documents_chain(
            self.knowledge_base.llm,
            self.prompt
        )
        
        # 创建检索链
        self.rag_chain = create_retrieval_chain(
            self.retriever,
            document_chain
        )
    
    def format_context(self, docs: List[Any]) -> str:
        """格式化上下文"""
        formatted_docs = []
        for doc in docs:
            if hasattr(doc, 'page_content'):
                formatted_docs.append(doc.page_content)
            else:
                formatted_docs.append(str(doc))
        return "\n\n".join(formatted_docs)
    
    def answer(
        self, 
        query: str, 
        filters: Optional[Dict[str, Any]] = None,
        return_context: bool = False
    ) -> Union[str, Dict[str, Any]]:
        """回答问题"""
        logger.info(f"处理查询: {query}")
        
        # 更新检索器的过滤条件
        if filters:
            self.retriever = self.knowledge_base.as_retriever(
                search_kwargs={"filter": filters, "k": self.config.search.top_k}
            )
            self._setup_chain()  # 重新设置链
        
        try:
            # 使用langchain的retrieval chain
            result = self.rag_chain.invoke({"input": query})
            
            answer = result.get("answer", "抱歉，我无法回答您的问题。")
            context = result.get("context", [])
            
            if return_context:
                return {
                    "answer": answer,
                    "context": [{"content": doc.page_content, "metadata": doc.metadata} 
                              for doc in context],
                    "query": query
                }
            
            return answer
            
        except Exception as e:
            logger.error(f"RAG处理失败: {e}")
            error_msg = "抱歉，处理您的问题时出现错误。请稍后再试。"
            if return_context:
                return {"answer": error_msg, "context": [], "query": query}
            return error_msg

class AgentRAG:
    """智能体RAG系统"""
    
    def __init__(self, config: AppConfig, enable_web_search: bool = True):
        self.config = config
        self.basic_rag = BasicRAG(config)
        self.enable_web_search = enable_web_search
        self.conversation_history = []
        
        # 设置网络搜索工具
        if enable_web_search:
            try:
                self.web_search = DuckDuckGoSearchRun()
                logger.info("网络搜索工具已启用")
            except Exception as e:
                logger.warning(f"网络搜索工具初始化失败: {e}")
                self.web_search = None
        else:
            self.web_search = None
    
    def _should_use_web_search(self, kb_results: List[Any]) -> bool:
        """判断是否需要网络搜索"""
        # 如果知识库结果为空或质量不高，使用网络搜索
        if not kb_results:
            return True
        
        # 可以根据结果的相似度分数等判断质量
        # 这里简化为检查结果数量
        return len(kb_results) < 2
    
    def _web_search_query(self, query: str) -> str:
        """执行网络搜索"""
        if not self.web_search:
            return ""
        
        try:
            # 构造医疗相关的搜索查询
            medical_query = f"医学 医疗 {query}"
            results = self.web_search.run(medical_query)
            return f"网络搜索结果：\n{results[:1000]}..."  # 限制长度
        except Exception as e:
            logger.error(f"网络搜索失败: {e}")
            return ""
    
    def _analyze_query_intent(self, query: str) -> Dict[str, Any]:
        """分析查询意图"""
        # 简化的意图分析
        intent = {
            "search_terms": [query],
            "use_filters": {},
            "priority": "knowledge_base"  # 默认优先知识库
        }
        
        # 可以基于关键词或LLM进行更复杂的意图分析
        medical_keywords = ["症状", "治疗", "药物", "疾病", "诊断", "检查"]
        if any(keyword in query for keyword in medical_keywords):
            intent["priority"] = "knowledge_base"
        
        return intent
    
    def answer(
        self, 
        query: str,
        max_iterations: int = 2,
        return_details: bool = False
    ) -> Union[str, Dict[str, Any]]:
        """智能体回答"""
        logger.info(f"智能体处理查询: {query}")
        
        search_history = []
        final_answer = ""
        
        try:
            # 1. 分析查询意图
            intent = self._analyze_query_intent(query)
            search_history.append({"step": "intent_analysis", "result": intent})
            
            # 2. 首先尝试知识库搜索
            kb_result = self.basic_rag.answer(query, return_context=True)
            kb_contexts = kb_result.get("context", [])
            search_history.append({
                "step": "knowledge_base_search", 
                "result": f"找到 {len(kb_contexts)} 个相关文档"
            })
            
            # 3. 判断知识库结果是否充分
            if not self._should_use_web_search(kb_contexts):
                # 知识库结果充分
                final_answer = kb_result["answer"]
                search_history.append({"step": "decision", "result": "使用知识库结果"})
            
            elif self.web_search and max_iterations > 1:
                # 4. 使用网络搜索补充
                web_results = self._web_search_query(query)
                search_history.append({
                    "step": "web_search",
                    "result": f"获得网络搜索结果 {len(web_results)} 字符"
                })
                
                # 5. 综合两个结果
                if web_results:
                    # 构建综合提示
                    combined_context = ""
                    if kb_contexts:
                        kb_text = "\n".join([ctx["content"] for ctx in kb_contexts[:3]])
                        combined_context += f"知识库信息：\n{kb_text}\n\n"
                    combined_context += f"网络搜索信息：\n{web_results}"
                    
                    # 使用LLM综合信息
                    synthesis_prompt = f"""
基于以下信息回答问题：{query}

{combined_context}

请综合以上信息，提供准确、专业的医学建议。如果信息不足或存在矛盾，请如实说明。
"""
                    try:
                        response = self.basic_rag.knowledge_base.llm.invoke([
                            {"role": "user", "content": synthesis_prompt}
                        ])
                        final_answer = response.content
                        search_history.append({"step": "synthesis", "result": "已综合多源信息"})
                    except Exception as e:
                        logger.error(f"信息综合失败: {e}")
                        final_answer = kb_result["answer"]
                else:
                    final_answer = kb_result["answer"]
            else:
                # 无网络搜索或达到迭代上限
                final_answer = kb_result["answer"]
                if not final_answer or "无法" in final_answer:
                    final_answer += "\n\n建议您咨询专业医生或查阅权威医疗资源获取更准确的信息。"
        
        except Exception as e:
            logger.error(f"智能体处理失败: {e}")
            final_answer = "抱歉，处理您的问题时出现错误。建议咨询专业医生。"
            search_history.append({"step": "error", "result": str(e)})
        
        # 更新对话历史
        self.conversation_history.append({"query": query, "answer": final_answer})
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]
        
        if return_details:
            return {
                "answer": final_answer,
                "query": query,
                "search_history": search_history,
                "iterations": len([h for h in search_history if h["step"] in ["knowledge_base_search", "web_search"]])
            }
        
        return final_answer

    def clear_history(self):
        """清空对话历史"""
        self.conversation_history = []

# 便捷函数
def create_basic_rag(config: AppConfig) -> BasicRAG:
    """创建基础RAG"""
    return BasicRAG(config)

def create_agent_rag(config: AppConfig, enable_web_search: bool = True) -> AgentRAG:
    """创建智能体RAG"""
    return AgentRAG(config, enable_web_search)