import asyncio
import logging

from langchain_community.chat_models.tongyi import ChatTongyi

from MedicalRag.agent.SearchGraph import SearchGraph
from MedicalRag.agent.tools import tencent_cloud_search
from MedicalRag.config.loader import ConfigLoader
from MedicalRag.core.KnowledgeBase import MedicalHybridKnowledgeBase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)


async def main():
    config_manager = ConfigLoader()
    # config_manager.change({
    #     "llm.model":"qwen3:32b",
    #     "agent.network_search_cnt": 5
    # })
    cfg = config_manager.config
    power_model = ChatTongyi(model="qwen-plus", temperature=0.1)
    kb = MedicalHybridKnowledgeBase(cfg.milvus, cfg.embedding)
    graph = SearchGraph(cfg.llm, cfg.agent, kb, power_model=power_model, websearch_func=tencent_cloud_search)
    try:
        result = await graph.answer("腹部疼痛的临床诊断")
    finally:
        await kb.close()
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
