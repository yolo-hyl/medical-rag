import asyncio
import logging

from _common import config_loader
from medrag.apps.agent import SearchGraph
from medrag.apps.agent.tools.tencent_search import tencent_cloud_search
from medrag.llm.langchain_adapter import create_chat_model
from medrag.storage.milvus import MedicalHybridKnowledgeBase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)


async def main():
    config_manager = config_loader()
    # config_manager.change({
    #     "llm.model":"qwen3:32b",
    #     "agent.network_search_cnt": 5
    # })
    cfg = config_manager.config
    # 负责工具调用的模型：同一个服务商下换一个模型即可
    power_model = create_chat_model(cfg.llm.model_copy(update={"model": "qwen-plus"}))
    kb = MedicalHybridKnowledgeBase(cfg.milvus, cfg.embedding)
    graph = SearchGraph(cfg.llm, cfg.agent, kb, power_model=power_model, websearch_func=tencent_cloud_search)
    try:
        result = await graph.answer("腹部疼痛的临床诊断")
    finally:
        await kb.close()
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
