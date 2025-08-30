from MedicalRag.agent.SearchGraph import SearchGraph
import logging
from MedicalRag.agent.tools import tencent_cloud_search
from MedicalRag.config.loader import ConfigLoader
from langchain_community.chat_models.tongyi import ChatTongyi

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

if __name__ == "__main__":
    config_manager = ConfigLoader()
    config_manager.change({
        "llm.model":"qwen3:32b",
        "agent.network_search_cnt": 5
    })
    power_model = ChatTongyi(model="qwen-plus", temperature=0.1)
    graph = SearchGraph(config_manager.config, power_model=power_model, websearch_func=tencent_cloud_search)
    result = graph.answer("腹部疼痛的临床诊断")
    print(result)
