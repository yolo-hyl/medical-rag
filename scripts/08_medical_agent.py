from MedicalRag.agent.MedicalAgent import MedicalAgent
from MedicalRag.config.loader import ConfigLoader
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.messages import HumanMessage
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

if __name__ == "__main__":
    # 初始化
    config_manager = ConfigLoader()
    power_model = ChatTongyi(model="qwen-plus", temperature=0.1)
    cg = MedicalAgent(config_manager.config, power_model=power_model)
    user_input = "我这两天肚子痛，还拉肚子"
    # 运行一步
    while True:
        state = cg.answer(user_input=user_input)
        print(f"Agent: \n\n{state['asking_messages'][-1][-1].content if state['ask_obj'].need_ask else state['dialogue_messages'][-1]}\n\nUser:\n")
        user_input = input()
        print("\n")
