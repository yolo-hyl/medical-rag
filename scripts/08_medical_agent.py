"""
MedicalAgent 多轮对话演示

流程：
  用户输入 → [ask] 判断是否需要追问
    ├─ 需要追问 → 打印问题，等待用户回复 → 继续下一轮
    └─ 信息充分 → [extract] → [split_query] → [run_query（并行）] → [answer（LLM合成）]
                                                                        ↓
                                                              打印完整答案后，等待用户下一个问题
                                                              输入 "q" 或 "exit" 可退出
"""
import logging
from MedicalRag.agent.MedicalAgent import MedicalAgent
from MedicalRag.config.loader import ConfigLoader
from MedicalRag.core.utils import create_llm_client

logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

SEP = "=" * 60

if __name__ == "__main__":
    config_manager = ConfigLoader()
    power_model = create_llm_client(config_manager.config.llm)

    cg = MedicalAgent(config_manager.config, power_model=power_model)
    user_input = "我这两天肚子痛，还拉肚子"

    while True:
        state = cg.answer(user_input=user_input)
        ask_obj = state.get("ask_obj")

        if ask_obj and ask_obj.need_ask:
            # 仍在追问阶段：打印 Agent 的最后一条追问消息
            reply = state["asking_messages"][-1][-1].content
            print(f"\nAgent（需要更多信息）:\n{SEP}\n{reply}\n{SEP}\n")
        else:
            # 已完成检索与回答：打印最终答案，并提示用户可继续提问
            reply = state.get("final_answer") or ""
            if not reply and state.get("dialogue_messages"):
                reply = state["dialogue_messages"][-1].content
            print(f"\nAgent（回答完毕）:\n{SEP}\n{reply}\n{SEP}")
            print("\n可继续提问，输入 q 或 exit 退出：\n")

        user_input = input().strip()
        print()
        if user_input.lower() in ("q", "exit", "quit", "退出"):
            print("对话结束。")
            break
