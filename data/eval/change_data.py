from datasets import load_dataset
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_ollama import OllamaLLM
import re

SYS_PROMPT = """你是一个乐于助人的人工智能助手"""
PROMPT = """请根据给定的QA问答对，对Q进行改写，但是又不偏离原意。
# 要求
1. 只输出改写后问题，请勿输出其他信息
2. 改写不偏离问题原意的情况下，能够对问题进行扩充或者加入部分上下文，用以评测RAG系统性能。
3. 尊重事实，不要胡编乱造
# 例子
## 输入
Q：突然晚上胃痛怎么办？
A：1. 观察症状，如果是隐隐作痛、灼烧感，可能和胃炎、胃酸过多、消化不良有关。如果是剧烈、持续的疼痛，甚至放射到背部、伴随冷汗、呕吐、黑便或吐血，需要立即就医。
2. 立即处理措施，暂时停止进食：不要再吃刺激性或难消化的食物。温热敷：可以用热水袋或暖宝宝敷在上腹部，有助于缓解胃痉挛。喝温水：小口慢饮，不要一次大量。避免平躺：可半坐位或抬高上半身，减少胃酸反流。
3. 药物对症（如家里有备药）胃酸过多、烧心：可服用抗酸药（如铝碳酸镁片、奥美拉唑等）。消化不良、胀气：可服用助消化药（如酶制剂、莫沙必利等）。注意：按说明书使用，若不确定不要随意吃药。
4. 何时要立即就医 ，剧烈难忍的持续性胃痛，疼痛伴随呕血或黑便，突然冷汗、面色苍白、晕厥感。有心脏病史，怀疑可能是心绞痛而非胃痛
## 输出
现在是晚上7点钟，刚从外面散步回来突然觉得自己胃痛，我该怎么办？
# 输入
Q：{question}
A：{answer}
# 输出
"""
llm = None
def get_llm():
    global llm
    if llm is None:
        llm = OllamaLLM(
            model="qwen3:32b",
            base_url="http://127.0.0.1:11434",
            temperature=0.3,
            num_ctx=8192,
        )
    return llm

def remove_think_blocks(text: str) -> str:
    # 删除 <think> ... </think> 之间的所有内容（含标签本身）
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

def change_question(item):
    model = get_llm()
    messages = [
        SystemMessage(content=SYS_PROMPT),
        HumanMessage(content=PROMPT.format(question=item["question"], answer=item["answer"]))
    ]
    result = model.invoke(messages)
    item["new_question"] = remove_think_blocks(result)
    return item

def main():
    data = load_dataset("json", data_files="/home/weihua/medical-rag/raw_data/raw/train/sample/eval/qa_200.jsonl", split="train")
    data = data.map(change_question, num_proc=8)
    data.to_json("/home/weihua/medical-rag/raw_data/raw/train/sample/eval/new_qa_200.jsonl", orient="records", force_ascii=False)
    
if __name__ == "__main__":
    main()