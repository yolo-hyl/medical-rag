from datasets import load_dataset
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_ollama import OllamaLLM
import re

SYS_PROMPT = """你是一个乐于助人的人工智能助手"""
PROMPT = """请根据给定的QA问答对，对Q和A进行改写，但是又不偏离原意。
# 要求
1. 使用json格式输出改写后的QA问答对，请勿输出其他无用信息
2. 改写不偏离问题原意的情况下，能够对问题进行精简或者问题模糊处理，用以评测RAG系统性能。
3. 尊重事实，不要胡编乱造
# 例子
## 输入
Q：突然晚上胃痛怎么办？
A：1. 观察症状，如果是隐隐作痛、灼烧感，可能和胃炎、胃酸过多、消化不良有关。如果是剧烈、持续的疼痛，甚至放射到背部、伴随冷汗、呕吐、黑便或吐血，需要立即就医。
2. 立即处理措施，暂时停止进食：不要再吃刺激性或难消化的食物。温热敷：可以用热水袋或暖宝宝敷在上腹部，有助于缓解胃痉挛。喝温水：小口慢饮，不要一次大量。避免平躺：可半坐位或抬高上半身，减少胃酸反流。
3. 药物对症（如家里有备药）胃酸过多、烧心：可服用抗酸药（如铝碳酸镁片、奥美拉唑等）。消化不良、胀气：可服用助消化药（如酶制剂、莫沙必利等）。注意：按说明书使用，若不确定不要随意吃药。
4. 何时要立即就医 ，剧烈难忍的持续性胃痛，疼痛伴随呕血或黑便，突然冷汗、面色苍白、晕厥感。有心脏病史，怀疑可能是心绞痛而非胃痛
## 输出
{{
    "question": "现在是晚上7点钟，刚从外面散步回来突然觉得自己肚子疼，怎么办？"
    "answer": "1. 症状观察：若出现隐隐作痛、灼烧感，常提示胃炎、胃酸分泌过多或功能性消化不良等可能。若为剧烈且持续的疼痛，疼痛可放射至背部，并伴随冷汗、呕吐、黑便或呕血，应高度警惕，需立即就医。
2. 初步处理措施：暂停进食：避免摄入辛辣、油腻及难以消化的食物。温热敷：在上腹部放置热水袋或暖贴，可帮助缓解胃部痉挛。温水小口饮用：少量多次饮用温水，避免一次性大量饮水。避免平卧：采取半坐位或抬高上半身体位，以减少胃酸反流。
3. 药物对症处理（如家中备用且使用得当）：胃酸过多、烧心：可选用抗酸药物（如铝碳酸镁片、奥美拉唑等）。消化不良、腹胀：可使用助消化药物（如酶制剂、莫沙必利等）。注意事项：严格按照说明书使用。如不确定适应症或用法，应避免随意用药。
4. 需立即就医的情况：剧烈、持续且难以缓解的胃痛。胃痛伴随呕血、黑便。突然出现冷汗、面色苍白、晕厥感。既往有心脏病史，需警惕心绞痛等心源性疼痛的可能。"
}}
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
            base_url="http://172.16.40.51:11434",
            temperature=0.3,
            num_ctx=8192,
        )
    return llm

def remove_think_blocks_and_get_qa(text: str) -> tuple[str, str]:
    import json
    import re
    
    # 删除 <think> ... </think> 之间的所有内容（含标签本身）
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    
    # 尝试提取JSON内容
    try:
        # 寻找JSON格式的内容，可能被其他文字包围
        json_match = re.search(r'\{.*?\}', text, flags=re.DOTALL)
        if json_match:
            json_str = json_match.group()
            qa_data = json.loads(json_str)
            return qa_data.get("question", ""), qa_data.get("answer", "")
        else:
            # 如果没找到JSON格式，尝试直接解析整个文本
            qa_data = json.loads(text)
            return qa_data.get("question", ""), qa_data.get("answer", "")
    except json.JSONDecodeError:
        # JSON解析失败时的容错处理
        print(f"JSON解析失败，原文本: {text}")
        return "", ""

def change_question(item):
    model = get_llm()
    messages = [
        SystemMessage(content=SYS_PROMPT),
        HumanMessage(content=PROMPT.format(question=item["question"], answer=item["answer"]))
    ]
    result = model.invoke(messages)
    item["new_question"], item["new_answer"] = remove_think_blocks_and_get_qa(result)
    return item

def main():
    data = load_dataset("json", data_files="../qa_50000.jsonl", split="train")
    data = data.shuffle().select(range(200))
    data = data.map(change_question, num_proc=8)
    data = data.remove_columns(["text"])
    # 删除解析失败的数据
    data = data.filter(lambda ex: (ex.get("new_question") or "") != "")
    data = data.filter(lambda ex: (ex.get("new_answer") or "") != "")
    data.to_json("./new_qa_200.jsonl", orient="records", force_ascii=False)
    
if __name__ == "__main__":
    main()