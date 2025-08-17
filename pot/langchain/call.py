import json
import asyncio
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
async def simple_qa_annotation():
    """简化的QA标注函数"""
    
    # 1. 初始化ollama模型
    llm = OllamaLLM(
        model="qwen3:32b",
        temperature=0.1
    )
    
    # 2. 创建分类提示
    prompt = PromptTemplate(
        input_variables=["question", "answer"],
        template="""
你是医学文本分类专家。请为以下QA对进行分类：

问题: {question}
答案: {answer}

科室分类(0-5)：
0-内科系统 1-外科系统 2-妇产儿科 3-五官感官 4-肿瘤影像 5-急诊综合

问题类别(0-7)：
0-疾病诊断与症状类 1-治疗方案类 2-药物与用药安全类 3-检查与化验类 4-预防保健类 5-特殊人群健康类 6-紧急情况与急救类 7-医学知识与科普类

请直接输出格式：
```json
{{
    "departments": [index1, index2...],
    "categories": [index1, index2...]
}}
```
## 输出示例：
```json
{{
    "departments": [0],
    "categories": [0, 2]
}}
```
"""
    )
    
    # 3. 原始数据
    raw_data = {
        "questions": [["口干的治疗方案是什么?", "请描述口干的治疗方案"]],
        "answers": ["口干症的治疗包括病因治疗和对症治疗。对因治疗在明确病因的情况下是最有效的，如药物性口干，通过调整药物及其剂量，可缓解口干。对唾液消耗增加而产生的口干，可通过消除张口呼吸等原因来解决。如果是由于唾液腺实质破坏所引起的口感，如头颈部恶性肿瘤放疗后、舍格伦综合征，目前主要通过对症治疗来缓解口干，减少并发症。"]
    }
    
    # 4. 处理数据
    questions = []
    for q_group in raw_data["questions"]:
        questions.extend(q_group)
    
    answers = raw_data["answers"]
    
    # 5. 执行分类
    results = []
    for i, question in enumerate(questions):
        answer = answers[min(i, len(answers)-1)]
        
        # 调用LLM进行分类
        formatted_prompt = prompt.format(question=question, answer=answer)
        classification_result = await llm.agenerate([formatted_prompt])
        
        # 解析结果
        response_text = classification_result.generations[0][0].text
        print(f"\n问题 {i+1}: {question}")
        print(f"LLM分类结果: {response_text}")
        
        # 简单解析（实际使用中可以更复杂）
        import re
        json_pattern = r'```json\s*(.*?)\s*```'
        json_match = re.search(json_pattern, response_text, re.DOTALL)
        
        if json_match:
            json_str = json_match.group(1)
        else:
            # 如果没有代码块，尝试直接解析
            json_str = response_text.strip()
            
        try:
            parsed_data: dict = json.loads(json_str)
        except (json.JSONDecodeError, ValueError) as e:            
            raise ValueError(f"Failed to parse structured output: {e}")
        # 构建结果
        result = {
            "question": question[:512],
            "answer": answer[:512],
            "departments": parsed_data.get("departments"),
            "categories": parsed_data.get("categories")
        }
        
        results.append(result)
    
    # 6. 保存结果
    with open("simple_annotated_data.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 完成！共处理 {len(results)} 个QA对")
    print("结果已保存到: simple_annotated_data.json")
    
    return results

# 运行示例
if __name__ == "__main__":
    print("🚀 开始简化QA标注...")
    print("请确保ollama服务正在运行且qwen3:32b模型已下载")
    
    try:
        results = asyncio.run(simple_qa_annotation())
        
        # 显示最终结果
        print("\n📋 最终标注结果:")
        for i, result in enumerate(results):
            print(f"\n{i+1}. {result['question']}")
            print(f"   科室分类: {result['departments']}")
            print(f"   问题类别: {result['categories']}")
            
    except Exception as e:
        print(f"❌ 错误: {e}")
        print("请检查ollama服务状态和模型是否可用")