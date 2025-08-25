# src/medical_rag/prompts/templates.py
"""
Prompt模板管理（保持与原项目一致）
"""
from typing import Dict, Any, Union

# =============================================================================
# 基础RAG提示模板
# =============================================================================
BASIC_RAG_SYSTEM_PROMPT = """你是一名专业的医学知识助手，能够基于提供的医学资料准确回答用户问题。"""

BASIC_RAG_USER_PROMPT = """# 规则
1. 必须严格基于提供的参考资料回答问题
2. 如果参考资料中没有相关信息，请明确说明"根据提供的资料无法回答此问题"
3. 回答要专业、准确，同时通俗易懂
4. 不要编造或推测未在资料中提及的信息
5. 如涉及具体诊疗建议，请提醒用户咨询专业医生
# 参考资料
{context}
# 用户问题
{input}
请基于以上参考资料回答用户问题。如果资料不足以回答问题，请如实说明。"""

# =============================================================================
# 数据标注提示模板（保留原项目内容）
# =============================================================================
ANNOTATION_SYSTEM_PROMPT = """你是一个专业的医疗数据标注专家，能够准确分析医疗问答内容并进行分类标注。"""

ANNOTATION_USER_PROMPT = """请对以下医疗问答进行分类标注：

问题: {question}
答案: {answer}

## 科室分类（选择1-3个最相关的）：
0-内科系统（内科、心血管内科、呼吸内科、消化内科、神经内科、内分泌科、肾内科等）
1-外科系统（外科、普外科、骨科、神经外科、泌尿外科、胸外科、心外科等）
2-妇产与儿科（妇科、产科、儿科、新生儿科等）
3-五官及感官系统（眼科、耳鼻喉科、口腔科、皮肤科等）
4-肿瘤与影像相关（肿瘤科、放疗科、影像科、病理科、核医学科等）
5-急诊与综合科室（急诊科、全科医学、康复科、中医科、营养科等）

## 问题类别分类（选择1-2个最相关的）：
0-疾病诊断与症状类（症状表现、诊断标准、鉴别诊断等）
1-治疗方案类（治疗方法、手术方案、康复计划等）
2-药物与用药安全类（药物使用、副作用、禁忌症等）
3-检查与化验类（检查方法、化验指标、影像学检查等）
4-预防与保健类（疾病预防、健康生活方式等）
5-特殊人群健康类（孕妇、儿童、老年人等特殊人群）
6-紧急情况与急救类（急救措施、紧急处理等）
7-医学知识与科普类（基础医学知识、健康科普等）

请以JSON格式返回标注结果：

```json
{{
    "departments": [0, 3],
    "categories": [1, 2],
    "reasoning": "简要说明分类依据"
}}
```"""

# =============================================================================
# 模板注册表
# =============================================================================
PROMPT_TEMPLATES = {
    # 基础RAG
    "basic_rag": {
        "system": BASIC_RAG_SYSTEM_PROMPT,
        "user": BASIC_RAG_USER_PROMPT
    },
    
    # 数据标注
    "medical_qa_annotation": {
        "system": ANNOTATION_SYSTEM_PROMPT,
        "user": ANNOTATION_USER_PROMPT
    },
    
    # 简单模板
    "simple_qa": "问题: {query}\n请回答:",
}

# =============================================================================
# 工具函数
# =============================================================================
def get_prompt_template(template_name: str) -> Union[Dict[str, str], str]:
    """获取提示模板"""
    return PROMPT_TEMPLATES.get(template_name, PROMPT_TEMPLATES["basic_rag"])

def register_prompt_template(name: str, template: Union[Dict[str, str], str]):
    """注册新的提示模板"""
    PROMPT_TEMPLATES[name] = template

def list_available_templates() -> list[str]:
    """列出所有可用模板"""
    return list(PROMPT_TEMPLATES.keys())

# =============================================================================
# 医疗专业术语和分类映射（保留原项目）
# =============================================================================
DEPARTMENT_MAPPING = {
    0: "内科系统",
    1: "外科系统", 
    2: "妇产与儿科",
    3: "五官及感官系统",
    4: "肿瘤与影像相关",
    5: "急诊与综合科室"
}

CATEGORY_MAPPING = {
    0: "疾病诊断与症状类",
    1: "治疗方案类",
    2: "药物与用药安全类",
    3: "检查与化验类",
    4: "预防与保健类",
    5: "特殊人群健康类",
    6: "紧急情况与急救类",
    7: "医学知识与科普类"
}

def parse_annotation_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """解析标注结果，添加名称映射"""
    parsed = result.copy()
    
    if "departments" in result:
        parsed["department_names"] = [
            DEPARTMENT_MAPPING.get(dept_id, "未知科室") for dept_id in result["departments"]
        ]
    
    if "categories" in result:
        parsed["category_names"] = [
            CATEGORY_MAPPING.get(cat_id, "未知类别") for cat_id in result["categories"]
        ]
    
    return parsed