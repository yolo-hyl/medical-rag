# file: mcp_label_server.py
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, validator
from fastmcp import FastMCP, tool  # 假设 fastmcp 提供类似接口
import re

# ---------- 配置 ----------
MAX_TEXT_LEN = 512
MAX_LIST_LEN = 6

# 科室枚举（索引即你给的编号）
DEPARTMENT_ENUM = {
    0: "内科系统",
    1: "外科系统",
    2: "妇产与儿科",
    3: "五官及感官系统",
    4: "肿瘤与影像相关",
    5: "急诊与综合科室",
}

# 问题类别枚举
CATEGORY_ENUM = {
    0: "疾病诊断与症状类",
    1: "治疗方案类",
    2: "药物与用药安全类",
    3: "检查与化验类",
    4: "预防与保健类",
    5: "特殊人群健康类",
    6: "紧急情况与急救类",
    7: "医学知识与科普类",
}

def _truncate(s: str, max_len: int = MAX_TEXT_LEN) -> str:
    s = s.strip()
    return s[:max_len]

# ---------- 简单规则 ----------
def classify_departments(text: str) -> List[int]:
    """
    非常朴素的关键词规则，请按需替换为你的模型或更复杂规则。
    """
    t = text.lower()
    tags: List[int] = []

    # 五官及感官（3）：口腔/耳鼻喉/皮肤/眼科 等
    if any(k in t for k in ["口干", "口腔", "口臭", "唾液", "龈", "牙", "咽喉", "耳鼻喉", "皮肤", "眼"]):
        tags.append(3)

    # 你可以继续追加其他系统的关键词……
    # 内科（0）
    if any(k in t for k in ["内科", "高血压", "糖尿病", "内分泌", "肾内", "风湿"]):
        tags.append(0)

    # 去重、限长
    tags = sorted(set(tags))[:MAX_LIST_LEN]
    return tags or [3]  # 兜底：归到 3（五官）

def classify_categories(text: str) -> List[int]:
    t = text
    tags: List[int] = []

    # 1: 治疗方案类
    if re.search(r"(治疗|治法|方案|如何治疗|怎么治|处置|处方|康复)", t):
        tags.append(1)

    # 0: 诊断/症状
    if re.search(r"(症状|表现|诊断|鉴别)", t):
        tags.append(0)

    # 2: 用药
    if re.search(r"(用药|药物|副作用|禁忌|剂量)", t):
        tags.append(2)

    # 3: 检查
    if re.search(r"(检查|化验|指标|影像|CT|核磁|MRI|超声|化验单)", t):
        tags.append(3)

    # 4: 预防/保健
    if re.search(r"(预防|保健|生活方式|复发预防|保养)", t):
        tags.append(4)

    # 5: 特殊人群
    if re.search(r"(孕|妊娠|儿童|新生儿|老年|老人|青少年)", t):
        tags.append(5)

    # 6: 紧急/急救
    if re.search(r"(急救|紧急|危急|休克|心梗|中风)", t):
        tags.append(6)

    # 7: 科普
    if re.search(r"(是什么|原理|科普|基础知识)", t):
        tags.append(7)

    tags = sorted(set(tags))[:MAX_LIST_LEN]
    return tags or [1]  # 兜底：治疗方案类

# ---------- Pydantic 模型 ----------
class SourceData(BaseModel):
    questions: List[List[str]] = Field(..., description="二维数组：每组问题的同义问列表")
    answers: List[str] = Field(..., description="与 questions 等长，每组一个答案")

    @validator("answers")
    def _len_match(cls, v, values):
        q = values.get("questions")
        if q and len(q) != len(v):
            raise ValueError("answers 长度必须与 questions 长度一致")
        return v

class LabeledItem(BaseModel):
    question: str = Field(..., max_length=MAX_TEXT_LEN)
    answer: str = Field(..., max_length=MAX_TEXT_LEN)
    departments: List[int] = Field(..., max_items=MAX_LIST_LEN)
    categories: List[int] = Field(..., max_items=MAX_LIST_LEN)

class LabeledResult(BaseModel):
    items: List[LabeledItem]

# ---------- FastMCP 工具 ----------
app = FastMCP(name="qa-labeler", version="0.1.0", description="将QA源数据标注为训练所需格式")

@tool(
    name="label_qa",
    description="将源JSON QA数据转换为（question, answer, departments, categories）列表",
    # 参数与返回值的 JSON Schema 由 Pydantic 自动生成；FastMCP 通常会读取函数签名/注解
)
def label_qa(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    payload 示例：
    {
      "questions": [["问1","问1改写"], ["问2"]],
      "answers": ["答1", "答2"]
    }
    返回：
    { "items": [ {question, answer, departments, categories}, ... ] }
    """
    src = SourceData(**payload)
    out: List[LabeledItem] = []

    for q_group, ans in zip(src.questions, src.answers):
        ans_t = _truncate(ans, MAX_TEXT_LEN)
        for q in q_group:
            q_t = _truncate(q, MAX_TEXT_LEN)
            deps = classify_departments(q_t + " " + ans_t)
            cats = classify_categories(q_t) or classify_categories(ans_t)
            out.append(
                LabeledItem(
                    question=q_t,
                    answer=ans_t,
                    departments=deps,
                    categories=cats,
                )
            )

    return LabeledResult(items=out).dict()

if __name__ == "__main__":
    # 以本地方式运行（FastMCP 通常提供 stdio/stdio+uvicorn 两种模式）
    # 1) 作为 MCP stdio server: app.run_stdio()
    # 2) 或启动 HTTP（用于调试/自测）:
    #    uvicorn 要求 FastMCP 暴露 ASGI app；若你的 fastmcp 版本已内置，使用 app.run_http()
    app.run_stdio()
