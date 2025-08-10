"""
QAMeta/EntityMeta/PaperMeta 等
"""
"""
QA元数据定义与验证
"""
from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum
from uuid import UUID, uuid4

class DepartmentEnum(int, Enum):
    """医疗科室枚举"""
    INTERNAL_MEDICINE = 0  # "内科系统"
    SURGERY = 1  # "外科系统"  
    OBSTETRICS_PEDIATRICS = 2  # "妇产与儿科"
    ENT_SENSORY = 3  # "五官及感官系统"
    ONCOLOGY_IMAGING = 4  # "肿瘤与影像相关"
    EMERGENCY_GENERAL = 5  # "急诊与综合科室"


class CategoryEnum(int, Enum):
    """问题类别枚举"""
    DISEASE_DIAGNOSIS = 0  # "疾病诊断与症状类"
    TREATMENT_PLAN = 1  # "治疗方案类"
    MEDICATION_SAFETY = 2  # "药物与用药安全类"
    EXAMINATION_LAB = 3  # "检查与化验类"
    PREVENTION_HEALTHCARE = 4  # "预防与保健类"
    SPECIAL_POPULATION = 5  # "特殊人群健康类"
    EMERGENCY_RESCUE = 6  # "紧急情况与急救类"
    MEDICAL_KNOWLEDGE = 7  # "医学知识与科普类"


class QAMetadata(BaseModel):
    """QA记录元数据模型"""
    id: int = Field(..., description="唯一标识符")
    question: str = Field(..., description="问题文本")
    answer: str = Field(..., description="答案文本")
    
    # 向量字段
    dense_vec_q: Optional[List[float]] = Field(None, description="问题嵌入向量")
    dense_vec_a: Optional[List[float]] = Field(None, description="答案嵌入向量") 
    dense_vec_qa: Optional[List[float]] = Field(None, description="问题+答案嵌入向量")
    sparse_vec_a: Optional[Dict[int, float]] = Field(None, description="答案稀疏向量")
    sparse_vec_qa: Optional[Dict[int, float]] = Field(None, description="问题+答案稀疏向量")
    
    # 标注字段
    category: Optional[List[CategoryEnum]] = Field(None, description="问题类别")
    department: Optional[List[DepartmentEnum]] = Field(None, description="相关科室")
    
    # 时间戳
    timestamp: datetime = Field(default_factory=datetime.now, description="创建时间")
    
    # 其他元数据
    source_file: Optional[str] = Field(None, description="来源文件")
    confidence_score: Optional[float] = Field(None, description="标注置信度")

    class Config:
        use_enum_values = True


class QAAnnotationRequest(BaseModel):
    """LLM标注请求模型"""
    question: str
    answer: str
    
    
class QAAnnotationResponse(BaseModel):
    """LLM标注响应模型"""
    id: UUID = Field(
        default_factory=str("QA-" + uuid4), 
        description="自动生成的唯一ID"
    )
    question: str = Field(..., description="问题")
    answers: str = Field(..., description="答案")
    departments: List[DepartmentEnum] = Field(..., description="推荐科室")
    categories: List[CategoryEnum] = Field(..., description="问题类别")
    reasoning: str = Field(..., description="标注理由")
    confidence: float = Field(..., ge=0.0, le=1.0, description="置信度")


class JSONLRecord(BaseModel):
    """JSONL原始记录模型"""
    questions: List[List[str]] = Field(..., description="问题列表")
    answers: List[str] = Field(..., description="答案列表")
    
    def to_qa_pairs(self) -> List[tuple[str, str]]:
        """转换为QA对"""
        qa_pairs = []
        for question in self.questions[0]:
            # 如果question是列表，取第一个作为主问题
            if isinstance(question, list):
                main_question = question[0] if question else ""
            else:
                main_question = question
                
            for answer in self.answers:
                qa_pairs.append((main_question, answer))
        return qa_pairs