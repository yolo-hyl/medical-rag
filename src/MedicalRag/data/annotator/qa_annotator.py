"""
QA 样本补齐（问题/答案/出处）
"""
"""
QA自动标注器 - 使用LLM自动标注问答对的科室和类别
"""
import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from pydantic import BaseModel
import re
from MedicalRag.core.llm.client import OllamaClient
from MedicalRag.schemas.metadata import (
    QAAnnotationResponse,
    DepartmentEnum,
    CategoryEnum
)
from MedicalRag.config.prompts import ANNOTATION_PROMPT
from MedicalRag.core.base import BaseAnnotator
from MedicalRag.core.base.BaseClient import LLMHttpClient
from MedicalRag.core.base.BaseAnnotator import BaseAnnotatorCfg

logger = logging.getLogger(__name__)


class QAAnnotator(BaseAnnotator):
    """QA问答对自动标注器"""

    def __init__(
        self,
        cfg: BaseAnnotatorCfg
    ):
        """
        初始化标注器
        
        Args:
            ollama_client: Ollama客户端实例
            base_url: Ollama服务地址
            model: 使用的模型名称
        """
        super().__init__(cfg)
        self.annotation_stats["department_distribution"] = {}
        self.annotation_stats["category_distribution"] = {}
    
    
    def parse_structured_output(
        self, 
        llm_result: str, 
        single_data: dict,
        target_model: BaseModel
    ) -> QAAnnotationResponse:
        """
        解析结构化输出
        
        Args:
            content: 响应内容
            target_model: 目标Pydantic模型
            
        Returns:
            解析后的模型实例
        """
        # 尝试提取JSON部分
        json_pattern = r'```json\s*(.*?)\s*```'
        json_match = re.search(json_pattern, llm_result, re.DOTALL)
        
        if json_match:
            json_str = json_match.group(1)
        else:
            # 如果没有代码块，尝试直接解析
            json_str = llm_result.strip()
        
        try:
            parsed_data: dict = json.loads(json_str)
            parsed_data.update(single_data)
            return target_model(**parsed_data)
        except (json.JSONDecodeError, ValueError) as e:            
            raise ValueError(f"Failed to parse structured output: {e}")
    
    def _validate_annotation(self, annotation: QAAnnotationResponse) -> bool:
        """验证标注结果的有效性"""
        try:
            # 检查科室是否在枚举范围内
            [DepartmentEnum(i) for i in annotation.departments]
            # 检查类别是否在枚举范围内  
            [CategoryEnum(i) for i in annotation.categories]
            # 检查置信度范围
            return 0.0 <= annotation.confidence <= 1.0
        except (ValueError, TypeError):
            return False
    
    def _update_stats(self, annotation: QAAnnotationResponse):
        """更新统计信息"""
        depts = annotation.departments
        cats = annotation.categories
        
        for dept in depts:
            if dept not in self.annotation_stats["department_distribution"]:
                self.annotation_stats["department_distribution"][dept] = 0
            self.annotation_stats["department_distribution"][dept] += 1

        for cat in cats:
            if cat not in self.annotation_stats["category_distribution"]:
                self.annotation_stats["category_distribution"][cat] = 0
            self.annotation_stats["category_distribution"][cat] += 1
    
    
    def _log_final_stats(self):
        """输出最终统计信息"""
        super()._log_final_stats()
        stats = self.annotation_stats
        logger.info(f"=== 科室分布 ===")
        for dept, count in stats["department_distribution"].items():
            logger.info(f"{dept}: {count}")
            
        logger.info(f"=== 类别分布 ===") 
        for cat, count in stats["category_distribution"].items():
            logger.info(f"{cat}: {count}")