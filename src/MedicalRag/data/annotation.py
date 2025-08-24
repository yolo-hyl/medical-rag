# src/medical_rag/knowledge/annotation.py
"""
自动标注模块（基于langchain简化实现）
"""
import json
import logging
import re
from typing import List, Dict, Any, Optional
from pathlib import Path
from datasets import Dataset, load_dataset
from tqdm import tqdm
from langchain_core.messages import HumanMessage, SystemMessage

from ..config.models import AppConfig, LLMConfig
from ..core.utils import create_llm_client  
from ..prompts.templates import get_prompt_template, parse_annotation_result

logger = logging.getLogger(__name__)

class SimpleAnnotator:
    """简化的标注器"""
    
    def __init__(self, llm_config: LLMConfig, batch_size: int = 10, max_retries: int = 3):
        self.llm = create_llm_client(llm_config)
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.success_count = 0
        self.failed_count = 0
    
    def _extract_json_from_response(self, response_text: str) -> Dict[str, Any]:
        """从响应中提取JSON"""
        # 尝试提取代码块中的JSON
        json_pattern = r'```json\s*(.*?)\s*```'
        json_match = re.search(json_pattern, response_text, re.DOTALL)
        
        if json_match:
            json_text = json_match.group(1)
        else:
            # 尝试找到第一个完整的JSON对象
            start_idx = response_text.find('{')
            if start_idx != -1:
                brace_count = 0
                end_idx = start_idx
                for i in range(start_idx, len(response_text)):
                    if response_text[i] == '{':
                        brace_count += 1
                    elif response_text[i] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end_idx = i + 1
                            break
                json_text = response_text[start_idx:end_idx]
            else:
                json_text = response_text.strip()
        
        return json.loads(json_text)
    
    def _validate_annotation(self, annotation: Dict[str, Any]) -> bool:
        """验证标注结果"""
        # 检查必要字段
        required_fields = ["departments", "categories"]
        for field in required_fields:
            if field not in annotation:
                return False
        
        # 验证departments
        departments = annotation["departments"]
        if not isinstance(departments, list) or len(departments) == 0 or len(departments) > 3:
            return False
        
        for dept in departments:
            if not isinstance(dept, int) or dept < 0 or dept > 5:
                return False
        
        # 验证categories  
        categories = annotation["categories"]
        if not isinstance(categories, list) or len(categories) == 0 or len(categories) > 3:
            return False
        
        for cat in categories:
            if not isinstance(cat, int) or cat < 0 or cat > 7:
                return False
        
        return True
    
    def annotate_single(self, question: str, answer: str) -> Optional[Dict[str, Any]]:
        """标注单个样本"""
        for attempt in range(self.max_retries):
            try:
                # 获取提示模板
                template = get_prompt_template("medical_qa_annotation")
                
                if isinstance(template, dict):
                    system_prompt = template.get('system', '')
                    user_prompt = template.get('user', '').format(
                        question=question,
                        answer=answer
                    )
                    
                    messages = [
                        SystemMessage(content=system_prompt),
                        HumanMessage(content=user_prompt)
                    ]
                else:
                    # 简单模板
                    formatted_prompt = template.format(
                        question=question,
                        answer=answer
                    )
                    messages = [HumanMessage(content=formatted_prompt)]
                
                # 调用LLM
                response = self.llm.invoke(messages)
                response_text = response.content
                
                # 解析响应
                annotation = self._extract_json_from_response(response_text)
                
                # 验证结果
                if self._validate_annotation(annotation):
                    self.success_count += 1
                    return parse_annotation_result(annotation)
                else:
                    logger.warning(f"标注结果验证失败，尝试 {attempt + 1}/{self.max_retries}")
                    
            except Exception as e:
                logger.error(f"标注失败，尝试 {attempt + 1}/{self.max_retries}: {e}")
                if attempt == self.max_retries - 1:
                    self.failed_count += 1
                    return None
        
        self.failed_count += 1
        return None
    
    def annotate_dataset(
        self, 
        dataset: Dataset, 
        question_field: str = "question",
        answer_field: str = "answer",
        save_path: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """标注整个数据集"""
        logger.info(f"开始标注数据集，共 {len(dataset)} 条数据")
        
        annotated_results = []
        self.success_count = 0
        self.failed_count = 0
        
        # 处理数据
        with tqdm(total=len(dataset), desc="标注进度") as pbar:
            for i, record in enumerate(dataset):
                question = record.get(question_field, "")
                answer = record.get(answer_field, "")
                
                if not question or not answer:
                    logger.warning(f"跳过空记录: index={i}")
                    pbar.update(1)
                    continue
                
                # 执行标注
                annotation = self.annotate_single(question, answer)
                
                if annotation:
                    # 构建结果记录
                    result = {
                        "id": record.get("id", f"item_{i}"),
                        "question": question,
                        "answer": answer,
                        **annotation  # 包含departments, categories等
                    }
                    
                    # 保留原始记录的其他字段
                    for key, value in record.items():
                        if key not in result:
                            result[key] = value
                    
                    annotated_results.append(result)
                
                pbar.update(1)
                
                # 定期保存中间结果
                if save_path and len(annotated_results) % 100 == 0:
                    temp_path = f"{save_path}.temp"
                    with open(temp_path, 'w', encoding='utf-8') as f:
                        json.dump(annotated_results, f, ensure_ascii=False, indent=2)
        
        # 保存最终结果
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(annotated_results, f, ensure_ascii=False, indent=2)
            logger.info(f"标注结果已保存到: {save_path}")
        
        logger.info(f"标注完成！成功: {self.success_count}, 失败: {self.failed_count}")
        return annotated_results

class AnnotationPipeline:
    """标注流水线"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.annotator = SimpleAnnotator(
            llm_config=config.llm,
            batch_size=config.data.batch_size
        )
    
    def run(
        self,
        data_path: str,
        output_path: str,
        question_field: str = "question",
        answer_field: str = "answer"
    ) -> bool:
        """运行标注流水线"""
        try:
            # 1. 加载数据
            logger.info(f"加载数据: {data_path}")
            
            if data_path.endswith('.jsonl'):
                dataset = load_dataset('json', data_files=data_path, split='train')
            elif data_path.endswith('.json'):
                dataset = load_dataset('json', data_files=data_path, split='train')
            elif data_path.endswith('.parquet'):
                dataset = load_dataset('parquet', data_files=data_path, split='train')
            else:
                raise ValueError(f"不支持的文件格式: {data_path}")
            
            # 2. 执行标注
            results = self.annotator.annotate_dataset(
                dataset=dataset,
                question_field=question_field,
                answer_field=answer_field,
                save_path=output_path
            )
            
            logger.info(f"标注流水线完成，生成 {len(results)} 条标注结果")
            return True
            
        except Exception as e:
            logger.error(f"标注流水线失败: {e}")
            return False

# 便捷函数
def run_annotation(
    config: AppConfig,
    data_path: str,
    output_path: str,
    question_field: str = "question",
    answer_field: str = "answer"
) -> bool:
    """运行标注的便捷函数"""
    pipeline = AnnotationPipeline(config)
    return pipeline.run(data_path, output_path, question_field, answer_field)