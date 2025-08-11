#!/usr/bin/env python3
"""
QA标注测试脚本
用于处理JSONL格式的QA数据并进行自动标注
"""

import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any
import sys
import re

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(project_root))

from MedicalRag.schemas.metadata import JSONLRecord, QAAnnotationResponse, DepartmentEnum, CategoryEnum
from MedicalRag.config.prompts import ANNOTATION_PROMPT
from MedicalRag.core.base.BaseClient import LLMHttpClientCfg
from MedicalRag.core.llm.client import OllamaClient

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class QAAnnotator:
    """QA问答对自动标注器"""

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        model: str = "qwen3:32b",
        max_retries: int = 3
    ):
        """
        初始化标注器
        
        Args:
            ollama_url: Ollama服务地址
            model: 使用的模型名称
            max_retries: 最大重试次数
        """
        self.client = OllamaClient(
            base_url=ollama_url,
            model=model,
            thinking_model=True
        )
        self.max_retries = max_retries
        self.annotation_stats = {
            "total": 0,
            "success": 0,
            "failed": 0,
            "category_stats": {},
            "department_stats": {}
        }
        
    def _extract_json_from_response(self, text: str) -> dict:
        """
        从响应文本中提取JSON内容
        
        Args:
            text: 响应文本
            
        Returns:
            解析出的JSON字典
        """
        # 移除thinking标签内容
        cleaned_text = re.sub(r'<think>.*?</think>\s*\n\n', '', text, flags=re.DOTALL)
        
        # 尝试匹配JSON代码块
        json_pattern = r'```json\s*(\{.*?\})\s*```'
        match = re.search(json_pattern, cleaned_text, re.DOTALL)
        
        if match:
            json_str = match.group(1)
        else:
            # 如果没有代码块，尝试找到第一个完整的JSON对象
            json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
            match = re.search(json_pattern, cleaned_text)
            if match:
                json_str = match.group(0)
            else:
                raise ValueError("无法在响应中找到有效的JSON")
        
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON解析失败: {e}, 原始文本: {json_str[:200]}...")
            raise
    
    def annotate_single_qa(
        self, 
        question: str, 
        answer: str
    ) -> QAAnnotationResponse:
        """
        标注单个QA对
        
        Args:
            question: 问题文本
            answer: 答案文本
            
        Returns:
            标注结果，失败时返回None
        """
        self.annotation_stats["total"] += 1
        
        for attempt in range(self.max_retries):
            try:
                # 构建提示词
                prompt = ANNOTATION_PROMPT.format(
                    question=question,
                    answer=answer
                )
                
                # 调用LLM进行标注
                result = self.client.generate_completion(
                    prompt=prompt,
                    system_prompt="你是一个专业的医疗问答标注专家，请严格按照要求输出JSON格式的标注结果。"
                )
                
                # 从响应中提取答案部分
                response_text = result.get("answer", result.get("raw_content", ""))
                
                # 解析结构化输出
                annotation_data = self._extract_json_from_response(response_text)
                
                # 验证必需字段
                required_fields = ["departments", "categories", "reasoning", "confidence"]
                for field in required_fields:
                    if field not in annotation_data:
                        raise ValueError(f"缺少必需字段: {field}")
                
                # 创建标注响应对象
                annotation = QAAnnotationResponse(
                    question=question,
                    answers=answer,  # 注意这里用的是answers字段
                    departments=[DepartmentEnum(d) for d in annotation_data["departments"]],
                    categories=[CategoryEnum(c) for c in annotation_data["categories"]],
                    reasoning=annotation_data["reasoning"],
                    confidence=float(annotation_data["confidence"])
                )
                
                # 验证标注结果
                if self._validate_annotation(annotation):
                    self.annotation_stats["success"] += 1
                    self._update_stats(annotation)
                    return annotation
                else:
                    logger.warning(f"标注结果验证失败: {annotation}")
                    
            except Exception as e:
                logger.error(f"标注尝试 {attempt + 1} 失败: {e}")
                if attempt == self.max_retries - 1:
                    self.annotation_stats["failed"] += 1
                    logger.error(f"标注最终失败，问题: {question[:50]}...")
        
        return None
    
    def _validate_annotation(self, annotation: QAAnnotationResponse) -> bool:
        """验证标注结果的有效性"""
        if not annotation:
            return False
        
        # 检查置信度范围
        if not (0.0 <= annotation.confidence <= 1.0):
            return False
        
        # 检查科室和类别不为空
        if not annotation.departments or not annotation.categories:
            return False
        
        # 检查推理不为空
        if not annotation.reasoning.strip():
            return False
        
        return True
    
    def _update_stats(self, annotation: QAAnnotationResponse):
        """更新统计信息"""
        # 统计科室分布
        for dept in annotation.departments:
            dept_name = dept.name
            self.annotation_stats["department_stats"][dept_name] = \
                self.annotation_stats["department_stats"].get(dept_name, 0) + 1
        
        # 统计类别分布
        for cat in annotation.categories:
            cat_name = cat.name
            self.annotation_stats["category_stats"][cat_name] = \
                self.annotation_stats["category_stats"].get(cat_name, 0) + 1
    
    def process_jsonl_file(
        self,
        input_file: Path,
        output_dir: Path,
        batch_size: int = 10
    ) -> List[QAAnnotationResponse]:
        """
        处理JSONL文件并进行批量标注
        
        Args:
            input_file: 输入JSONL文件路径
            output_dir: 输出目录
            batch_size: 批处理大小
            
        Returns:
            标注结果列表
        """
        logger.info(f"开始处理文件: {input_file}")
        
        # 读取JSONL文件
        qa_pairs = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    record = JSONLRecord(**data)
                    pairs = record.to_qa_pairs()
                    qa_pairs.extend(pairs)
                    logger.info(f"从第 {line_num} 行提取了 {len(pairs)} 个QA对")
                except Exception as e:
                    logger.error(f"处理第 {line_num} 行时出错: {e}")
                    continue
        
        logger.info(f"总共提取了 {len(qa_pairs)} 个QA对")
        
        # 批量标注
        results = []
        failed_count = 0
        
        for i in range(0, len(qa_pairs), batch_size):
            batch = qa_pairs[i:i + batch_size]
            logger.info(f"处理批次 {i//batch_size + 1}/{(len(qa_pairs)-1)//batch_size + 1}")
            
            for question, answer in batch:
                try:
                    annotation = self.annotate_single_qa(question, answer)
                    if annotation:
                        results.append(annotation)
                    else:
                        failed_count += 1
                        
                except KeyboardInterrupt:
                    logger.info("用户中断，保存当前结果...")
                    break
                except Exception as e:
                    logger.error(f"处理QA对时出错: {e}")
                    failed_count += 1
                    continue
            
            # 保存中间结果
            if results:
                temp_file = output_dir / f"annotations_batch_{i//batch_size + 1}.json"
                self._save_results(results[-len(batch):], temp_file)
        
        logger.info(f"标注完成，成功: {len(results)}, 失败: {failed_count}")
        self._log_final_stats()
        
        # 保存最终结果
        if results:
            final_file = output_dir / f"annotations_{input_file.stem}.json"
            self._save_results(results, final_file)
        
        return results
    
    def _save_results(self, results: List[QAAnnotationResponse], output_file: Path):
        """保存标注结果到文件"""
        try:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(
                    [result.model_dump() for result in results],
                    f,
                    ensure_ascii=False,
                    indent=2
                )
            logger.info(f"结果已保存: {output_file}")
        except Exception as e:
            logger.error(f"保存结果失败: {e}")
    
    def _log_final_stats(self):
        """输出最终统计信息"""
        stats = self.annotation_stats
        logger.info(f"=== 标注统计 ===")
        logger.info(f"总数: {stats['total']}")
        logger.info(f"成功: {stats['success']}")
        logger.info(f"失败: {stats['failed']}")
        logger.info(f"成功率: {stats['success'] / max(stats['total'], 1) * 100:.1f}%")
        
        if stats['department_stats']:
            logger.info("科室分布:")
            for dept, count in stats['department_stats'].items():
                logger.info(f"  {dept}: {count}")
        
        if stats['category_stats']:
            logger.info("类别分布:")
            for cat, count in stats['category_stats'].items():
                logger.info(f"  {cat}: {count}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="QA标注测试脚本")
    parser.add_argument("input_file", help="输入JSONL文件路径")
    parser.add_argument("-o", "--output-dir", default="./output", help="输出目录")
    parser.add_argument("--ollama-url", default="http://localhost:11434", help="Ollama服务地址")
    parser.add_argument("--model", default="qwen3:32b", help="使用的模型名称")
    parser.add_argument("--batch-size", type=int, default=5, help="批处理大小")
    parser.add_argument("--max-retries", type=int, default=3, help="最大重试次数")
    
    args = parser.parse_args()
    
    # 验证输入文件
    input_file = Path(args.input_file)
    if not input_file.exists():
        logger.error(f"输入文件不存在: {input_file}")
        return
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 初始化标注器
    annotator = QAAnnotator(
        ollama_url=args.ollama_url,
        model=args.model,
        max_retries=args.max_retries
    )
    
    # 检查Ollama服务
    if not annotator.client.health_check():
        logger.error(f"无法连接到Ollama服务: {args.ollama_url}")
        logger.error("请确保Ollama服务正在运行")
        return
    
    logger.info(f"连接到Ollama服务成功，可用模型: {annotator.client.list_models()}")
    
    try:
        # 处理文件
        results = annotator.process_jsonl_file(
            input_file=input_file,
            output_dir=output_dir,
            batch_size=args.batch_size
        )
        
        logger.info(f"处理完成，共标注 {len(results)} 个QA对")
        
    except Exception as e:
        logger.error(f"处理过程中出错: {e}")
        raise


if __name__ == "__main__":
    main()