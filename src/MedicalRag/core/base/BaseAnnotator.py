"""
抽象Annotator接口（LLM/规则）
"""
import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from pydantic import BaseModel, Field, HttpUrl, field_validator
from MedicalRag.core.base.BaseClient import LLMHttpClientCfg
import re
from MedicalRag.core.llm.client import LLMHttpClient
from abc import ABC, abstractmethod
from tqdm import tqdm

logger = logging.getLogger(__name__)

class BaseAnnotatorCfg(BaseModel):
    llm_config: LLMHttpClientCfg
    max_retries: int = Field(10, description="重试次数")
    system_prompt: Optional[str] = Field(None, description="系统提示词")
    batch_size: int = Field(5, description="批量大小")
    save_intermediate: bool = Field(True, description="是否保存批量中间结果")
    checkpoint_num: int = Field(2, description="积累定量数据保存一次")
    output_file_path: str = Field("qwen3:32b", description="保存的文件路径")
    
class BaseAnnotator(ABC):
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
        self.output_file_path = cfg.output_file_path
        self.save_intermediate = cfg.save_intermediate
        self.batch_size = cfg.batch_size
        self.system_prompt = cfg.system_prompt
        self.max_retries = cfg.max_retries
        self.client = LLMHttpClient(cfg.llm_config)
        self.max_retries = cfg.max_retries
        self.checkpoint_num = cfg.checkpoint_num
        self.annotation_stats = {
            "total": 0,
            "success": 0,
            "failed": 0
        }
        
    @abstractmethod
    def build_prompt(
        self,
        single_data: dict
    ):
        pass
    
    def annotate_single(
        self, 
        single_data: dict,
        base_model : BaseModel,
        system_prompt: str
    ) -> Optional[BaseModel]:
        """
        标注单个QA对
        
        Args:
            question: 问题文本
            answer: 答案文本
            max_retries: 最大重试次数
            
        Returns:
            标注结果，失败时返回None
        """
        self.annotation_stats["total"] += 1
        prompt = self.build_prompt(single_data=single_data)
        for attempt in range(self.max_retries):
            try:                
                # 调用LLM进行标注
                result = self.client.generate_completion(
                    prompt=prompt,
                    system_prompt=system_prompt
                )
                
                # 解析结构化输出
                annotation : base_model = self.parse_structured_output(
                    llm_result=result, 
                    single_data=single_data,
                    target_model=base_model
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
                    logger.error(f"标注最终失败，部分提示词: {prompt[-50:]}...")
        
        return None
    
    @abstractmethod
    def parse_structured_output(
        self, 
        llm_result: Dict[str, Any],
        single_data: dict,
        target_model: BaseModel
    ) -> BaseModel:
        """
        解析结构化输出
        
        Args:
            content: 响应内容
            target_model: 目标Pydantic模型
            
        Returns:
            解析后的模型实例，由子类实现
        """
        pass
    
    def annotate_batch(
        self,
        content: List[dict],
        base_model: BaseModel,
    ) -> List[BaseModel]:
        """
        批量标注
        
        Args:
            content: 原始内容
            base_model: 期望的数据格式
            batch_size: 批处理大小
            save_intermediate: 是否保存中间结果
            output_file: 输出文件路径
            
        Returns:
            标注结果列表
        """
        results = []
        total_count = len(content)
        
        logger.info(f"开始批量标注，总计 {total_count} 条数据")
        
        for i in tqdm(range(0, total_count, self.batch_size)):
            batch = content[i:i + self.batch_size]
            batch_results = []
            for item in batch:
                try:
                    annotation = self.annotate_single(
                        single_data=item, 
                        base_model=base_model,
                        system_prompt=self.system_prompt
                    )
                    if annotation:
                        batch_results.append(annotation)
                        
                except KeyboardInterrupt:
                    logger.info("用户中断，保存当前结果...")
                    break
                except Exception as e:
                    logger.error(f"处理QA对时出错: {e}")
                    continue
            
            results.extend(batch_results)
            
            # 保存中间结果
            if self.save_intermediate and self.output_file_path and len(results) % self.checkpoint_num == 0:
                self._save_intermediate_results(results, self.output_file_path, i + self.batch_size)
            
            # 输出进度信息
            self._log_progress(i + self.batch_size, total_count)
        
        logger.info(f"批量标注完成，成功标注 {len(results)}/{total_count} 个QA对")
        self._log_final_stats()
        
        return results
    
    @abstractmethod
    def _validate_annotation(self, annotation: BaseModel) -> bool:
        """验证标注结果的有效性，由子类实现"""
        pass
    
    @abstractmethod
    def _update_stats(self, annotation: BaseModel):
        """更新统计信息，由子类实现"""
        pass
    
    def _save_intermediate_results(
        self, 
        results: List[BaseModel], 
        output_file: Path,
        current_count: int
    ):
        """保存中间结果"""
        temp_file = Path(output_file).with_suffix(f'.temp_{current_count}.json')
        try:
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(
                    [r.model_dump(mode="json") for r in results],
                    f,
                    ensure_ascii=False
                )
            logger.info(f"中间结果已保存: {temp_file}")
        except Exception as e:
            logger.error(f"保存中间结果失败: {e}")
    
    def _log_progress(self, current: int, total: int):
        """输出进度日志"""
        progress = (current / total) * 100
        success_rate = (self.annotation_stats["success"] / 
                       max(self.annotation_stats["total"], 1)) * 100
        if int(progress) % 10 == 0:
            logger.info(
                f"进度: {current}/{total} ({progress:.1f}%), "
                f"成功率: {success_rate:.1f}%"
            )
    
    def _log_final_stats(self):
        """输出最终统计信息"""
        stats = self.annotation_stats
        logger.info(f"=== 标注统计 ===")
        logger.info(f"总数: {stats['total']}")
        logger.info(f"成功: {stats['success']}")
        logger.info(f"失败: {stats['failed']}")
        logger.info(f"成功率: {stats['success'] / max(stats['total'], 1)*100:.1f}%")
        
    def save_annotations(
        self,
        annotations: List[BaseModel],
        output_file: Path
    ):
        """保存标注结果到文件"""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(
                    [ann.model_dump(mode="json") for ann in annotations],
                    f,
                    ensure_ascii=False,
                    indent=2
                )
            logger.info(f"标注结果已保存: {output_file}")
        except Exception as e:
            logger.error(f"保存标注结果失败: {e}")
            raise
    
    def load_annotations(self, input_file: Path) -> List[BaseModel]:
        """从文件加载标注结果"""
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return [BaseModel(**item) for item in data]
        except Exception as e:
            logger.error(f"加载标注结果失败: {e}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """获取当前统计信息"""
        return self.annotation_stats.copy()