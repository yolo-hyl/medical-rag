import asyncio
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pandas as pd
from datasets import Dataset, load_dataset
from tqdm import tqdm
import re
from ...core.base.BaseClient import LLMClient
from ...config.data.annotator import LabelingConfig
from abc import ABC, abstractclassmethod

class DatasetLabeler(ABC):
    """数据集标注器"""
    
    def __init__(self, llm_client: LLMClient, config: LabelingConfig, logger):
        self.llm_client = llm_client
        self.config = config
        self.logger = logger
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 创建检查点和失败记录文件路径
        self.checkpoint_file = self.output_dir / config.checkpoint_file
        self.failed_file = self.output_dir / config.failed_file
        self.results_file = self.output_dir / config.results_file
    
    def save_checkpoint(self, processed_count: int, failed_samples: List[Dict]):
        """保存检查点"""
        checkpoint = {
            "processed_count": processed_count,
            "timestamp": time.time(),
            "failed_count": len(failed_samples)
        }
        
        with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint, f, ensure_ascii=False, indent=2)
        
        # 保存失败样本
        with open(self.failed_file, 'w', encoding='utf-8') as f:
            json.dump(failed_samples, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"检查点已保存: 已处理 {processed_count} 条，失败 {len(failed_samples)} 条")
    
    def load_checkpoint(self) -> tuple[int, List[Dict]]:
        """加载检查点"""
        if not self.checkpoint_file.exists():
            return 0, []
        
        try:
            with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint = json.load(f)
            
            failed_samples = []
            if self.failed_file.exists():
                with open(self.failed_file, 'r', encoding='utf-8') as f:
                    failed_samples = json.load(f)
            
            processed_count = checkpoint.get("processed_count", 0)
            self.logger.info(f"从检查点恢复: 已处理 {processed_count} 条")
            return processed_count, failed_samples
            
        except Exception as e:
            self.logger.error(f"加载检查点失败: {e}")
            return 0, []
        
    @abstractclassmethod
    def sangle_response_get_label(self, response_text: str):
        """单条数据抽取标注的标签"""
        pass
    
    @abstractclassmethod
    async def label_single_sample(self, sample: Dict, retries: int = 0) -> Optional[Dict]:
        """标注单个样本（带重试）"""
        pass
            
    def _normalize_batch(self, batch_obj) -> List[Dict]:
        """标准化批次数据格式"""
        # 单条样本：就是 dict（列->标量）
        if isinstance(batch_obj, dict):
            # 可能是 dict-of-lists 或 单条 dict
            first_val = next(iter(batch_obj.values())) if batch_obj else None
            if isinstance(first_val, list):
                # dict-of-lists -> list[dict]
                size = len(first_val)
                return [{k: batch_obj[k][i] for k in batch_obj.keys()} for i in range(size)]
            else:
                return [batch_obj]

        # Dataset 切片对象
        if hasattr(batch_obj, 'to_dict'):
            data = batch_obj.to_dict()
            first_val = next(iter(data.values())) if data else None
            if isinstance(first_val, list):
                size = len(first_val)
                return [{k: data[k][i] for k in data.keys()} for i in range(size)]
            else:
                return [data]

        # 已经是 list[dict]
        if isinstance(batch_obj, list):
            return batch_obj

        # 兜底
        return [batch_obj]
    
    @abstractclassmethod
    async def label_batch(self, batch: List[Dict]) -> tuple[List[Dict], List[Dict]]:
        """标注一批样本"""
        pass
    
    async def label_dataset(self, dataset: Dataset) -> Dataset:
        """标注整个数据集"""
        self.logger.info(f"开始标注数据集，共 {len(dataset)} 条数据")
        
        # 加载检查点
        start_idx = 0
        all_failed = []
        if self.config.resume_from_checkpoint:
            start_idx, all_failed = self.load_checkpoint()
        
        # 准备数据
        all_labeled = []
        
        # 如果有已处理的结果，先加载
        if start_idx > 0 and self.results_file.exists():
            try:
                with open(self.results_file, 'r', encoding='utf-8') as f:
                    existing_results = json.load(f)
                    all_labeled.extend(existing_results[:start_idx])
                self.logger.info(f"加载了 {len(all_labeled)} 条已标注数据")
            except Exception as e:
                self.logger.error(f"加载已有结果失败: {e}")
                start_idx = 0
                all_labeled = []
        
        # 开始标注
        pbar = tqdm(total=len(dataset), initial=start_idx, desc="标注进度")
        
        for i in range(start_idx, len(dataset), self.config.batch_size):
            batch_end = min(i + self.config.batch_size, len(dataset))
            raw_batch = dataset[i:batch_end]
            batch = self._normalize_batch(raw_batch)
            
            # 标注批次
            successful, failed = await self.label_batch(batch)
            
            all_labeled.extend(successful)
            all_failed.extend(failed)
            
            # 更新进度
            pbar.update(len(batch))
            
            # 定期保存检查点
            if (i + len(batch)) % self.config.checkpoint_interval == 0:
                self.save_checkpoint(i + len(batch), all_failed)
                
                # 保存中间结果
                with open(self.results_file, 'w', encoding='utf-8') as f:
                    json.dump(all_labeled, f, ensure_ascii=False, indent=2)
        
        pbar.close()
        
        # 最终保存
        self.save_checkpoint(len(dataset), all_failed)
        
        # 保存最终结果
        self._save_final_results(all_labeled)
        
        self.logger.info(f"标注完成！成功 {len(all_labeled)} 条，失败 {len(all_failed)} 条")
        
        # 转换为Dataset并返回
        if all_labeled:
            return Dataset.from_list(all_labeled)
        else:
            return Dataset.from_dict({})
    
    def _save_final_results(self, labeled_data: List[Dict]):
        """保存最终结果到多种格式"""
        if not labeled_data:
            self.logger.warning("没有标注数据需要保存")
            return
        
        for format_type in self.config.output_formats:
            try:
                if format_type == "json":
                    with open(self.results_file, 'w', encoding='utf-8') as f:
                        json.dump(labeled_data, f, ensure_ascii=False, indent=2)
                    self.logger.info(f"已保存JSON格式: {self.results_file}")
                
                elif format_type == "dataset":
                    dataset = Dataset.from_list(labeled_data)
                    dataset_path = self.output_dir / self.config.dataset_dir
                    dataset.save_to_disk(str(dataset_path))
                    self.logger.info(f"已保存Dataset格式: {dataset_path}")
                
                elif format_type == "csv":
                    df = pd.DataFrame(labeled_data)
                    csv_path = self.output_dir / f"{self.config.dataset_dir}.csv"
                    df.to_csv(csv_path, index=False, encoding='utf-8')
                    self.logger.info(f"已保存CSV格式: {csv_path}")
                
                elif format_type == "parquet":
                    df = pd.DataFrame(labeled_data)
                    parquet_path = self.output_dir / f"{self.config.dataset_dir}.parquet"
                    df.to_parquet(parquet_path, index=False)
                    self.logger.info(f"已保存Parquet格式: {parquet_path}")
                    
            except Exception as e:
                self.logger.error(f"保存 {format_type} 格式失败: {e}")