# src/medical_rag/knowledge/ingestion.py
"""
数据入库模块（使用langchain标准组件）
"""
import logging
import hashlib
from typing import List, Dict, Any, Optional
from pathlib import Path
from datasets import load_dataset, Dataset
from tqdm import tqdm
from langchain_core.documents import Document

from ..config.models import AppConfig, DataConfig
from ..core.components import KnowledgeBase

logger = logging.getLogger(__name__)

class DataProcessor:
    """数据处理器"""
    
    def __init__(self, config: DataConfig):
        self.config = config
    
    def load_dataset(self) -> Dataset:
        """加载数据集"""
        logger.info(f"加载数据集: {self.config.path}")
        
        if self.config.format in ['json', 'jsonl']:
            dataset = load_dataset('json', data_files=self.config.path, split='train')
        elif self.config.format == 'parquet':
            dataset = load_dataset('parquet', data_files=self.config.path, split='train')
        else:
            raise ValueError(f"不支持的数据格式: {self.config.format}")
        dataset = dataset.select(range(20))
        logger.info(f"成功加载数据集，共 {len(dataset)} 条记录")
        return dataset
    
    def process_records(self, dataset: Dataset) -> List[Document]:
        """处理记录，转换为langchain Document格式"""
        documents = []
        
        for i, record in enumerate(tqdm(dataset, desc="处理数据")):
            # 提取字段
            question = record.get(self.config.question_field, "")
            answer = record.get(self.config.answer_field, "")
            
            # 组合文本
            content = f"{question}\n\n{answer}"
            
            # 构建元数据
            metadata = {
                "question": question,
                "answer": answer,
                "source": record.get(self.config.source_field, self.config.default_source),
            }
            
            # 处理ID
            if self.config.id_field and self.config.id_field in record:
                doc_id = str(record[self.config.id_field])
            else:
                doc_id = hashlib.md5(question.encode('utf-8')).hexdigest()
            metadata["id"] = doc_id
            
            # 保留其他字段
            for key, value in record.items():
                if key not in [self.config.question_field, self.config.answer_field, 
                              self.config.source_field, self.config.id_field]:
                    metadata[key] = value
            
            documents.append(Document(
                page_content=content,
                metadata=metadata
            ))
        
        return documents

class IngestionPipeline:
    """数据入库流水线"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.data_processor = DataProcessor(config.data)
        self.knowledge_base = KnowledgeBase(config)
    
    def run(self, build_vocab: bool = True) -> bool:
        """运行入库流水线"""
        try:
            # 1. 加载数据
            logger.info("开始数据入库流水线...")
            dataset = self.data_processor.load_dataset()
            
            # 2. 处理数据
            documents = self.data_processor.process_records(dataset)
            logger.info(f"处理完成，共 {len(documents)} 个文档")
            
            # 3. 如果使用自管理BM25，先构建词表
            if build_vocab and self.knowledge_base.use_self_bm25:
                texts = [doc.page_content for doc in documents]
                self.knowledge_base.build_vocab_if_needed(texts)
            
            # 4. 批量插入
            batch_size = self.config.data.batch_size
            total_inserted = 0
            
            logger.info(f"开始插入文档，批次大小: {batch_size}")
            with tqdm(total=len(documents), desc="插入文档") as pbar:
                for i in range(0, len(documents), batch_size):
                    batch = documents[i:i + batch_size]
                    
                    try:
                        ids = self.knowledge_base.add_documents(batch)
                        total_inserted += len(ids)
                        pbar.update(len(batch))
                    except Exception as e:
                        logger.error(f"插入批次失败 {i//batch_size + 1}: {e}")
                        continue
            
            logger.info(f"数据入库完成！成功插入 {total_inserted} 个文档")
            return True
            
        except Exception as e:
            logger.error(f"数据入库失败: {e}")
            return False

def run_ingestion(config: AppConfig, build_vocab: bool = True) -> bool:
    """运行数据入库的便捷函数"""
    pipeline = IngestionPipeline(config)
    return pipeline.run(build_vocab)