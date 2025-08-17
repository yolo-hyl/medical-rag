import asyncio
import logging
from typing import Any, Dict, List, Optional, Union
from datasets import Dataset, load_dataset
from MedicalRag.config.data.annotator import load_config, LabelingConfig
from MedicalRag.data.annotator.huatuo_qa_labeler import QALabeler
from MedicalRag.core.llm import create_llm_client


def setup_logging(config: Dict[str, Any]):
    """设置日志配置"""
    log_config = config.get('logging', {})
    
    logging.basicConfig(
        level=getattr(logging, log_config.get('level', 'INFO')),
        format=log_config.get('format', '%(asctime)s - %(levelname)s - %(message)s'),
        handlers=[
            logging.FileHandler(log_config.get('log_file', 'labeling.log')),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def load_dataset_from_config(config: Dict[str, Any]) -> Dataset:
    """根据配置加载数据集"""
    dataset_config = config.get('dataset', {})
    dataset_type = dataset_config.get('type', 'json')
    
    if dataset_type in ['json', 'csv', 'parquet']:
        data_files = dataset_config.get('data_files')
        if not data_files:
            raise ValueError("data_files配置不能为空")
        
        split = dataset_config.get('split', 'train')
        return load_dataset(dataset_type, data_files=data_files, split=split)
    
    elif dataset_type == 'huggingface':
        dataset_name = dataset_config.get('name')
        if not dataset_name:
            raise ValueError("huggingface数据集需要指定name")
        
        split = dataset_config.get('split', 'train')
        return load_dataset(dataset_name, split=split)
    
    else:
        raise ValueError(f"不支持的数据集类型: {dataset_type}")


# 使用示例
async def main(config_path: str = "config.yaml"):
    """主函数"""
    
    # 1. 加载配置
    config = load_config(config_path)
    
    # 2. 设置日志
    logger = setup_logging(config)
    logger.info(f"从配置文件加载: {config_path}")
    
    # 3. 加载数据集
    logger.info("加载数据集...")
    dataset = load_dataset_from_config(config)
    logger.info(f"数据集加载完成，共 {len(dataset)} 条数据")
    
    # 4. 创建标注配置
    labeling_config = LabelingConfig.from_config(config)
    
    # 5. 创建LLM客户端
    logger.info("创建LLM客户端...")
    llm_client = create_llm_client(config)
    
    # 6. 创建标注器并开始标注
    labeler = QALabeler(llm_client, labeling_config, logger)
    
    # 标注数据集（支持断点续标）
    labeled_dataset = await labeler.label_dataset(dataset)
    
    logger.info("标注任务完成！")
    
    return labeled_dataset


if __name__ == "__main__":
    import sys
    
    # 支持命令行指定配置文件
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    
    # 运行标注任务
    asyncio.run(main(config_path))