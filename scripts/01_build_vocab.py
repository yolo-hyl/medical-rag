"""
构建BM25词表（仅在使用自管理BM25时需要）
"""
import logging
from pathlib import Path
from datasets import load_dataset
from MedicalRag.config.loader import load_config_from_file
from MedicalRag.knowledge.bm25 import SimpleBM25Manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # 加载配置
    config = load_config_from_file("/home/weihua/medical-rag/src/MedicalRag/config/app_config.yaml")
    
    # 检查是否需要构建词表
    if config.embedding.sparse.manager == "milvus":
        logger.info("使用Milvus内置BM25，无需构建词表")
        return
    
    # 创建BM25管理器
    bm25_manager = SimpleBM25Manager(
        vocab_path=config.embedding.sparse.vocab_path,
        domain_model=config.embedding.sparse.domain_model
    )
    
    # 加载数据
    logger.info(f"加载数据: {config.data.path}")
    if config.data.format == "jsonl":
        dataset = load_dataset("json", data_files=config.data.path, split="train")
    elif config.data.format == "json":
        dataset = load_dataset("json", data_files=config.data.path, split="train")
    else:
        dataset = load_dataset(config.data.format, data_files=config.data.path, split="train")
    
    # 提取文本用于构建词表
    texts = []
    for record in dataset:
        question = record.get(config.data.question_field, "")
        answer = record.get(config.data.answer_field, "")
        texts.append(f"{question}\n\n{answer}")
    
    logger.info(f"开始构建词表，文本数量: {len(texts)}")
    
    # 构建词表
    bm25_manager.build_vocab_from_texts(texts)
    
    logger.info("词表构建完成！")

if __name__ == "__main__":
    main()