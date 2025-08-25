import logging
from pathlib import Path
import json
from datasets import load_dataset
from MedicalRag.config.loader import ConfigLoader
from MedicalRag.core.IngestionPipeline import IngestionPipeline
import traceback
from datasets import load_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """主函数""" 
    # 1. 加载配置
    config_manager = ConfigLoader()
    data = load_dataset("json", data_files="/home/weihua/medical-rag/raw_data/raw/train/sample/qa_embed_50000.json", split="train")
    print(f"配置加载成功")
    print(f"   集合名称: {config_manager.config.milvus.collection_name}")
    
    # 3. 运行入库流水线
    print(f"\n=== 数据入库 ===")
    pipeline = IngestionPipeline(config_manager.config)
    success = pipeline.run(data)
    
    if not success:
        print("入库失败")
        return
    
    print(f"✅ 数据入库完成")

if __name__ == "__main__":
    main()