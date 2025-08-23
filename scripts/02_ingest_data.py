"""
数据入库脚本
"""
import logging
from MedicalRag.config.loader import load_config_from_file
from MedicalRag.knowledge.ingestion import run_ingestion

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # 加载配置
    config = load_config_from_file("config/app_config.yaml")
    
    logger.info("开始数据入库...")
    
    # 运行入库流水线
    success = run_ingestion(config, build_vocab=True)
    
    if success:
        logger.info("数据入库完成！")
    else:
        logger.error("数据入库失败！")

if __name__ == "__main__":
    main()