import asyncio
import logging
from datasets import load_dataset
from MedicalRag.config.loader import ConfigLoader
from MedicalRag.core.IngestionPipeline import IngestionPipeline
from MedicalRag.core.KnowledgeBase import MedicalHybridKnowledgeBase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

async def main():
    """主函数"""
    # 1. 加载配置
    config_manager = ConfigLoader()
    # config_manager.change({"milvus.auto_id": True})  # 自动管理id
    cfg = config_manager.config
    data = load_dataset("json", data_files="data/qa_50000.jsonl", split="train")
    data = data.select(range(100))  # 快速体验插入100条数据
    print(f"配置加载成功")
    print(f"   集合名称: {cfg.milvus.collection_name}")

    # 2. 创建知识库，交给入库流水线
    kb = MedicalHybridKnowledgeBase(cfg.milvus, cfg.embedding)
    pipeline = IngestionPipeline(cfg.data, kb)

    # 3. 运行入库流水线
    print(f"\n=== 数据入库 ===")
    try:
        success = await pipeline.run(data)
    finally:
        await kb.close()

    if not success:
        print("入库失败")
        return

    print(f"✅ 数据入库完成")

if __name__ == "__main__":
    asyncio.run(main())
