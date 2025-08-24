import logging
from pathlib import Path
import json
from datasets import load_dataset
from MedicalRag.config.loader import load_config_from_file, ConfigLoader
from MedicalRag.core.IngestionPipeline import IngestionPipeline
import traceback
from datasets import load_dataset
from MedicalRag.core.KnowledgeBase import MedicalHybridKnowledgeBase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """主函数"""
    print("=== 多向量字段混合检索演示 ===\n")
    
    try:
        # 1. 加载配置
        milvus_config = ConfigLoader.load_milvus_config(file_path="/home/weihua/medical-rag/scripts/search/milvus.yaml")
        embedding_config = ConfigLoader.load_embedding_config(file_path="/home/weihua/medical-rag/scripts/search/embedding.yaml")
        print(f"配置加载成功")
        print(f"   集合名称: {milvus_config.collection_name}")
        
        # 4. 创建知识库实例
        kb = MedicalHybridKnowledgeBase(milvus_config=milvus_config, embedding_config=embedding_config)
        kb.initialize_collection(drop_old=False)  # 集合已存在
        
        collection_info = kb.get_collection_info()
        print(f"✅ 知识库初始化完成")
        print(f"   向量字段: {collection_info['vector_fields']}")
        
        search_config = ConfigLoader.load_search_config("/home/weihua/medical-rag/scripts/search/search.yaml")
        
        # 5. 测试不同的检索策略
        print(f"\n=== 混合检索测试 ===")
        test_queries = "高血压症状"
        print(f"\n=============== weights = [0.5, 0.3, 0.2] ===============")
        
        weights = [0.5, 0.3, 0.2]  # 更重视问题匹配
        results = kb.hybrid_search(
            query=test_queries,
            config=search_config
        )
        for i, doc in enumerate(results):
            summary = doc.metadata.get('summary', '')
            document = doc.metadata.get('document', '')
            print(f"  {i+1}. {summary} [{document}]")
        
        print(f"\n==========================================================")
        
        
        print(f"\n=============== weights = [0.3, 0.4, 0.3] ===============")
        test_queries = "高血压症状"
        search_config.weights = [0.3, 0.4, 0.3]  # 更重视问题匹配
        results = kb.hybrid_search(
            query=test_queries,
            config=search_config
        )
        for i, doc in enumerate(results):
            summary = doc.metadata.get('summary', '')
            document = doc.metadata.get('document', '')
            print(f"  {i+1}. {summary} [{document}]")
        
        print(f"\n==========================================================")
        
        
        # RRF重排对比
        search_config.ranker_type = "rrf"
        search_config.rrf_k = 100
        results_rrf = kb.hybrid_search(
            query=test_queries,
            config=search_config
        )
        
        print(f"\n========================= RRF 重排 =================================")
        for i, doc in enumerate(results_rrf):
            summary = doc.metadata.get('summary', '')
            document = doc.metadata.get('document', '')
            print(f"  {i+1}. {summary} [{document}]")
        print(f"\n==========================================================")
        
    except Exception as e:
        logger.error(f"演示运行失败: {e}")
        print(traceback(e))
        raise

if __name__ == "__main__":
    main()