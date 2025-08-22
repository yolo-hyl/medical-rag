"""
查询 Pipeline 测试脚本
支持使用独立的搜索配置文件
"""

import sys
import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any
from MedicalRag.pipeline.query.query_pipeline import QueryPipeline

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)
# 关闭 httpx 的日志
logging.getLogger("httpx").setLevel(logging.WARNING)

def print_search_results(queries: List[str], results: List[List[Dict[str, Any]]]):
    """
    格式化打印搜索结果
    
    Args:
        queries: 查询列表
        results: 搜索结果
    """
    for i, (query, hits) in enumerate(zip(queries, results)):
        if len(queries):
            print(f"查询 [{i+1}]: {query}")
        
        if not hits:
            print("查询无结果")
            continue
            
        for j, hit in enumerate(hits):
            print(f"\n结果 [{j+1}]:")
            print(f"  ID: {hit.get('id', 'N/A')}")
            print(f"  距离: {hit.get('distance', 'N/A'):.4f}")
            print(f"  评分: {hit.get('score', 'N/A'):.4f}")
            
            # 打印配置的输出字段
            question = hit.get('question', '')
            answer = hit.get('answer', '')
            if question:
                print(f"  问题: {question[:100]}{'...' if len(question) > 100 else ''}")
            if answer:
                print(f"  答案: {answer[:100]}{'...' if len(answer) > 100 else ''}")
            
            # 打印其他字段
            for key, value in hit.items():
                if key not in ['id', 'distance', 'score', 'question', 'answer']:
                    print(f"  {key}: {value}")


def test_pipeline_setup(config_path: str) -> QueryPipeline:
    """
    测试 Pipeline 设置
    
    Args:
        config_path: 配置文件路径
        use_search_config: 是否使用搜索专用配置
        
    Returns:
        QueryPipeline: 设置好的查询 Pipeline
    """
    logger.info(f"=== pipeline setup ===")
    
    try:
        # 从专门的配置类型创建 Pipeline
        pipeline = QueryPipeline.create_from_search_config(config_path)
        # 也可以从默认配置加载
        # pipeline = QueryPipeline(config_path=config_path)
        
        # 显示配置信息
        logger.info(f"集合名称: {pipeline.collection_name}")
        logger.info(f"Milvus URI: {pipeline.cfg.milvus.client.uri}")
        logger.info(f"嵌入模型: {pipeline.cfg.embedding.dense.provider}/{pipeline.cfg.embedding.dense.model}")
        logger.info(f"稀疏向量: {pipeline.cfg.embedding.sparse_bm25.vocab_path}")
        logger.info(f"配置类型: {pipeline._config_type}")
        
        # 执行设置
        success = pipeline.setup()
        
        if success:
            logger.info("查询 Pipeline 设置成功")
            return pipeline
        else:
            logger.error("查询 Pipeline 设置失败")
            return None
            
    except Exception as e:
        logger.error(f"设置过程中出现异常: {e}")
        return None



def channel_update(pipeline: QueryPipeline):
    """
    测试动态更新搜索通道
    
    Args:
        pipeline: 查询 Pipeline
    """
    logger.info("=== 动态更新搜索通道 ===")
    
    # 获取当前配置
    original_config = pipeline.get_search_config()
    logger.info("原始通道配置:")
    for ch in original_config['channels']:
        if ch['enabled']:
            logger.info(f"  {ch['name']}: enabled={ch['enabled']}, weight={ch['weight']}")
    
    # 测试更新通道权重
    channel_updates = [
        {"name": "sparse_doc", "weight": 0.6},
        {"name": "sparse_q", "weight": 0.4}
    ]
    
    success = pipeline.update_search_channels(channel_updates)
    if success:
        # 显示更新后的配置
        updated_config = pipeline.get_search_config()
        logger.info("更新后通道配置:")
        for ch in updated_config['channels']:
            if ch['enabled']:
                logger.info(f"  {ch['name']}: enabled={ch['enabled']}, weight={ch['weight']}")
    else:
        logger.error("通道配置更新失败")
        
    return success


def config_dict_creation():
    """
    测试从配置字典创建Pipeline
    """
    logger.info("=== 测试从配置字典创建Pipeline ===")
    
    try:
        # 示例配置字典
        config_dict = {
            "milvus": {
                "client": {
                    "uri": "http://localhost:19530",
                    "token": "root:Milvus",
                    "db_name": "",
                    "timeout_ms": 30000,
                    "tls": False,
                    "tls_verify": False
                },
                "collection": {
                    "name": "qa_knowledge",
                    "description": "RAG QA knowledge base",
                    "load_on_start": True
                }
            },
            "search": {
                "default_limit": 5,
                "output_fields": ["question", "answer"],
                "pagination": {"page_size": 10, "max_pages": 100},
                "rrf": {"enabled": False, "k": 100},
                "channels": [
                    {
                        "name": "sparse_doc",
                        "field": "sparse_vec_qa",
                        "enabled": True,
                        "kind": "sparse_document",
                        "metric_type": "IP",
                        "limit": 5,
                        "params": {"drop_ratio_search": 0.0},
                        "expr": "",
                        "weight": 1.0
                    }
                ]
            },
            "embedding": {
                "dense": {
                    "provider": "ollama",
                    "model": "bge-m3:latest",
                    "base_url": "http://172.16.40.51:11434",
                    "dim": 1024,
                    "normalize": False,
                    "prefixes": {"query": "", "document": ""}
                },
                "sparse_bm25": {
                    "vocab_path": "vocab.pkl.gz",
                    "domain_model": "medicine",
                    "prune_empty_sparse": True,
                    "empty_sparse_fallback": {"0": 0.0},
                    "k1": 1.5,
                    "b": 0.75
                }
            }
        }
        
        # 从字典创建Pipeline
        pipeline = QueryPipeline.create_from_config_dict(config_dict)
        logger.info("从配置字典创建Pipeline成功")
        logger.info(f"配置类型: {pipeline._config_type}")
        
        return pipeline
    except Exception as e:
        logger.error(f"从配置字典创建Pipeline失败: {e}")
        return None


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="查询 Pipeline 测试脚本")
    parser.add_argument(
        "-c", "--config", 
        default="src/MedicalRag/config/default.yaml",
        help="核心配置文件路径"
    )
    parser.add_argument(
        "-s", "--search-config",
        default="src/MedicalRag/config/search/search_answer.yaml",
        help="搜索配置文件路径"
    )
    parser.add_argument(
        "--test-config-dict",
        action="store_true",
        help="测试从配置字典创建Pipeline，用于在使用工具时动态传入search配置"
    )
    
    args = parser.parse_args()
    
    query = "曲匹地尔用于治疗什么疾病？"
    # batch_queries = ["请描述呼吸衰竭的诊断方法", "巨肠症是什么疾病？可以治愈吗？", "最广泛的传染病"]
    
    # 选择配置文件，也可以不传，default配置也需要配置一些默认search规则
    config_path = args.search_config
    
    # 检查配置文件是否存在
    config_file = Path(config_path)
    if not config_file.exists():
        logger.error(f"配置文件不存在: {config_path}")
        logger.info("请确保配置文件路径正确")
        sys.exit(1)

    # 测试从配置字典创建Pipeline
    if args.test_config_dict:
        pipeline = config_dict_creation()
    else:
        # 设置 Pipeline
        pipeline = test_pipeline_setup(str(config_path))
    
    # 单个查询
    print(f"\n{'='*50}")
    print(f"单次查询")
    print(f"{'='*50}")
    results = pipeline.search_single(query, expr_vars={}, limit=2)
    print_search_results([query], [results])
    
    # 批量查询
    # print(f"\n{'='*50}")
    # print(f"批量查询")
    # print(f"{'='*50}")
    # results = pipeline.search(batch_queries, expr_vars={})
    # print_search_results(batch_queries, results)
    
    # # 过滤查询
    # print(f"\n{'='*50}")
    # print(f"过滤查询")
    # print(f"{'='*50}")
    # test_cases = [
    #     {"sparse_q": "source_name == 'qa'"},
    # ]
    # for i, expr_vars in enumerate(test_cases):
    #     print(f"当前表达式：{expr_vars}")
    #     results = pipeline.search_single(query, expr_vars=expr_vars)
    #     print_search_results([query], [results])
    
    # # 自定义通道更新
    # channel_update(pipeline)


if __name__ == "__main__":
    main()