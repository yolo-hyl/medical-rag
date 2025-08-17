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


def print_search_results(queries: List[str], results: List[List[Dict[str, Any]]]):
    """
    格式化打印搜索结果
    
    Args:
        queries: 查询列表
        results: 搜索结果
    """
    for i, (query, hits) in enumerate(zip(queries, results)):
        print(f"\n{'='*50}")
        print(f"查询 [{i+1}]: {query}")
        print(f"{'='*50}")
        
        if not hits:
            print("❌ 无结果")
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


def test_pipeline_setup(config_path: str, use_search_config: bool = False) -> QueryPipeline:
    """
    测试 Pipeline 设置
    
    Args:
        config_path: 配置文件路径
        use_search_config: 是否使用搜索专用配置
        
    Returns:
        QueryPipeline: 设置好的查询 Pipeline
    """
    config_type = "搜索专用" if use_search_config else "完整"
    logger.info(f"=== 开始测试查询 Pipeline 设置（{config_type}配置） ===")
    
    try:
        # 根据配置类型创建 Pipeline
        if use_search_config:
            pipeline = QueryPipeline.create_from_search_config(config_path)
        else:
            pipeline = QueryPipeline(config_path=config_path)
        
        # 显示配置信息
        logger.info(f"集合名称: {pipeline.collection_name}")
        logger.info(f"Milvus URI: {pipeline.cfg.milvus.client.uri}")
        logger.info(f"嵌入模型: {pipeline.cfg.embedding.dense.provider}/{pipeline.cfg.embedding.dense.model}")
        logger.info(f"稀疏向量: {pipeline.cfg.embedding.sparse_bm25.vocab_path}")
        logger.info(f"配置类型: {pipeline._config_type}")
        
        # 执行设置
        success = pipeline.setup()
        
        if success:
            logger.info("✅ 查询 Pipeline 设置成功")
            
            # 显示搜索配置
            search_config = pipeline.get_search_config()
            logger.info("搜索配置:")
            logger.info(f"  默认限制: {search_config['default_limit']}")
            logger.info(f"  输出字段: {search_config['output_fields']}")
            logger.info(f"  RRF启用: {search_config['rrf_enabled']}")
            
            enabled_channels = [ch for ch in search_config['channels'] if ch['enabled']]
            logger.info(f"  启用通道数: {len(enabled_channels)}")
            for ch in enabled_channels:
                logger.info(f"    - {ch['name']} ({ch['kind']}, weight: {ch['weight']})")
            
            return pipeline
        else:
            logger.error("❌ 查询 Pipeline 设置失败")
            return None
            
    except Exception as e:
        logger.error(f"❌ 设置过程中出现异常: {e}")
        return None


def test_single_query(pipeline: QueryPipeline, query: str, expr_vars: Dict[str, Any] = None):
    """
    测试单个查询
    
    Args:
        pipeline: 查询 Pipeline
        query: 查询文本
        expr_vars: 表达式变量
    """
    logger.info(f"=== 测试单个查询: {query} ===")
    
    try:
        results = pipeline.search_single(query, expr_vars=expr_vars)
        
        logger.info(f"查询完成，返回 {len(results)} 个结果")
        print_search_results([query], [results])
        
        return True
    except Exception as e:
        logger.error(f"❌ 单个查询测试失败: {e}")
        return False


def test_batch_query(pipeline: QueryPipeline, queries: List[str], expr_vars: Dict[str, Any] = None):
    """
    测试批量查询
    
    Args:
        pipeline: 查询 Pipeline
        queries: 查询列表
        expr_vars: 表达式变量
    """
    logger.info(f"=== 测试批量查询，共 {len(queries)} 个查询 ===")
    
    try:
        results = pipeline.search(queries, expr_vars=expr_vars)
        
        total_results = sum(len(r) for r in results)
        logger.info(f"批量查询完成，总共返回 {total_results} 个结果")
        
        print_search_results(queries, results)
        
        return True
    except Exception as e:
        logger.error(f"❌ 批量查询测试失败: {e}")
        return False


def test_filtered_query(pipeline: QueryPipeline, query: str):
    """
    测试带过滤条件的查询
    
    Args:
        pipeline: 查询 Pipeline
        query: 查询文本
    """
    logger.info(f"=== 测试带过滤条件的查询: {query} ===")
    
    try:
        # 测试不同的过滤条件
        test_cases = [
            {"src": "huatuo_qa"},
            {"dept": "0"},
            {"src": "huatuo_qa", "dept": "1"}
        ]
        
        for i, expr_vars in enumerate(test_cases):
            logger.info(f"测试过滤条件 [{i+1}]: {expr_vars}")
            results = pipeline.search_single(query, expr_vars=expr_vars)
            logger.info(f"  结果数量: {len(results)}")
            
            if results:
                logger.info(f"  第一个结果ID: {results[0].get('id', 'N/A')}")
        
        return True
    except Exception as e:
        logger.error(f"❌ 过滤查询测试失败: {e}")
        return False


def test_channel_update(pipeline: QueryPipeline):
    """
    测试动态更新搜索通道
    
    Args:
        pipeline: 查询 Pipeline
    """
    logger.info("=== 测试动态更新搜索通道 ===")
    
    try:
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
            logger.info("✅ 通道配置更新成功")
            
            # 显示更新后的配置
            updated_config = pipeline.get_search_config()
            logger.info("更新后通道配置:")
            for ch in updated_config['channels']:
                if ch['enabled']:
                    logger.info(f"  {ch['name']}: enabled={ch['enabled']}, weight={ch['weight']}")
        else:
            logger.error("❌ 通道配置更新失败")
            
        return success
    except Exception as e:
        logger.error(f"❌ 通道更新测试失败: {e}")
        return False


def test_config_dict_creation():
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
                "expr_template": "",
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
                        "expr_template": "",
                        "weight": 1.0
                    }
                ]
            },
            "embedding": {
                "dense": {
                    "provider": "ollama",
                    "model": "nomic-embed-text",
                    "base_url": "http://172.16.40.51:11434",
                    "dim": 768,
                    "normalize": False,
                    "prefixes": {"query": "search_query:", "document": "search_document:"}
                },
                "sparse_bm25": {
                    "vocab_path": "vocab.pkl.gz",
                    "domain_model": "medicine",
                    "prune_empty_sparse": True,
                    "empty_sparse_fallback": {"0": 0.0},
                    "stopwords": ["什么", "？", "是", "如何"],
                    "k1": 1.5,
                    "b": 0.75
                }
            }
        }
        
        # 从字典创建Pipeline
        pipeline = QueryPipeline.create_from_config_dict(config_dict)
        logger.info("✅ 从配置字典创建Pipeline成功")
        logger.info(f"配置类型: {pipeline._config_type}")
        
        return True
    except Exception as e:
        logger.error(f"❌ 从配置字典创建Pipeline失败: {e}")
        return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="查询 Pipeline 测试脚本")
    parser.add_argument(
        "-c", "--config", 
        default="src/MedicalRag/config/milvus.yaml",
        help="配置文件路径"
    )
    parser.add_argument(
        "-s", "--search-config",
        default="src/MedicalRag/config/search/search_answer.yaml",
        help="搜索配置文件路径"
    )
    parser.add_argument(
        "--use-search-config",
        action="store_true",
        help="使用搜索专用配置文件"
    )
    parser.add_argument(
        "-q", "--query",
        default="梅毒",
        help="测试查询文本"
    )
    parser.add_argument(
        "--batch-queries",
        nargs="+",
        default=["梅毒", "巨肠症是什么东西？", "最广泛的性病是什么？"],
        help="批量查询列表"
    )
    parser.add_argument(
        "--setup-only", 
        action="store_true",
        help="仅测试Pipeline设置，不执行查询"
    )
    parser.add_argument(
        "--test-filter", 
        action="store_true",
        help="测试过滤查询"
    )
    parser.add_argument(
        "--test-update", 
        action="store_true",
        help="测试动态更新通道配置"
    )
    parser.add_argument(
        "--test-config-dict",
        action="store_true",
        help="测试从配置字典创建Pipeline"
    )
    
    args = parser.parse_args()
    
    # 选择配置文件
    config_path = args.search_config if args.use_search_config else args.config
    
    # 检查配置文件是否存在
    config_file = Path(config_path)
    if not config_file.exists():
        logger.error(f"配置文件不存在: {config_path}")
        logger.info("请确保配置文件路径正确")
        sys.exit(1)
    
    config_type = "搜索专用" if args.use_search_config else "完整"
    logger.info(f"使用{config_type}配置文件: {config_path}")
    
    try:
        success_count = 0
        total_tests = 0
        
        # 测试从配置字典创建Pipeline
        if args.test_config_dict:
            total_tests += 1
            if test_config_dict_creation():
                success_count += 1
        
        # 设置 Pipeline
        pipeline = test_pipeline_setup(str(config_path), args.use_search_config)
        if not pipeline:
            logger.error("Pipeline 设置失败，退出测试")
            sys.exit(1)
        
        if args.setup_only:
            logger.info("🎉 Pipeline 设置测试完成")
            sys.exit(0)
        
        # 测试单个查询
        total_tests += 1
        if test_single_query(pipeline, args.query, expr_vars={'src': 'huatuo'}):
            success_count += 1
        
        # 测试批量查询
        total_tests += 1
        if test_batch_query(pipeline, args.batch_queries, expr_vars={'src': 'huatuo'}):
            success_count += 1
        
        # 测试过滤查询
        if args.test_filter:
            total_tests += 1
            if test_filtered_query(pipeline, args.query):
                success_count += 1
        
        # 测试通道更新
        if args.test_update:
            total_tests += 1
            if test_channel_update(pipeline):
                success_count += 1
        
        # 输出测试结果
        logger.info(f"=== 测试完成: {success_count}/{total_tests} 通过 ===")
        
        if success_count == total_tests:
            logger.info("🎉 所有测试通过")
            sys.exit(0)
        else:
            logger.error("💥 部分测试失败")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("用户中断测试")
        sys.exit(1)
    except Exception as e:
        logger.error(f"未处理的异常: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()