#!/usr/bin/env python3
"""
集合创建 Pipeline 测试脚本
"""

import sys
import logging
import argparse
from pathlib import Path
from MedicalRag.pipeline.ingestion.ingestion_pipeline import CollectionCreationPipeline
from MedicalRag.config.milvus_cfg import load_cfg

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_collection_creation(config_path: str, force_recreate: bool = False):
    """
    测试集合创建 Pipeline
    
    Args:
        config_path: 配置文件路径
        force_recreate: 是否强制重建集合
    """
    logger.info("=== 开始测试集合创建 Pipeline ===")
    
    try:
        # 如果需要强制重建，临时修改配置
        cfg = None
        if force_recreate:
            cfg = load_cfg(config_path)
            cfg.milvus.collection.recreate_if_exists = True
            logger.info("强制重建模式：将删除现有集合")
        
        # 初始化 Pipeline
        pipeline = CollectionCreationPipeline(
            config_path=config_path if cfg is None else None,
            cfg=cfg
        )
        
        # 显示配置信息
        logger.info(f"集合名称: {pipeline.collection_name}")
        logger.info(f"Milvus URI: {pipeline.cfg.milvus.client.uri}")
        logger.info(f"重建模式: {pipeline.cfg.milvus.collection.recreate_if_exists}")
        
        # 执行创建流程
        success = pipeline.run()
        
        if success:
            logger.info("✅ 集合创建 Pipeline 执行成功")
            
            # 获取集合信息
            collection_info = pipeline.get_collection_info()
            logger.info("集合信息:")
            for key, value in collection_info.items():
                logger.info(f"  {key}: {value}")
                
        else:
            logger.error("❌ 集合创建 Pipeline 执行失败")
            return False
        
    except Exception as e:
        logger.error(f"❌ 测试过程中出现异常: {e}")
        return False
    
    logger.info("=== 集合创建 Pipeline 测试完成 ===")
    return True


def test_connection_only(config_path: str):
    """
    仅测试连接功能
    
    Args:
        config_path: 配置文件路径
    """
    logger.info("=== 开始测试 Milvus 连接 ===")
    
    try:
        pipeline = CollectionCreationPipeline(config_path=config_path)
        
        # 仅测试连接
        if pipeline.connect():
            logger.info("✅ Milvus 连接测试成功")
            
            # 检查集合状态
            exists = pipeline.check_collection_exists()
            logger.info(f"集合 '{pipeline.collection_name}' 存在状态: {exists}")
            
            return True
        else:
            logger.error("❌ Milvus 连接测试失败")
            return False
            
    except Exception as e:
        logger.error(f"❌ 连接测试过程中出现异常: {e}")
        return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="集合创建 Pipeline 测试脚本")
    parser.add_argument(
        "-c", "--config", 
        default="src/MedicalRag/config/milvus.yaml",
        help="配置文件路径"
    )
    parser.add_argument(
        "--force-recreate", 
        action="store_true",
        help="强制重建集合（会删除现有集合）"
    )
    parser.add_argument(
        "--connection-only", 
        action="store_true",
        help="仅测试连接，不创建集合"
    )
    
    args = parser.parse_args()
    
    # 检查配置文件是否存在
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"配置文件不存在: {config_path}")
        logger.info("请确保配置文件路径正确，或使用默认配置文件")
        sys.exit(1)
    
    logger.info(f"使用配置文件: {config_path}")
    
    try:
        if args.connection_only:
            # 仅测试连接
            success = test_connection_only(str(config_path))
        else:
            # 完整的集合创建测试
            success = test_collection_creation(str(config_path), args.force_recreate)
        
        if success:
            logger.info("🎉 所有测试通过")
            sys.exit(0)
        else:
            logger.error("💥 测试失败")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("用户中断测试")
        sys.exit(1)
    except Exception as e:
        logger.error(f"未处理的异常: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()