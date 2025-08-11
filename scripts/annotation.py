#!/usr/bin/env python3
"""
QA标注测试脚本 - 使用MedicalRag包中的QAAnnotator
"""

import json
import logging
import argparse
from pathlib import Path
from typing import List
import sys

from MedicalRag.data.annotator.qa_annotator import QAAnnotator
from MedicalRag.data.loader.huatuo_qa_jsonl_loader import JSONLLoader
from MedicalRag.core.base.BaseAnnotator import BaseAnnotatorCfg
from MedicalRag.core.base.BaseClient import LLMHttpClientCfg
from MedicalRag.core.llm.client import OllamaClientCfg, OllamaClient
from MedicalRag.schemas.metadata import QAAnnotationResponse

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="QA标注测试脚本")
    parser.add_argument("input_file", help="输入JSONL文件路径")
    parser.add_argument("-o", "--output-dir", default="./output", help="输出目录")
    parser.add_argument("--ollama-url", default="http://localhost:11434", help="Ollama服务地址")
    parser.add_argument("--model", default="qwen3:32b", help="使用的模型名称")
    parser.add_argument("--batch-size", type=int, default=5, help="批处理大小")
    parser.add_argument("--max-retries", type=int, default=3, help="最大重试次数")
    
    args = parser.parse_args()
    
    # 验证输入文件
    input_file = Path(args.input_file)
    print(input_file)
    if not input_file.exists():
        logger.error(f"输入文件不存在: {input_file}")
        return
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 配置LLM客户端
    llm_config = OllamaClientCfg(
        base_url=args.ollama_url,
        model=args.model,
        thinking_model=True
    )
    
    # 配置标注器
    annotator_config = BaseAnnotatorCfg(
        llm_config=llm_config,
        max_retries=args.max_retries,
        batch_size=args.batch_size,
        save_intermediate=True,
        output_file_path=str(output_dir),
        system_prompt="你是一个专业的医疗问答标注专家，请严格按照要求输出JSON格式的标注结果。"
    )
    
    # 初始化组件
    loader = JSONLLoader()
    annotator = QAAnnotator(annotator_config)
    ollama_cfg = OllamaClientCfg()
    annotator.client = OllamaClient(ollama_cfg)
    # 检查服务连接
    if not annotator.client.health_check():
        logger.error(f"无法连接到Ollama服务: {args.ollama_url}")
        logger.error("请确保Ollama服务正在运行")
        return
    
    logger.info(f"连接到Ollama服务成功，可用模型: {annotator.client.list_models()}")
    
    try:
        # 加载JSONL文件
        logger.info(f"开始加载JSONL文件: {input_file}")
        qa_pairs = loader.load_file(input_file)
        
        if not qa_pairs:
            logger.error("未能加载到任何QA对")
            return
        
        logger.info(f"成功加载 {len(qa_pairs)} 个QA对")
        
        # 转换为标注器需要的格式
        qa_data = []
        for i, (question, answer) in enumerate(qa_pairs):
            qa_data.append({
                "question": question,
                "answer": answer
            })
        
        # 批量标注
        logger.info("开始批量标注...")
        results = annotator.annotate_batch(
            content=qa_data,
            base_model=QAAnnotationResponse
        )
        
        if results:
            # 保存最终结果
            output_file = output_dir / f"qa_annotations_{input_file.stem}.json"
            annotator.save_annotations(results, output_file)
            logger.info(f"标注完成，共处理 {len(results)} 个QA对")
            
            # 输出统计信息
            stats = annotator.get_stats()
            logger.info(f"成功率: {stats['success'] / max(stats['total'], 1) * 100:.1f}%")
        else:
            logger.error("标注过程失败，未生成任何结果")
        
    except Exception as e:
        logger.error(f"处理过程中出错: {e}")
        raise


def test_single_qa():
    """测试单个QA对标注功能"""
    logger.info("=== 测试单个QA对标注 ===")
    
    # 配置
    llm_config = LLMHttpClientCfg(
        base_url="http://localhost:11434",
        model="qwen3:32b",
        thinking_model=True
    )
    
    annotator_config = BaseAnnotatorCfg(
        llm_config=llm_config,
        max_retries=3,
        batch_size=1,
        save_intermediate=False,
        output_file_path="./test_output",
        system_prompt="你是一个专业的医疗问答标注专家。"
    )
    
    annotator = QAAnnotator(annotator_config)
    
    # 测试数据
    test_data = {
        "id": 1,
        "question": "自体及异体CIK的临床表现有些什么？",
        "answer": "低热"
    }
    
    try:
        result = annotator.annotate_single(
            single_data=test_data,
            base_model=QAAnnotationResponse,
            system_prompt=annotator_config.system_prompt
        )
        
        if result:
            logger.info("单个QA标注成功:")
            logger.info(f"问题: {result.question}")
            logger.info(f"答案: {result.answers}")
            logger.info(f"科室: {[dept.name for dept in result.departments]}")
            logger.info(f"类别: {[cat.name for cat in result.categories]}")
            logger.info(f"推理: {result.reasoning}")
            logger.info(f"置信度: {result.confidence}")
        else:
            logger.error("单个QA标注失败")
            
    except Exception as e:
        logger.error(f"测试单个QA标注时出错: {e}")


if __name__ == "__main__":
    # 检查参数，如果没有提供文件路径则运行单个测试
    if len(sys.argv) == 1:
        test_single_qa()
    else:
        main()