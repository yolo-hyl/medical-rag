"""
自动标注脚本
"""
import argparse
import logging
from MedicalRag.config.loader import load_config_from_file
from MedicalRag.knowledge.annotation import run_annotation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="医疗QA数据自动标注")
    parser.add_argument("--config", default="config/app_config.yaml", help="配置文件路径")
    parser.add_argument("--input", required=True, help="输入数据文件路径")
    parser.add_argument("--output", required=True, help="输出标注文件路径")
    parser.add_argument("--question-field", default="question", help="问题字段名")
    parser.add_argument("--answer-field", default="answer", help="答案字段名")
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config_from_file(args.config)
    
    logger.info(f"开始标注数据: {args.input}")
    
    # 运行标注
    success = run_annotation(
        config=config,
        data_path=args.input,
        output_path=args.output,
        question_field=args.question_field,
        answer_field=args.answer_field
    )
    
    if success:
        logger.info(f"标注完成，结果保存到: {args.output}")
    else:
        logger.error("标注失败！")

if __name__ == "__main__":
    main()