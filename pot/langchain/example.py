#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
标注框架使用示例
展示如何使用可扩展的标注框架处理不同类型的数据
"""

import asyncio
import json
from pathlib import Path
from base import (
    AnnotationFramework, AnnotationConfig, 
    TextClassificationAnnotator, DataItem
)

# ==================== 示例1: 医学QA数据标注 ====================

async def example_medical_qa_annotation():
    """医学QA数据标注示例"""
    print("=== 示例1: 医学QA数据标注 ===")
    
    # 配置
    config = AnnotationConfig(
        model_base_url = "172.16.40.51:11434",
        model_backend="ollama",
        model_name="qwen3:32b",
        temperature=0.1,
        max_concurrent=3,  # 根据你的硬件调整
        batch_size=5,
        cache_enabled=True
    )
    
    # 创建框架
    framework = AnnotationFramework(config)
    
    # 测试数据
    qa_data = {
        "questions": [
            ["口干的治疗方案是什么?", "请描述口干的治疗方案"],
            ["高血压的症状有哪些?"],
            ["如何预防糖尿病?"]
        ],
        "answers": [
            "口干症的治疗包括病因治疗和对症治疗",
            "高血压的症状包括头痛、头晕、心悸等",
            "预防糖尿病需要控制饮食、适量运动、定期检查"
        ]
    }
    
    # 执行标注
    results = await framework.annotate(
        data_source=qa_data,
        loader_type="qa",
        annotator_type="medical_qa",
        output_path=Path("medical_qa_results.json")
    )
    
    # 显示结果
    for i, result in enumerate(results):
        print(f"\n第{i+1}个QA对:")
        print(f"问题: {result.original_data.metadata['question']}")
        print(f"科室分类: {result.annotations.get('departments')}")
        print(f"问题类别: {result.annotations.get('categories')}")
        print(f"处理时间: {result.processing_time:.2f}s")

# ==================== 示例2: 文本文件批量标注 ====================

async def example_text_file_annotation():
    """文本文件批量标注示例"""
    print("\n=== 示例2: 文本文件批量标注 ===")
    
    # 创建测试文本文件
    test_text = """心肌梗死是冠心病的严重表现形式
主要由冠状动脉粥样硬化引起

糖尿病是一种慢性代谢性疾病
需要长期监测血糖水平

高血压是心血管疾病的重要危险因素
建议定期测量血压"""
    
    text_file = Path("test_medical_texts.txt")
    with open(text_file, 'w', encoding='utf-8') as f:
        f.write(test_text)
    
    # 配置
    config = AnnotationConfig(
        model_base_url = "172.16.40.51:11434",
        model_backend="ollama",
        model_name="qwen3:32b",
        max_concurrent=2,
        batch_size=3
    )
    
    framework = AnnotationFramework(config)
    
    # 注册自定义文本分类标注器
    classification_schema = {
        "medical_domain": ["心血管", "内分泌", "呼吸系统", "消化系统", "其他"],
        "content_type": ["疾病描述", "症状说明", "治疗建议", "预防措施", "检查指导"]
    }
    
    text_annotator = TextClassificationAnnotator(
        model_backend=framework.model_backend,
        config=config,
        classification_schema=classification_schema
    )
    
    framework.register_annotator("medical_text", text_annotator)
    
    # 执行标注
    results = await framework.annotate(
        data_source=text_file,
        loader_type="txt",
        annotator_type="medical_text",
        output_path=Path("text_classification_results.json")
    )
    
    # 显示结果
    for i, result in enumerate(results):
        print(f"\n文本段落{i+1}:")
        print(f"内容: {result.original_data.content[:50]}...")
        print(f"分类结果: {result.annotations}")
    
    # 清理
    text_file.unlink()

# ==================== 示例3: 名词解释对标注 ====================

async def example_term_pair_annotation():
    """名词解释对标注示例"""
    print("\n=== 示例3: 名词解释对标注 ===")
    
    # 测试数据
    term_data = {
        "心肌梗死": "心肌梗死是指心肌细胞因缺血缺氧而发生坏死的疾病",
        "高血压": "高血压是指动脉血压持续升高的慢性疾病",
        "糖尿病": "糖尿病是一组以高血糖为特征的代谢性疾病"
    }
    
    config = AnnotationConfig(
        model_base_url = "172.16.40.51:11434",
        model_backend="ollama",
        model_name="qwen3:32b",
        max_concurrent=2
    )
    
    framework = AnnotationFramework(config)
    
    # 创建医学术语分类标注器
    class MedicalTermAnnotator(TextClassificationAnnotator):
        def __init__(self, model_backend, config):
            schema = {
                "medical_specialty": ["心血管科", "内分泌科", "神经科", "消化科", "呼吸科", "其他"],
                "term_type": ["疾病名称", "症状", "检查项目", "药物", "治疗方法", "解剖结构"],
                "complexity_level": ["基础", "中级", "高级", "专业"]
            }
            super().__init__(model_backend, config, schema)
        
        def create_prompt(self, data_item):
            term = data_item.metadata.get("term", "")
            definition = data_item.metadata.get("definition", "")
            
            return f"""
请对以下医学术语进行分类：

术语: {term}
解释: {definition}

请根据以下标准分类：

医学专科(0-5): 0-心血管科 1-内分泌科 2-神经科 3-消化科 4-呼吸科 5-其他
术语类型(0-5): 0-疾病名称 1-症状 2-检查项目 3-药物 4-治疗方法 5-解剖结构  
复杂程度(0-3): 0-基础 1-中级 2-高级 3-专业

输出JSON格式：
```json
{{
    "medical_specialty": [index],
    "term_type": [index],
    "complexity_level": [index]
}}
```
"""
    
    term_annotator = MedicalTermAnnotator(framework.model_backend, config)
    framework.register_annotator("medical_term", term_annotator)
    
    # 执行标注
    results = await framework.annotate(
        data_source=term_data,
        loader_type="term_pair",
        annotator_type="medical_term",
        output_path=Path("term_classification_results.json")
    )
    
    # 显示结果
    for result in results:
        print(f"\n术语: {result.original_data.metadata['term']}")
        print(f"分类结果: {result.annotations}")

# ==================== 示例4: 高性能批量处理 ====================

async def example_high_performance_batch():
    """高性能批量处理示例"""
    print("\n=== 示例4: 高性能批量处理 ===")
    
    # 模拟大量数据
    large_qa_data = {
        "questions": [[f"问题{i}的内容是什么?"] for i in range(20)],
        "answers": [f"这是问题{i}的详细答案..." for i in range(20)]
    }
    
    # 高性能配置
    high_perf_config = AnnotationConfig(
        model_base_url = "172.16.40.51:11434",
        model_backend="ollama",
        model_name="qwen3:32b",
        temperature=0.1,
        max_concurrent=8,  # 增加并发数
        batch_size=5,     # 优化批次大小
        cache_enabled=True,
        max_retries=2
    )
    
    framework = AnnotationFramework(high_perf_config)
    
    import time
    start_time = time.time()
    
    results = await framework.annotate(
        data_source=large_qa_data,
        loader_type="qa",
        annotator_type="medical_qa",
        output_path=Path("batch_results.json")
    )
    
    total_time = time.time() - start_time
    
    print(f"批量处理完成:")
    print(f"总数据量: {len(results)}")
    print(f"总处理时间: {total_time:.2f}s")
    print(f"平均每条: {total_time/len(results):.2f}s")
    print(f"处理速度: {len(results)/total_time:.2f} items/s")

# ==================== 示例5: 自定义数据加载器 ====================

class CSVDataLoader:
    """CSV数据加载器示例"""
    
    async def load_data(self, source):
        import pandas as pd
        
        df = pd.read_csv(source)
        
        for _, row in df.iterrows():
            yield DataItem(
                content=f"Question: {row['question']}\nAnswer: {row['answer']}",
                metadata={
                    "question": row['question'],
                    "answer": row['answer'],
                    "type": "csv_qa"
                }
            )
    
    def get_supported_formats(self):
        return ["csv"]

async def example_custom_loader():
    """自定义加载器示例"""
    print("\n=== 示例5: 自定义CSV数据加载器 ===")
    
    # 创建测试CSV
    import pandas as pd
    
    csv_data = pd.DataFrame({
        'question': ['什么是高血压?', '糖尿病如何预防?'],
        'answer': ['高血压是...', '糖尿病预防包括...']
    })
    csv_file = Path("test_qa.csv")
    csv_data.to_csv(csv_file, index=False, encoding='utf-8')
    
    config = AnnotationConfig(
        model_base_url = "172.16.40.51:11434",
        model_backend="ollama",
        model_name="qwen3:32b"
    )
    
    framework = AnnotationFramework(config)
    
    # 注册自定义加载器
    framework.register_data_loader("csv", CSVDataLoader())
    
    # 使用自定义加载器
    results = await framework.annotate(
        data_source=csv_file,
        loader_type="csv",
        annotator_type="medical_qa"
    )
    
    for result in results:
        print(f"问题: {result.original_data.metadata['question']}")
        print(f"分类: {result.annotations}")
    
    # 清理
    csv_file.unlink()

# ==================== 主函数 ====================

async def main():
    """主函数，运行所有示例"""
    print("🚀 开始运行标注框架示例...")
    print("请确保ollama服务正在运行且qwen3:32b模型已下载\n")
    
    try:
        # 运行各个示例
        await example_medical_qa_annotation()
        await example_text_file_annotation()
        await example_term_pair_annotation()
        await example_high_performance_batch()
        await example_custom_loader()
        
        print("\n✅ 所有示例运行完成！")
        print("查看生成的结果文件获取详细信息。")
        
    except Exception as e:
        print(f"❌ 运行出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())