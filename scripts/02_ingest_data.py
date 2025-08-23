import logging
from pathlib import Path
import json
from datasets import load_dataset

from MedicalRag.config.loader import load_config_from_file
from MedicalRag.core.advanced_components import (
    MedicalHybridKnowledgeBase,
    AdvancedIngestionPipeline
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_sample_data():
    """加载示例数据"""
    sample_data = [
        {
            "question": "高血压的主要症状有哪些？",
            "answer": "高血压的主要症状包括：1.头痛，多位于后脑；2.眩晕；3.心悸胸闷；4.肢体麻木。早期可能无明显症状。",
            "department": "心血管内科",
            "category": "症状咨询",
            "source": "医疗指南"
        },
        {
            "question": "糖尿病如何预防？",
            "answer": "糖尿病预防措施：1.合理膳食，控制总热量；2.适量运动，每周至少150分钟；3.控制体重；4.定期体检；5.戒烟限酒。",
            "department": "内分泌科", 
            "category": "预防保健",
            "source": "预防医学"
        },
        {
            "question": "冠心病的治疗方法？",
            "answer": "冠心病治疗包括：1.药物治疗：抗血小板、降脂药等；2.介入治疗：支架植入术；3.外科治疗：搭桥手术；4.生活方式改变。",
            "department": "心血管内科",
            "category": "治疗方案", 
            "source": "临床指南"
        },
        {
            "question": "感冒发烧怎么办？",
            "answer": "感冒发烧处理：1.多休息多喝水；2.物理降温；3.体温超过38.5°C可服用退烧药；4.症状严重时及时就医。",
            "department": "呼吸内科",
            "category": "治疗方案",
            "source": "常见疾病"
        },
        {
            "question": "胃痛的常见原因？",
            "answer": "胃痛常见原因：1.胃炎、胃溃疡；2.胃食管反流；3.功能性消化不良；4.幽门螺杆菌感染；5.药物刺激等。",
            "department": "消化内科",
            "category": "疾病诊断",
            "source": "诊断手册"
        }
    ]
    
    return sample_data

def main():
    """主函数"""
    print("=== 多向量字段混合检索演示 ===\n")
    
    try:
        # 1. 加载配置
        config = load_config_from_file("/home/weihua/medical-rag/src/MedicalRag/config/app_config.yaml")
        print(f"✅ 配置加载成功")
        print(f"   集合名称: {config.milvus.collection_name}")
        print(f"   多向量模式: {config.embedding.multi_vector.enabled}")
        
        # 2. 加载示例数据
        sample_data = load_sample_data()
        print(f"✅ 加载示例数据: {len(sample_data)} 条")
        
        # 3. 运行入库流水线
        print(f"\n=== 数据入库 ===")
        pipeline = AdvancedIngestionPipeline(config)
        success = pipeline.run(sample_data, drop_old=True)
        
        if not success:
            print("❌ 数据入库失败")
            return
        
        print(f"✅ 数据入库完成")
        
        # 4. 创建知识库实例
        kb = MedicalHybridKnowledgeBase(config)
        kb.initialize_collection(drop_old=False)  # 集合已存在
        
        collection_info = kb.get_collection_info()
        print(f"✅ 知识库初始化完成")
        print(f"   向量字段: {collection_info['vector_fields']}")
        
        # 5. 测试不同的检索策略
        print(f"\n=== 混合检索测试 ===")
        
        test_queries = [
            {
                "query": "高血压症状", 
                "type": "symptom",
                "description": "症状类查询"
            },
            {
                "query": "糖尿病预防措施",
                "type": "prevention", 
                "description": "预防类查询"
            },
            {
                "query": "冠心病怎么治疗",
                "type": "treatment",
                "description": "治疗类查询"
            }
        ]
        
        for test in test_queries:
            print(f"\n--- {test['description']}: '{test['query']}' ---")
            
            # 加权重排 - 根据查询类型调整权重
            if test['type'] == 'symptom':
                weights = [0.5, 0.3, 0.2]  # 更重视问题匹配
            elif test['type'] in ['treatment', 'prevention']:
                weights = [0.3, 0.4, 0.3]  # 更重视内容匹配
            else:
                weights = [0.4, 0.3, 0.3]  # 默认权重
            
            results = kb.hybrid_search(
                query=test['query'],
                k=3,
                ranker_type="weighted",
                ranker_params={"weights": weights}
            )
            
            print(f"加权重排结果 (权重: {weights}):")
            for i, doc in enumerate(results):
                question = doc.metadata.get('question', '')[:50] + '...'
                category = doc.metadata.get('category', '')
                print(f"  {i+1}. {question} [{category}]")
            
            # RRF重排对比
            results_rrf = kb.hybrid_search(
                query=test['query'],
                k=3,
                ranker_type="rrf",
                ranker_params={"k": 60}
            )
            
            print(f"RRF重排结果:")
            for i, doc in enumerate(results_rrf):
                question = doc.metadata.get('question', '')[:50] + '...'
                category = doc.metadata.get('category', '')
                print(f"  {i+1}. {question} [{category}]")
        
        # 6. 测试过滤检索
        print(f"\n=== 过滤检索测试 ===")
        
        filtered_results = kb.hybrid_search(
            query="症状",
            k=5,
            ranker_type="weighted",
            ranker_params={"weights": [0.4, 0.3, 0.3]},
            filters={"category": "症状咨询"}
        )
        
        print(f"过滤检索结果 (category='症状咨询'):")
        for i, doc in enumerate(filtered_results):
            question = doc.metadata.get('question', '')
            category = doc.metadata.get('category', '')
            print(f"  {i+1}. {question} [{category}]")
        
        # 7. 测试多部门过滤
        multi_dept_results = kb.hybrid_search(
            query="治疗",
            k=5,
            ranker_type="weighted", 
            ranker_params={"weights": [0.3, 0.4, 0.3]},
            filters={"department": ["心血管内科", "内分泌科"]}
        )
        
        print(f"\n多部门过滤结果 (心血管内科或内分泌科):")
        for i, doc in enumerate(multi_dept_results):
            question = doc.metadata.get('question', '')
            department = doc.metadata.get('department', '')
            print(f"  {i+1}. {question} [{department}]")
        
        print(f"\n=== 演示完成 ===")
        print(f"✅ 成功实现与main分支一致的多向量字段混合检索！")
        print(f"   - vec_question: 问题向量检索")
        print(f"   - vec_text: 问题+答案向量检索") 
        print(f"   - sparse: BM25稀疏向量检索")
        print(f"   - 支持动态权重调整和过滤检索")
        
    except Exception as e:
        logger.error(f"演示运行失败: {e}")
        raise

if __name__ == "__main__":
    main()