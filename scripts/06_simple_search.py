"""
简单检索功能演示
"""
import logging
from MedicalRag.config.loader import load_config_from_file
from MedicalRag.core.components import KnowledgeBase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # 加载配置
    config = load_config_from_file("config/app_config.yaml")
    
    # 创建知识库
    kb = KnowledgeBase(config)
    
    # 交互式检索
    print("知识库检索系统已启动！输入 'quit' 退出")
    print("=" * 50)
    
    while True:
        try:
            query = input("\n请输入检索关键词: ").strip()
            
            if query.lower() in ['quit', 'exit', '退出']:
                break
            
            if not query:
                continue
            
            print("正在检索...")
            
            # 执行检索
            results = kb.search(query, k=5)
            
            if results:
                print(f"\n找到 {len(results)} 条相关结果:")
                for i, doc in enumerate(results, 1):
                    metadata = doc.metadata
                    print(f"\n{i}. [来源: {metadata.get('source', 'unknown')}]")
                    
                    # 显示问题和答案
                    if 'question' in metadata:
                        print(f"   问题: {metadata['question']}")
                    if 'answer' in metadata:
                        answer = metadata['answer'][:200] + "..." if len(metadata['answer']) > 200 else metadata['answer']
                        print(f"   答案: {answer}")
            else:
                print("未找到相关结果")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"检索出错: {e}")
            continue
    
    print("\n再见！")

if __name__ == "__main__":
    main()