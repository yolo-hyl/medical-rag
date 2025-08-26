"""
智能体RAG功能演示
"""
import logging
from MedicalRag.config.loader import load_config_from_file
from MedicalRag.rag.SimpleRag import create_agent_rag

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # 加载配置
    config = load_config_from_file("config/app_config.yaml")
    
    # 创建智能体RAG系统
    agent = create_agent_rag(config, enable_web_search=True)
    
    # 交互式问答
    print("智能体RAG系统已启动！输入 'quit' 退出")
    print("该系统可以:")
    print("- 搜索医学知识库")
    print("- 进行网络搜索补充")
    print("- 综合多源信息回答")
    print("=" * 50)
    
    while True:
        try:
            query = input("\n请输入您的问题: ").strip()
            
            if query.lower() in ['quit', 'exit', '退出']:
                break
            
            if not query:
                continue
            
            print("智能体正在分析和搜索...")
            
            # 获取详细回答
            result = agent.answer(query, return_details=True)
            
            print(f"\n回答: {result['answer']}")
            
            # 显示搜索历程
            if result['search_history']:
                print(f"\n搜索历程:")
                for i, step in enumerate(result['search_history'], 1):
                    print(f"{i}. {step['step']}: {step['result']}")
                print(f"总迭代次数: {result['iterations']}")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"处理出错: {e}")
            continue
    
    print("\n再见！")

if __name__ == "__main__":
    main()
