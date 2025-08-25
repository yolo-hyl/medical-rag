"""
基础RAG功能演示
"""
import logging
from MedicalRag.config.loader import ConfigLoader
from MedicalRag.rag.basic_rag import BasicRAG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # 加载配置
    config_manager = ConfigLoader()
    
    # 创建基础RAG系统
    rag = BasicRAG(config_manager.config)
    
    # 交互式问答
    print("基础RAG系统已启动！输入 'quit' 退出")
    print("=" * 50)
    
    while True:
        try:
            query = input("\n请输入您的问题: ").strip()
            
            if query.lower() in ['quit', 'exit', '退出']:
                break
            
            if not query:
                continue
            
            print("正在思考...")
            
            # 获取详细回答
            result = rag.answer(query, return_context=True)
            
            print(f"\n回答: {result['answer']}")
            
            # 显示参考资料
            if result['context']:
                print(f"\n参考资料 ({len(result['context'])} 条):")
                for i, ctx in enumerate(result['context'][:3], 1):
                    print(f"{i}. {ctx['metadata'].get('source', 'unknown')}")
                    content = ctx['content'][:200] + "..." if len(ctx['content']) > 200 else ctx['content']
                    print(f"   {content}")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"处理出错: {e}")
            continue
    
    print("\n再见！")

if __name__ == "__main__":
    main()
