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
    
    query = "我有点肚子痛，该怎么办？"
    result = rag.answer(query, return_context=True)
    
            
    print(f"\n回答: {result['answer']}")
            
    # 显示参考资料
    if result['context']:
        print(f"\n参考资料 ({len(result['context'])} 条):")
        for i, ctx in enumerate(result['context'][:3], 1):
            print(f"{i}. {ctx['metadata'].get('source', 'unknown')}")
            content = ctx['content'][:200] + "..." if len(ctx['content']) > 200 else ctx['content']
            print(f"   {content}")

if __name__ == "__main__":
    main()
