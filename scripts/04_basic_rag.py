"""
基础RAG功能演示
"""
import logging
from MedicalRag.config.loader import ConfigLoader
from MedicalRag.rag.SimpleRag import SimpleRAG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # 加载配置
    config_manager = ConfigLoader()
    # 创建基础RAG系统
    rag = SimpleRAG(config_manager.config)
    query = "我有点肚子痛，该怎么办？" # 在传统中医中，蜣螂及其粪球"转丸"被用于治疗哪些疾病，具体有哪些药用价值？
    result = rag.answer(query, return_document=True)
    print(f"\n{result['answer']}")
    # 显示参考资料
    if "documents" in result:
        print(f"\n参考资料 ({len(result['documents'])} 条)，展示前3条:\n\n")
        for i, ctx in enumerate(result['documents'][:3], 1):
            print(f"{i}. 数据源： {ctx.metadata.get('source')} 数据源名：{ctx.metadata.get('source_name')} 向量距离：{ctx.metadata.get('distance')}\n")
            content = ctx.page_content[:200] + "..." if len(ctx.page_content) > 200 else ctx.page_content
            print(f"{content}\n\n")

if __name__ == "__main__":
    main()
