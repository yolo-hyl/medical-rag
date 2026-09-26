"""
基础RAG功能演示
"""
import asyncio
import logging
from MedicalRag.config.loader import ConfigLoader
from MedicalRag.core.KnowledgeBase import MedicalHybridKnowledgeBase
from MedicalRag.rag.SimpleRag import SimpleRAG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    # 加载配置
    cfg = ConfigLoader().config
    # 知识库由调用方创建后注入，多个组件可以共用同一个实例
    kb = MedicalHybridKnowledgeBase(cfg.milvus, cfg.embedding)
    rag = SimpleRAG(cfg.llm, kb)
    query = "我有点肚子痛，该怎么办？" # 在传统中医中，蜣螂及其粪球"转丸"被用于治疗哪些疾病，具体有哪些药用价值？
    try:
        result = await rag.answer(query, return_document=True)
    finally:
        await kb.close()
    print(f"\n检索完成，检索用时：{result['search_time']} s，生成用时：{result['generation_time']} s \n\n{result['answer']}")
    # 显示参考资料
    if "documents" in result:
        print(f"\n参考资料 ({len(result['documents'])} 条)，展示前3条:\n\n")
        for i, ctx in enumerate(result['documents'][:3], 1):
            print(f"{i}. 数据源： {ctx.metadata.get('source')} 数据源名：{ctx.metadata.get('source_name')} 向量距离：{ctx.metadata.get('distance')}\n")
            content = ctx.page_content[:200] + "..." if len(ctx.page_content) > 200 else ctx.page_content
            print(f"{content}\n\n")

if __name__ == "__main__":
    asyncio.run(main())
