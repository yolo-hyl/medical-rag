"""
多轮对话RAG功能演示（会话历史存 Redis，需要先执行 deploy/start.sh）
"""
import asyncio
import logging
from MedicalRag.config.loader import ConfigLoader
from MedicalRag.core.KnowledgeBase import MedicalHybridKnowledgeBase
from MedicalRag.core.session_store import RedisSessionStore
from MedicalRag.rag.MultiDialogueRag import MultiDialogueRag

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)


def print_output(result):
    print(f"\n检索完成，检索用时：{result['search_time']} s，重写查询生成用时：{result['rewriten_generate_time']} s，增强生成用时：{result['out_generate_time']} s \n\n{result['answer']}")
    # 显示参考资料
    if "documents" in result:
        print(f"\n参考资料 ({len(result['documents'])} 条)，展示前3条:\n\n")
        for i, ctx in enumerate(result['documents'][:3], 1):
            print(f"{i}. 数据源： {ctx.metadata.get('source')} 数据源名：{ctx.metadata.get('source_name')} 向量距离：{ctx.metadata.get('distance')}\n")
            content = ctx.page_content[:200] + "..." if len(ctx.page_content) > 200 else ctx.page_content
            print(f"{content}\n\n")

async def main():
    # 加载配置
    config_manager = ConfigLoader()

    # # 支持自定义token估计方法
    # from MedicalRag.rag.utils import register_estimate_function
    # # 1) 注册自己的函数
    # @register_estimate_function("self_fun")
    # def estimate_tokens(text: str) -> int:
    #     """ 示例：简单的线性关系 你需要自己实现根据传入的自然语言来估计可能会被模型编码的token数量"""
    #     tokens = len(text) * 0.8  #
    #     return tokens
    # # 2) 修改配置文件（已有默认实现：avg、tiktoken）
    # config_manager.change({"multi_dialogue_rag.estimate_token_fun": "self_fun"})

    cfg = config_manager.config
    kb = MedicalHybridKnowledgeBase(cfg.milvus, cfg.embedding)
    store = RedisSessionStore(cfg.redis)
    rag = MultiDialogueRag(cfg.llm, cfg.multi_dialogue_rag, kb, store)
    session_id = "U123"

    try:
        while True:
            query = input()
            if query.strip().lower() in ("q", "exit", "quit", "退出"):
                break
            result = await rag.answer(query, session_id=session_id, return_document=True)
            print_output(result=result)
            print("---------------------------")
    finally:
        await kb.close()
        await store.close()


if __name__ == "__main__":
    asyncio.run(main())
