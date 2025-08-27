"""
基础RAG功能演示
"""
import logging
from MedicalRag.config.loader import ConfigLoader
from MedicalRag.rag.MultiDialogueRag import MultiDialogueRag
from MedicalRag.rag.utils import register_estimate_function

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)


@register_estimate_function("self_fun")
def estimate_tokens(text: str) -> int:
    tokens = len(text) * 0.8
    return tokens

def print_output(result):
    print(f"\n检索完成，检索用时：{result['search_time']} s，重写查询生成用时：{result['rewriten_generate_time']} s，增强生成用时：{result['out_generate_time']} s \n\n{result['answer']}")
    # 显示参考资料
    if "documents" in result:
        print(f"\n参考资料 ({len(result['documents'])} 条)，展示前3条:\n\n")
        for i, ctx in enumerate(result['documents'][:3], 1):
            print(f"{i}. 数据源： {ctx.metadata.get('source')} 数据源名：{ctx.metadata.get('source_name')} 向量距离：{ctx.metadata.get('distance')}\n")
            content = ctx.page_content[:200] + "..." if len(ctx.page_content) > 200 else ctx.page_content
            print(f"{content}\n\n")

def main():
    # 加载配置
    config_manager = ConfigLoader()
    # 创建基础RAG系统
    config_manager.change({"multi_dialogue_rag.estimate_token_fun": "self_fun"})
    rag = MultiDialogueRag(config_manager.config)
    query = "我有点肚子痛，该怎么办？" # 在传统中医中，蜣螂及其粪球"转丸"被用于治疗哪些疾病，具体有哪些药用价值？
    session_id = "U123"
    while True:
        query = input()
        result = rag.answer(query, session_id=session_id, return_document=True)
        print_output(result=result)
        print("---------------------------")
    

if __name__ == "__main__":
    main()
