import logging
from MedicalRag.config.loader import ConfigLoader
from MedicalRag.core.KnowledgeBase import MedicalHybridKnowledgeBase
from MedicalRag.config.models import SingleSearchRequest, SearchRequest, FusionSpec

def main():
    config_manager = ConfigLoader()
    kb = MedicalHybridKnowledgeBase(config_manager.config)
    # 单一向量检索
    query = "我脑袋有点晕，怎么办？"
    ssr = SingleSearchRequest(
        anns_field="summary_dense",  # 检索的字段
        metric_type="COSINE",  # 标尺
        search_params={"ef": 64},  # 参数
        limit=10,  # 查询数量
        expr=""  # 过滤参数
    )
    sr1 = SearchRequest(
        query=query,
        collection_name=config_manager.config.milvus.collection_name,
        requests=[ssr],
        output_fields=["summary", "document", "source", "source_name", "lt_doc_id", "chunk_id", "text"],
        limit=20
    )
    result = kb.search(req=sr1)
    print(result[0] if len(result) != 0 else "没有找到数据")
    
    # 多向量联合检索
    print("==============================================")
    query = "今早起来肚子有点痛，一阵一阵的，怎么办？"
    ssr1 = SingleSearchRequest(
        anns_field="summary_dense",  # 检索的字段
        metric_type="COSINE",  # 标尺
        search_params={"ef": 64},  # 参数
        limit=10,  # 查询数量
        expr=""  # 过滤参数
    )
    ssr2 = SingleSearchRequest(
        anns_field="text_sparse",  # 检索的字段
        metric_type="IP",  # 标尺
        search_params={ "drop_ratio_search": 0.0 },  # 参数
        limit=10,  # 查询数量
        expr=""  # 过滤参数
    )
    sr2 = SearchRequest(
        query=query,
        collection_name=config_manager.config.milvus.collection_name,
        requests=[ssr1, ssr2],
        output_fields=["summary", "document", "source", "source_name", "lt_doc_id", "chunk_id", "text"],
        limit=20
    )
    result = kb.search(req=sr2)
    print(result[0] if len(result) != 0 else "没有找到数据")
    
    print("==============================================")
    query = "吃完饭想吐怎么办？"
    ssr1 = SingleSearchRequest(
        anns_field="text_dense",  # 检索的字段
        metric_type="COSINE",  # 标尺
        search_params={"ef": 64},  # 参数
        limit=10,  # 查询数量
        expr=""  # 过滤参数
    )
    ssr2 = SingleSearchRequest(
        anns_field="text_sparse",  # 检索的字段
        metric_type="IP",  # 标尺
        search_params={ "drop_ratio_search": 0.0 },  # 参数
        limit=10,  # 查询数量
        expr=""  # 过滤参数
    )
    fuse = FusionSpec(
        method="weighted",
        weights=[0.6, 0.4]
    )
    sr3 = SearchRequest(
        query=query,
        collection_name=config_manager.config.milvus.collection_name,
        requests=[ssr1, ssr2],
        output_fields=["summary", "document", "source", "source_name", "lt_doc_id", "chunk_id", "text"],
        fuse=fuse,
        limit=20
    )
    result = kb.search(req=sr3)
    print(result[0] if len(result) != 0 else "没有找到数据")
    
if __name__ == "__main__":
    main()