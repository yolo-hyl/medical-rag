from typing import Tuple, Dict
from langchain_core.documents import Document
from langchain_core.tools import StructuredTool
from pymilvus import Collection, AnnSearchRequest, RRFRanker, WeightedRanker
from typing import List, Literal, Optional, Dict
from pydantic import BaseModel, Field, conint, confloat
from ..config.models import *
from ..core.utils import create_embedding_client
from pymilvus import (
    MilvusClient, DataType, Function, FunctionType
)

class SearchTools:
    def __init__(self, milvus_config: MilvusConfig, embedding_config: EmbeddingConfig):
        self.milvus_config = milvus_config
        self.embedding_config = embedding_config
        
        self.summary_embedding = create_embedding_client(self.embedding_config.summary_dense)
        self.text_embedding = create_embedding_client(self.embedding_config.summary_dense)
        self.EMBEDDERS = {
            "summary_dense": self.summary_embedding,
            "text_dense": self.text_embedding
        }
        self.client = MilvusClient(
            uri=milvus_config.uri,
            token=milvus_config.token
        )

    def _search(
        self,
        query: str, # 查询的问题
        single_search_request: SingleSearchRequest,
        collection_name: str,
        output_fields: list[str]
    ):
        data = self.EMBEDDERS[single_search_request.anns_field].embed_query(query)
        result = self.client.search(
            collection_name=collection_name,
            data=[data],
            filter=single_search_request.expr,
            limit=single_search_request.limit,
            output_fields=output_fields,
            search_params={
                "metric_type": single_search_request.metric_type, 
                "params": {single_search_request.search_params}
            },
            anns_field=single_search_request.anns_field
        )
        return result
    
    def _build_ann_search_request(
        self,
        query,
        single_search_request: SingleSearchRequest
    ) -> AnnSearchRequest:
        data = self.EMBEDDERS[single_search_request.anns_field].embed_query(query)
        search_param = {
            "data": [data],
            "anns_field": single_search_request.anns_field,
            "param": {
                "metric_type": single_search_request.metric_type, 
                "params": {single_search_request.search_params}
            },
            "limit": single_search_request.limit,
            "expr": single_search_request.expr
        }
        return AnnSearchRequest(**search_param)
    
    def _hybrid_search(
        self,
        search: SearchRequest
    ):
        anns = []
        for i, item in enumerate(search.requests):
            anns.append(
                self._build_ann_search_request(
                    query=search.queries[i], 
                    single_search_request=item
                )
            )
        if search.fuse.method == "rrf":
            rank = RRFRanker(search.fuse.k)
        elif search.fuse.method == "weighted":
            rank = WeightedRanker(search.fuse.weights)
        result = self.client.hybrid_search(
            collection_name=search.collection_name,
            reqs=anns,
            ranker=rank,
            limit=search.limit,
            output_fields=search.output_fields
        )
        return result

    def make_hybrid_search_tool(self):
        """
        返回一个 LangChain StructuredTool。外层可把它 bind 给 Agent。
        """
        def _tool(req: SearchRequest) -> list:
            assert len(req.requests) != 0

            if len(req.requests) == 1:
                # 只有一个请求搜索，走普通的search
                output = self._search(
                    req.queries[0], 
                    req.requests[0], 
                    req.collection_name, 
                    req.output_fields
                )
            else:
                # 有多个请求搜索，走混合search
                output = self._hybrid_search(req)

            results = []
            for r in output:
                results.append(
                    Document(
                        page_content=output["text"], 
                        metadata=r
                    )
                )
            return results

        # 用 StructuredTool 暴露给 Agent（让LLM能“看见”Schema）
        return StructuredTool.from_function(
            name="hybrid_search",
            func=_tool,
            args_schema=SearchRequest,
            description="对 Milvus 进行多向量/稀疏/过滤的混合检索，并按RRF或加权融合返回文档"
        )
