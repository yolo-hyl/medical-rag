import logging
from typing import List, Dict, Any, Optional, Union
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_milvus import Milvus, BM25BuiltInFunction
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from tqdm import tqdm
from ..config.models import *
from .utils import create_embedding_client, create_llm_client
import hashlib
from pymilvus import MilvusClient, DataType, Collection, connections, FunctionType, Function, AnnSearchRequest, RRFRanker, WeightedRanker
from functools import lru_cache
from pathlib import Path
from ..config.models import AppConfig
from ..embed.sparse import Vocabulary, BM25Vectorizer
from .insert import insert_rows
from copy import deepcopy
from langchain_core.tools import StructuredTool
from ..embed.bm25 import BM25SparseEmbedding

get_resolve_path = lambda path, file=__file__: (Path(file).parent / Path(path)).resolve()

logger = logging.getLogger(__name__)

AllFields = [
    "pk", "text", 
    "summary", "document", 
    "source", "source_name", 
    "lt_doc_id", "chunk_id", 
    "summary_dense", "text_dense", "text_sparse"
]

class MedicalHybridKnowledgeBase:
    """医疗混合知识库 - 支持多向量字段检索"""
    
    def __init__(self, app_config: AppConfig):
        
        self.milvus_config = app_config.milvus
        self.embedding_config = app_config.embedding
        
        # 创建多个嵌入模型实例
        self.summary_embedding = self._create_summary_embedding()
        self.text_embedding = self._create_text_embedding()
        
        # 向量存储实例
        self.client = MilvusClient(uri=self.milvus_config.uri, token=self.milvus_config.token)            
        self.EMBEDDERS = {
            "summary_dense": self.summary_embedding,
            "text_dense": self.text_embedding
        }
        
        if self.embedding_config.text_sparse.provider == "self":
            # 如果自己管理词表，则还要创建一个BM25 Embedding
            self._vocab = Vocabulary.load(self.embedding_config.text_sparse.vocab_path_or_name)
            self._bm25 = BM25Vectorizer(
                vocab=self._vocab,
                domain_model=self.embedding_config.text_sparse.domain_model,
                k1=self.embedding_config.text_sparse.k1,
                b=self.embedding_config.text_sparse.b
            )
            self.EMBEDDERS["text_sparse"] = BM25SparseEmbedding(self._vocab, self._bm25)
        
    def _create_summary_embedding(self) -> Embeddings:
        """创建问题嵌入模型（用于summary_dense字段）"""
        return create_embedding_client(self.embedding_config.summary_dense)
    
    def _create_text_embedding(self) -> Embeddings:
        """创建文本嵌入模型（用于text_dense字段）"""
        return create_embedding_client(self.embedding_config.text_dense)
    
    def _create_collection(self):
        """ 使用原生 Milvus 客户端创建Collection"""
        assert self.embedding_config.summary_dense.dimension == self.embedding_config.summary_dense.dimension, "多向量单行存储时，两个嵌入模型嵌入向量维度必须相同"
        dim = self.embedding_config.summary_dense.dimension
        if self.milvus_config.drop_old:
            if self.client.has_collection(collection_name=self.milvus_config.collection_name):
                self.client.drop_collection(collection_name=self.milvus_config.collection_name)
            schema = MilvusClient.create_schema(
                auto_id=self.milvus_config.auto_id,
                enable_dynamic_field=True,
            )
            if self.milvus_config.auto_id:
                schema.add_field(field_name="pk",datatype=DataType.INT64, is_primary=True)
            else:
                schema.add_field(field_name="pk",datatype=DataType.VARCHAR, max_length=65535, is_primary=True)
            schema.add_field(
                field_name="text",
                datatype=DataType.VARCHAR,
                max_length=65535,
                enable_analyzer=True
            )
            schema.add_field(
                field_name="summary",
                datatype=DataType.VARCHAR,
                max_length=65535
            )
            schema.add_field(
                field_name="document",
                datatype=DataType.VARCHAR,
                max_length=65535
            )
            schema.add_field(
                field_name="source",
                datatype=DataType.VARCHAR,
                max_length=65535
            )
            schema.add_field(
                field_name="source_name",
                datatype=DataType.VARCHAR,
                max_length=65535
            )
            schema.add_field(
                field_name="lt_doc_id",
                datatype=DataType.VARCHAR,
                max_length=65535
            )
            schema.add_field(
                field_name="chunk_id",
                datatype=DataType.INT64,
                max_length=65535
            )
            schema.add_field(
                field_name="summary_dense",
                datatype=DataType.FLOAT_VECTOR,
                dim=dim
            )
            schema.add_field(
                field_name="text_dense",
                datatype=DataType.FLOAT_VECTOR,
                dim=dim
            )
            schema.add_field(
                field_name="text_sparse",
                datatype=DataType.SPARSE_FLOAT_VECTOR
            )
            if self.embedding_config.text_sparse.provider == "Milvus":
                bm25_fn = Function(
                    name="bm25_text_to_sparse",
                    function_type=FunctionType.BM25,
                    input_field_names=["text"],
                    output_field_names=["text_sparse"],
                )
                schema.add_function(bm25_fn)
                
            self.client.create_collection(collection_name=self.milvus_config.collection_name, schema=schema)
            
            return self.client
        else:
            return self.client  # 如果不删除老集合，那就直接返回，不要创建
    
    def build_index(self):
        """ 构建合适的索引，构建完成之后load """
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="summary_dense", 
            index_type="HNSW",
            index_name="summary_dense_index",
            metric_type="COSINE",
            params={ "M": 32, "efConstruction": 200 }
        )
        index_params.add_index(
            field_name="text_dense",
            index_type="HNSW",
            index_name="text_dense_index",
            metric_type="COSINE",
            params={ "M": 32, "efConstruction": 200 }
        )
        if self.embedding_config.text_sparse.provider == "self":
            index_params.add_index(
                field_name="text_sparse",
                index_type="SPARSE_INVERTED_INDEX",
                index_name="text_sparse_index",
                metric_type="IP",
                params={ "inverted_index_algo": "DAAT_MAXSCORE" }
            )
        else:
            index_params.add_index(
                field_name="text_sparse",
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="BM25",
                params={
                    "inverted_index_algo": "DAAT_MAXSCORE",
                    "bm25_k1": self.embedding_config.text_sparse.k1,
                    "bm25_b": self.embedding_config.text_sparse.b
                }
            )
        self.client.create_index(
            collection_name=self.milvus_config.collection_name,
            index_params=index_params
        )
        self.client.load_collection(self.milvus_config.collection_name)
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """添加文档，自动处理多向量字段"""
        tokenizer_docs = []
        
        rows = []
        for doc in documents:
            # 提取问题和答案
            summary = doc.metadata.get("summary", "")
            text = doc.page_content
            
            # 上线这里要删掉
            if len(doc.metadata.get("summary_dense", [])) == 0:
                doc.metadata["summary_dense"] = self.EMBEDDERS["summary_dense"].embed_documents([summary])[0]
                
            if len(doc.metadata.get("text_dense", [])) == 0:
                doc.metadata["text_dense"] = self.EMBEDDERS["text_dense"].embed_documents([text])[0]
            
            tokenizer_docs.append(text)  # 需要进行稀疏向量编码的text字段
            doc_dict = deepcopy(doc.metadata)
            filtered = {k: v for k, v in doc_dict.items() if k in AllFields}
            if not self.milvus_config.auto_id:
                # 如果不采用自动id，则默认id实现为quesiton的hash值，以便插入时覆盖重复数据
                filtered["pk"] = doc.metadata.get("hash_id", "")
            filtered["text"] = text
            if self.embedding_config.text_sparse.provider == "self":
                # 如果自管理词表，则还需要进行稀疏向量的构建
                filtered["text_sparse"] = self.EMBEDDERS["text_sparse"].embed_documents([summary])[0]
            rows.append(filtered)
                
        insert_rows(
            client=self.client,
            collection_name=self.milvus_config.collection_name,
            rows=rows,
            show_progress=False  # 小批量不显示进度条
        )
        return len(rows)
    
    
    def _encode_query(self, query, anns_field):
        if anns_field != "text_sparse":
            data = self.EMBEDDERS[anns_field].embed_query(query)
        else:
            if self.embedding_config.text_sparse.provider == "self":
                data = self.EMBEDDERS[anns_field].embed_query(query)  # 自己管理的词表
            else:
                data = data  # 自动托管的BM25算法时，传入的查询不需要做任何处理
        return data
    
    def _search(
        self,
        query: str, # 查询的问题
        single_search_request: SingleSearchRequest,
        collection_name: str,
        output_fields: list[str]
    ):
        """ Milvus 原生的查询单个问题 https://milvus.io/docs/zh/filtered-search.md """
        data = self._encode_query(query=query, anns_field=single_search_request.anns_field)
                
        result = self.client.search(
            collection_name=collection_name,
            data=[data],
            filter=single_search_request.expr,
            limit=single_search_request.limit,
            output_fields=output_fields,
            search_params={
                "metric_type": single_search_request.metric_type, 
                "params": single_search_request.search_params
            },
            anns_field=single_search_request.anns_field
        )
        return result
    
    def _build_ann_search_request(
        self,
        query,
        single_search_request: SingleSearchRequest
    ) -> AnnSearchRequest:
        """ 构建子 AnnSearchRequest 请求"""
        data = self._encode_query(query=query, anns_field=single_search_request.anns_field)
        search_param = {
            "data": [data],
            "anns_field": single_search_request.anns_field,
            "param": {
                "metric_type": single_search_request.metric_type, 
                "params": single_search_request.search_params
            },
            "limit": single_search_request.limit,
            "expr": single_search_request.expr
        }
        return AnnSearchRequest(**search_param)
    
    def _hybrid_search(
        self,
        search: SearchRequest
    ):
        """ Milvus 原生混合查询 https://milvus.io/docs/zh/multi-vector-search.md"""
        anns = []
        for item in search.requests:  # 构建子查询
            anns.append(
                self._build_ann_search_request(
                    query=search.query, 
                    single_search_request=item
                )
            )
        if search.fuse.method == "rrf":
            rank = RRFRanker(search.fuse.k)
        elif search.fuse.method == "weighted":
            rank = WeightedRanker(*search.fuse.weights)
        result = self.client.hybrid_search(
            collection_name=search.collection_name,
            reqs=anns,
            ranker=rank,
            limit=search.limit,
            output_fields=search.output_fields
        )
        return result
    
    def search(self, req: SearchRequest) -> List[Document]:
        if len(req.requests) == 1:
            # 只有一个请求搜索，走普通的search
            outputs = self._search(
                req.query, 
                req.requests[0], 
                req.collection_name, 
                req.output_fields
            )[0]  # 批量中的第一条，这里先不支持批量查询
        else:
            # 有多个请求搜索，走混合search
            outputs = self._hybrid_search(req)[0]

        results = []  
        
        for i in range(len(outputs)): # 封装获得 List[Document]
            results.append(
                Document(
                    page_content=outputs[i]["entity"]["text"], 
                    metadata={
                        "pk": outputs[i]["pk"],
                        "distance": outputs[i]["distance"],
                        "chunk_id": outputs[i]["entity"]["chunk_id"],
                        "summary": outputs[i]["entity"]["summary"],
                        "document": outputs[i]["entity"]["document"],
                        "source": outputs[i]["entity"]["source"],
                        "source_name": outputs[i]["entity"]["source_name"],
                        "lt_doc_id": outputs[i]["entity"]["lt_doc_id"],
                    }
                )
            )
            
        return results
    