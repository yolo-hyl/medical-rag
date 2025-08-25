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
from pymilvus import MilvusClient, DataType, Collection, connections, FunctionType, Function
from functools import lru_cache
from pathlib import Path
from ..config.models import SearchConfig
get_resolve_path = lambda path, file=__file__: (Path(file).parent / Path(path)).resolve()

logger = logging.getLogger(__name__)


@lru_cache()
def get_stopwords(source="all"):
    """
    Params:
        langs: string，支持的语言，目前仅支持中文(zh)
        source: string, 停用词来源，目前支持
          - baidu: 百度停用词表
          - hit: 哈工大停用词表
          - ict: 中科院计算所停用词表
          - scu: 四川大学机器智能实验室停用词库
          - cn: 广为流传未知来源的中文停用词表
          - marimo: Marimo multi-lingual stopwords collection 内的中文停用词
          - iso: Stopwords ISO 内的中文停用词
          - all: 上述所有停用词并集
          - en: 英文
    Return:
        a set, 停用词表集合
    """

    supported_source = ["cn", "baidu", "hit", "scu", "marimo", "ict", "iso", "all", 'en']
    if source not in supported_source:
        raise NotImplementedError("请求了未知来源，请使用`help(stopwords)`查看支持的来源")
    return set(get_resolve_path(f"./stopwords/stopwords.zh.{source}.txt").read_text(encoding='utf8').strip().split())


class MedicalHybridKnowledgeBase:
    """医疗混合知识库 - 支持多向量字段检索"""
    
    def __init__(self, milvus_config: MilvusConfig, embedding_config: EmbeddingConfig, llm_config: LLMConfig):
        self.milvus_config = milvus_config
        self.embedding_config = embedding_config
        self.llm = create_llm_client(llm_config)
        # 创建多个嵌入模型实例
        self.summary_embedding = self._create_summary_embedding()
        self.text_embedding = self._create_text_embedding()
        
        # 向量存储实例
        self.vectorstore: Optional[Milvus] = None
        
        # 停用词
        self.stopwords = get_stopwords(source="all")
        
    def _create_summary_embedding(self) -> Embeddings:
        """创建问题嵌入模型（用于vec_字段）"""
        return create_embedding_client(self.embedding_config.summary_dense)
    
    def _create_text_embedding(self) -> Embeddings:
        """创建文本嵌入模型（用于vec_text字段）"""
        return create_embedding_client(self.embedding_config.text_dense)
    
    def _create_collection(self, drop_old: bool = False):
        client = MilvusClient(uri=self.milvus_config.uri, token=self.milvus_config.token)
        assert self.embedding_config.summary_dense.dimension == self.embedding_config.summary_dense.dimension, "多向量单行存储时，两个嵌入模型嵌入向量维度必须相同"
        dim = self.embedding_config.summary_dense.dimension
        if drop_old:
            if client.has_collection(collection_name=self.milvus_config.collection_name):
                client.drop_collection(collection_name=self.milvus_config.collection_name)
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
                field_name="vec_summary",
                datatype=DataType.FLOAT_VECTOR,
                dim=dim
            )
            schema.add_field(
                field_name="vec_text",
                datatype=DataType.FLOAT_VECTOR,
                dim=dim
            )
            schema.add_field(
                field_name="sparse",
                datatype=DataType.SPARSE_FLOAT_VECTOR
            )
            bm25_fn = Function(
                name="bm25_text_to_sparse",
                function_type=FunctionType.BM25,
                input_field_names=["text"],
                output_field_names=["sparse"],
            )
            schema.add_function(bm25_fn)
            client.create_collection(collection_name=self.milvus_config.collection_name, schema=schema)
        else:
            return  # 如果不删除老集合，那就直接返回，不要创建
        
    def initialize_collection(self, drop_old: bool = False) -> None:
        """初始化集合，使用多向量字段"""
        self._create_collection(drop_old)
        try:
            # 定义BM25内置函数
            bm25 = BM25BuiltInFunction(
                input_field_names="text",
                output_field_names="sparse",
                analyzer_params={
                    "tokenizer": "jieba",
                    "filter": [{"type": "stop", "stop_words": self.stopwords}],
                }
            )
            
            # 创建向量存储，支持多个向量字段
            self.vectorstore = Milvus(
                embedding_function=[
                    self.summary_embedding, 
                    self.text_embedding
                ],      # 只有 dense 在这里；BM25 走 builtin_function
                builtin_function=bm25,                  # <== 通过 builtin_function 开 BM25
                vector_field=["vec_summary", "vec_text", "sparse"],  # 顺序要对齐：[q_emb, d_emb, bm25]
                text_field="text",                      # BM25 的输入字段
                collection_name=self.milvus_config.collection_name,
                connection_args={
                    "uri": self.milvus_config.uri,
                    "token": self.milvus_config.token
                },
                consistency_level="Strong",
                drop_old=False,
                auto_id=self.milvus_config.auto_id,
                 # 添加索引参数（参考test.py）
                index_params=[
                    {"index_type": "HNSW", "metric_type": "IP", "params": {"M": 16, "efConstruction": 200}},  # vec_summary
                    {"index_type": "HNSW", "metric_type": "IP", "params": {"M": 16, "efConstruction": 200}},  # vec_text
                    {"index_type": "AUTOINDEX", "metric_type": "BM25", "params": {}},                         # sparse
                ],
                # 添加搜索参数
                search_params=[
                    {"metric_type": "IP", "params": {"ef": 64}},  # vec_question
                    {"metric_type": "IP", "params": {"ef": 64}},  # vec_text
                    {"metric_type": "BM25", "params": {}},        # sparse
                ],
            )
            
            logger.info(f"初始化多向量字段集合: {self.vectorstore.vector_fields}")
            
        except Exception as e:
            logger.error(f"初始化集合失败: {e}")
            raise
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """添加文档，自动处理多向量字段"""
        if not self.vectorstore:
            raise ValueError("请先初始化集合")
        num = 0
        for doc in documents:
            # 提取问题和答案
            try:
                summary = doc.metadata.get("summary", "")
                text = doc.page_content
                if len(doc.metadata.get("summary_embed", [])) != 0:
                    q_vec = doc.metadata.get("summary_embed")
                else:
                    q_vec = self.summary_embedding.embed_documents([summary])[0]
                    
                if len(doc.metadata.get("text_embed", [])) != 0:
                    d_vec = doc.metadata.get("text_embed")
                else:
                    d_vec = self.summary_embedding.embed_documents([summary])[0]
                    
                # q_vec = self.summary_embedding.embed_documents([summary])[0]  # -> vec_question
                # d_vec = self.text_embedding.embed_documents([text])[0]
                
                self.vectorstore.add_embeddings(
                    texts=[text],                            # 写入 text 字段，供 BM25 读取
                    embeddings=[[q_vec, d_vec]],       # None 可省略；BM25 会自己写 sparse
                    metadatas=[doc.metadata],
                    ids=None if self.milvus_config.auto_id else [doc.metadata.get("hash_id", "")]
                )
                num += 1
            except Exception as e:
                logger.info(f"添加{summary}数据失败")
        return num
    
    def hybrid_search(
        self,
        query: str,
        config: SearchConfig
    ) -> List[Document]:
        """
        混合检索：同时使用多个向量字段
        
        Args:
            query: 查询问题
            k: 返回结果数量
            ranker_type: 重排类型 ("weighted" 或 "rrf")
            ranker_params: 重排参数
            filters: 过滤条件
        """
        if not self.vectorstore:
            raise ValueError("请先初始化集合")
        
        # 默认重排参数
        if config.ranker_type == "weighted":
            ranker_params = {"weight": config.weights}
        else:  # rrf
            ranker_params = {"k": config.rrf_k}
        
        try:
            # 执行混合检索
            results = self.vectorstore.similarity_search(
                query,
                k=config.top_k,
                ranker_type=config.ranker_type,
                ranker_params=ranker_params,
                expr=config.filters
            )
            
            logger.info(f"检索到 {len(results)} 个结果")
            return results
            
        except Exception as e:
            logger.error(f"混合检索失败: {e}")
            raise
    
    def get_collection_info(self) -> Dict[str, Any]:
        """获取集合信息"""
        if not self.vectorstore:
            return {"status": "not_initialized"}
        
        return {
            "collection_name": self.vectorstore.collection_name,
            "vector_fields": self.vectorstore.vector_fields,
            "status": "initialized"
        }
    
    def as_retriever(self) -> Any:
        """Return a retriever configured according to the search config."""
        if self.vectorstore is None:
            raise RuntimeError("Collection has not been initialised.  Call initialize_collection() first.")
        return self.vectorstore.as_retriever(
            search_kwargs={"k": 10}
        )