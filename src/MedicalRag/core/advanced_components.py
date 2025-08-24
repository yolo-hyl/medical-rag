"""
高级组件：使用langchain-milvus的多向量字段混合检索
实现与main分支一致的混合检索策略：
- vec_question: question的稠密向量检索
- vec_text: question+answer的稠密向量检索  
- sparse: BM25稀疏向量检索（基于question+answer）
"""
import logging
from typing import List, Dict, Any, Optional, Union
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_milvus import Milvus, BM25BuiltInFunction
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from tqdm import tqdm
from ..config.models import AppConfig, EmbeddingConfig, DenseConfig
from .components import create_embedding_client
import hashlib
from pymilvus import MilvusClient, DataType, Collection, connections, FunctionType, Function
from functools import lru_cache
from pathlib import Path

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
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.milvus_config = config.milvus
        self.embedding_config = config.embedding
        
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
        return create_embedding_client(self.embedding_config.text_sparse)
    
    def _create_collection(self, drop_old: bool = False):
        client = MilvusClient(uri=self.milvus_config.uri, token=self.milvus_config.token)
        assert self.config.embedding.summary_dense.dimension == self.config.embedding.summary_dense.dimension, "多向量单行存储时，两个嵌入模型嵌入向量维度必须相同"
        dim = self.config.embedding.summary_dense.dimension
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
                    "tokenizer": "defalut",
                    "filter": [{"type": "stop", "stop_words": self.stopwords}],
                },
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
                auto_id=self.milvus_config.auto_id
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
                
                q_vec = self.summary_embedding.embed_documents([summary])[0]  # -> vec_question
                d_vec = self.text_embedding.embed_documents([text])[0]
                
                self.vectorstore.add_embeddings(
                    texts=[text],                            # 写入 text 字段，供 BM25 读取
                    embeddings=[[q_vec, d_vec]],       # None 可省略；BM25 会自己写 sparse
                    metadatas=[doc.metadata],
                    ids=None if self.config.milvus.auto_id else [doc.metadata.get("hash_id", "")]
                )
                num += 1
            except Exception as e:
                logger.info(f"添加{summary}数据失败")
        logger.info(f"成功添加 {len(documents)} 个文档到多向量字段集合")
        return num
    
    def hybrid_search(
        self,
        query: str,
        k: int = 10,
        ranker_type: str = "weighted",
        ranker_params: Optional[Dict[str, Any]] = None,
        filters: Optional[str] = None
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
        if ranker_params is None:
            if ranker_type == "weighted":
                # 权重：问题向量0.4，文本向量0.3，BM25稀疏向量0.3
                ranker_params = {"weights": [0.4, 0.3, 0.3]}
            else:  # rrf
                ranker_params = {"k": 100}
        
        try:
            # 执行混合检索
            # query_emb = self.summary_embedding.embed_documents([query])[0]
            results = self.vectorstore.similarity_search(
                query,
                k=k,
                ranker_type=ranker_type,
                ranker_params=ranker_params,
                expr=filters
            )
            
            logger.info(f"混合检索返回 {len(results)} 个结果")
            return results
            
        except Exception as e:
            logger.error(f"混合检索失败: {e}")
            raise
    
    def as_retriever(self, **kwargs):
        """转换为检索器"""
        if not self.vectorstore:
            raise ValueError("请先初始化集合")
        
        # 设置默认参数
        search_kwargs = {
            "k": 10,
            "ranker_type": "weighted",
            "ranker_params": {"weights": [0.4, 0.3, 0.3]},
            **kwargs
        }
        
        return self.vectorstore.as_retriever(search_kwargs=search_kwargs)
    
    def get_collection_info(self) -> Dict[str, Any]:
        """获取集合信息"""
        if not self.vectorstore:
            return {"status": "not_initialized"}
        
        return {
            "collection_name": self.vectorstore.collection_name,
            "vector_fields": self.vectorstore.vector_fields,
            "status": "initialized"
        }
        
# 高级数据处理器
class AdvancedDataProcessor:
    """高级数据处理器 - 支持多向量字段数据准备"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.data_config = config.data
    
    def prepare_multi_vector_documents(self, raw_documents: List[Dict[str, Any]]) -> List[Document]:
        """
        准备多向量字段文档
        与main分支保持一致：分别处理question和question+answer
        """
        documents = []
        
        for record in tqdm(raw_documents, desc="预处理文档"):
            summary = record.get(self.data_config.summary_field, "")
            document = record.get(self.data_config.document_field, "")
            
            # 构建完整文本（用于BM25和vec_text）
            if record.get(self.data_config.default_source, "qa") == "qa":
                text = f"问题: {summary}\n\n答案: {document}"
            else:
                text = document
            
            # 构建元数据
            metadata = {
                "summary": summary,
                "document": document,
                "source": record.get(self.data_config.source_field, self.data_config.default_source),
                "source_name": record.get(self.data_config.source_name_field, self.data_config.default_source_name),
                "hash_id": hashlib.md5(summary.encode('UTF-8')).hexdigest(),
                "lt_doc_id": record.get(self.data_config.lt_doc_id_field, self.data_config.default_lt_doc_id),
                "chunk_id": record.get(self.data_config.chunk_id_field, self.data_config.default_chunk_id)
            }
            
            # 保留其他字段
            for key, value in record.items():
                if key not in [
                    self.data_config.summary_field,
                    self.data_config.document_field,
                    self.data_config.source_field,
                    self.data_config.source_name_field,
                    self.data_config.lt_doc_id_field,
                    self.data_config.chunk_id_field
                ]:
                    metadata[key] = value
            
            # 创建文档（page_content用于BM25稀疏向量）
            document = Document(
                page_content=text,  # BM25基于完整文本
                metadata=metadata
            )
            
            documents.append(document)
        
        return documents
    
# 高级入库流水线
class AdvancedIngestionPipeline:
    """高级入库流水线 - 使用多向量字段"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.kb = MedicalHybridKnowledgeBase(config)
        self.processor = AdvancedDataProcessor(config)
    
    def run(self, raw_data: List[Dict[str, Any]], drop_old: bool = False) -> bool:
        """运行高级入库流水线"""
        try:
            # 1. 初始化多向量字段集合
            logger.info("初始化多向量字段集合...")
            self.kb.initialize_collection(drop_old=drop_old)
            
            # 2. 处理数据
            logger.info("预处理多向量字段文档...")
            documents = self.processor.prepare_multi_vector_documents(raw_data)
            
            # 3. 批量插入
            logger.info(f"开始插入 {len(documents)} 个文档...")
            batch_size = self.config.ingestion.batch_size
            total_inserted = 0
            
            for i in tqdm(range(0, len(documents), batch_size)):
                batch = documents[i:i + batch_size]
                try:
                    ids = self.kb.add_documents(batch)
                    total_inserted += ids
                except Exception as e:
                    logger.error(f"插入批次失败: {e}")
                    continue
            
            logger.info(f"高级入库完成！总共插入 {total_inserted} 个文档")
            logger.info(f"集合信息: {self.kb.get_collection_info()}")
            
            return True
            
        except Exception as e:
            logger.error(f"高级入库流水线失败: {e}")
            return False