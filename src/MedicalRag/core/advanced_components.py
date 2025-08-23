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

from ..config.models import AppConfig, EmbeddingConfig, DenseConfig
from .components import create_embedding_client

logger = logging.getLogger(__name__)

class MedicalHybridKnowledgeBase:
    """医疗混合知识库 - 支持多向量字段检索"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.milvus_config = config.milvus
        self.embedding_config = config.embedding
        
        # 创建多个嵌入模型实例
        self.question_embedding = self._create_question_embedding()
        self.text_embedding = self._create_text_embedding()
        
        # 向量存储实例
        self.vectorstore: Optional[Milvus] = None
        
    def _create_question_embedding(self) -> Embeddings:
        """创建问题嵌入模型（用于vec_question字段）"""
        # 复用配置，但可以针对question优化
        return create_embedding_client(self.embedding_config.dense)
    
    def _create_text_embedding(self) -> Embeddings:
        """创建文本嵌入模型（用于vec_text字段）"""
        # 可以使用不同的模型或参数
        config = self.embedding_config.dense
        
        # 示例：可以使用不同的模型专门处理长文本
        if config.provider == "openai":
            return OpenAIEmbeddings(
                model="text-embedding-3-large",  # 使用更大的模型处理长文本
                api_key=config.api_key,
                base_url=config.base_url
            )
        elif config.provider == "ollama":
            return OllamaEmbeddings(
                model=config.model,
                base_url=config.base_url
            )
        
        return create_embedding_client(config)
    
    def initialize_collection(self, drop_old: bool = False) -> None:
        """初始化集合，使用多向量字段"""
        try:
            # 定义BM25内置函数
            bm25_function = BM25BuiltInFunction(
                output_field_names="sparse"  # 稀疏向量字段名
            )
            
            # 创建向量存储，支持多个向量字段
            self.vectorstore = Milvus(
                embedding_function=[
                    self.question_embedding,  # vec_question字段
                    self.text_embedding       # vec_text字段
                ],
                builtin_function=bm25_function,
                vector_field=["vec_question", "vec_text", "sparse"],
                collection_name=self.milvus_config.collection_name,
                connection_args={
                    "uri": self.milvus_config.uri,
                    "token": self.milvus_config.token
                },
                consistency_level="Strong",
                drop_old=drop_old,
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
        
        # 预处理文档，为不同字段准备内容
        processed_docs = []
        
        for doc in documents:
            # 提取问题和答案
            question = doc.metadata.get("question", "")
            answer = doc.metadata.get("answer", "")
            full_text = f"问题: {question}\n\n答案: {answer}"
            
            # 创建新文档，page_content用于BM25
            processed_doc = Document(
                page_content=full_text,  # BM25使用完整文本
                metadata={
                    **doc.metadata,
                    "question_text": question,      # vec_question字段使用
                    "full_text": full_text          # vec_text字段使用
                }
            )
            processed_docs.append(processed_doc)
        
        try:
            # langchain-milvus会自动处理多向量字段的嵌入
            ids = self.vectorstore.add_documents(processed_docs)
            logger.info(f"成功添加 {len(ids)} 个文档到多向量字段集合")
            return ids
            
        except Exception as e:
            logger.error(f"添加文档失败: {e}")
            raise
    
    def hybrid_search(
        self,
        query: str,
        k: int = 10,
        ranker_type: str = "weighted",
        ranker_params: Optional[Dict[str, Any]] = None,
        filters: Optional[Dict[str, Any]] = None
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
            results = self.vectorstore.similarity_search(
                query,
                k=k,
                ranker_type=ranker_type,
                ranker_params=ranker_params,
                expr=self._build_filter_expr(filters) if filters else None
            )
            
            logger.info(f"混合检索返回 {len(results)} 个结果")
            return results
            
        except Exception as e:
            logger.error(f"混合检索失败: {e}")
            raise
    
    def _build_filter_expr(self, filters: Dict[str, Any]) -> str:
        """构建Milvus过滤表达式"""
        expressions = []
        
        for key, value in filters.items():
            if isinstance(value, str):
                expressions.append(f'{key} == "{value}"')
            elif isinstance(value, (int, float)):
                expressions.append(f'{key} == {value}')
            elif isinstance(value, list):
                # 处理列表类型的过滤条件
                if all(isinstance(v, str) for v in value):
                    expr = " or ".join([f'{key} == "{v}"' for v in value])
                    expressions.append(f"({expr})")
                else:
                    expr = " or ".join([f'{key} == {v}' for v in value])
                    expressions.append(f"({expr})")
        
        return " and ".join(expressions)
    
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
        
        for i, record in enumerate(raw_documents):
            question = record.get(self.data_config.question_field, "")
            answer = record.get(self.data_config.answer_field, "")
            
            # 构建完整文本（用于BM25和vec_text）
            full_text = f"问题: {question}\n\n答案: {answer}"
            
            # 构建元数据
            metadata = {
                "question": question,
                "answer": answer,
                "source": record.get(self.data_config.source_field, self.data_config.default_source),
                "question_text": question,      # 专门用于vec_question字段
                "full_text": full_text,         # 专门用于vec_text字段
                "id": record.get(self.data_config.id_field, f"doc_{i}")
            }
            
            # 保留其他字段
            for key, value in record.items():
                if key not in [
                    self.data_config.question_field,
                    self.data_config.answer_field,
                    self.data_config.source_field,
                    self.data_config.id_field
                ]:
                    metadata[key] = value
            
            # 创建文档（page_content用于BM25稀疏向量）
            document = Document(
                page_content=full_text,  # BM25基于完整文本
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
            logger.info("准备多向量字段文档...")
            documents = self.processor.prepare_multi_vector_documents(raw_data)
            
            # 3. 批量插入
            logger.info(f"开始插入 {len(documents)} 个文档...")
            batch_size = self.config.ingestion.batch_size
            total_inserted = 0
            
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                try:
                    ids = self.kb.add_documents(batch)
                    total_inserted += len(ids)
                    logger.info(f"已插入批次 {i//batch_size + 1}，累计 {total_inserted} 个文档")
                except Exception as e:
                    logger.error(f"插入批次失败: {e}")
                    continue
            
            logger.info(f"高级入库完成！总共插入 {total_inserted} 个文档")
            logger.info(f"集合信息: {self.kb.get_collection_info()}")
            
            return True
            
        except Exception as e:
            logger.error(f"高级入库流水线失败: {e}")
            return False