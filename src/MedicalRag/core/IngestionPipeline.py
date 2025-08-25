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
from ..config.models import AppConfig, EmbeddingConfig, DenseConfig, DataConfig
from .utils import create_embedding_client
import hashlib
from pymilvus import MilvusClient, DataType, Collection, connections, FunctionType, Function
from functools import lru_cache
from pathlib import Path
from .KnowledgeBase import MedicalHybridKnowledgeBase
import traceback
from ..knowledge.sparse import Vocabulary

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


def prepare_multi_vector_documents(data_config: DataConfig, raw_documents: List[Dict[str, Any]]) -> List[Document]:
    documents = []
        
    for record in tqdm(raw_documents, desc="预处理文档"):
        summary = record.get(data_config.summary_field, "")
        document = record.get(data_config.document_field, "")
        
        # 构建完整文本（用于BM25和vec_text）
        if record.get(data_config.default_source, "qa") == "qa":
            text = f"问题: {summary}\n\n答案: {document}"
        else:
            text = document
        
        # 构建元数据
        metadata = {
            "summary": summary,
            "document": document,
            "source": record.get(data_config.source_field, data_config.default_source),
            "source_name": record.get(data_config.source_name_field, data_config.default_source_name),
            "hash_id": hashlib.md5(summary.encode('UTF-8')).hexdigest(),
            "lt_doc_id": record.get(data_config.lt_doc_id_field, data_config.default_lt_doc_id),
            "chunk_id": record.get(data_config.chunk_id_field, data_config.default_chunk_id)
        }
        
        # 保留其他字段
        for key, value in record.items():
            if key not in [
                data_config.summary_field,
                data_config.document_field,
                data_config.source_field,
                data_config.source_name_field,
                data_config.lt_doc_id_field,
                data_config.chunk_id_field
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
class IngestionPipeline:
    """高级入库流水线 - 使用多向量字段"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.kb = MedicalHybridKnowledgeBase(config)
    
    def run(self, raw_data: List[Dict[str, Any]]) -> bool:
        """运行高级入库流水线"""
        
        if self.config.embedding.text_sparse.provider == "self":
            _vocab = Vocabulary.load(self.config.embedding.text_sparse.vocab_path_or_name)
            if _vocab is None:  # 未完成初始化
                raise "请完成词表初始化，或者把稀疏向量交给Milvus管理"
        try:
            # 1. 初始化多向量字段集合
            logger.info("初始化多向量Milvus集合...")
            client = self.kb._create_collection()
            
            # 2. 处理数据
            logger.info("预处理多向量字段文档...")
            documents = prepare_multi_vector_documents(data_config=self.config.data, raw_documents=raw_data)
            
            # 3. 批量插入
            logger.info(f"开始插入 {len(documents)} 个文档...")
            batch_size = 10
            total_inserted = 0
            
            for i in tqdm(range(0, len(documents), batch_size)):
                batch = documents[i:i + batch_size]
                try:
                    ids = self.kb.add_documents(batch)
                    total_inserted += ids
                except Exception as e:
                    logger.error(f"插入批次失败: {e}")
                    continue
            
            logger.info(f"开始构建索引")
            self.kb.build_index()
            
            logger.info(f"高级入库完成！总共插入 {total_inserted} 个文档")
            logger.info(f"集合信息: {self.kb.get_collection_info()}")
            
            return True
            
        except Exception as e:
            logger.error(f"高级入库流水线失败: {e}")
            print(traceback(e))
            return False