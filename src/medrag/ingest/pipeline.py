"""入库流水线（v2 行为）

M0 只做迁移：字段映射、文本拼接、主键规则与 v2 一致；唯一的行为变化是写入后先 flush 再建索引。
已知问题留到 M1 随统一 Schema 一起改：
- 主键 hash_id = md5(summary)，summary 为空的文献 chunk 会互相覆盖；
- 只区分 qa 与其他来源，QA 文本的中文前缀写死在这里。
v2 里没有调用方的 get_stopwords（读本地停用词表）已删除，停用词表文件保留在 retrieval/sparse/stopword_lists/。
"""
import asyncio
import hashlib
import logging
from typing import Any, Dict, List

from tqdm import tqdm

from ..config.models import DataConfig
from ..storage.milvus import MedicalHybridKnowledgeBase
from ..types import LegacyDocument

logger = logging.getLogger(__name__)


def prepare_multi_vector_documents(data_config: DataConfig, raw_documents: List[Dict[str, Any]]) -> List[LegacyDocument]:
    documents = []
        
    for record in tqdm(raw_documents, desc="预处理文档"):
        summary = record.get(data_config.summary_field, "")
        document = record.get(data_config.document_field, "")
        summary = "" if summary is None else str(summary)
        document = "" if document is None else str(document)
        source = record.get(data_config.source_field, data_config.default_source)
        source_name = record.get(data_config.source_name_field, data_config.default_source_name)
        source = "" if source is None else str(source)
        source_name = "" if source_name is None else str(source_name)
        
        # 构建完整文本（用于BM25和text_dense）
        # 如果是 QA 数据，则 summary = 问题、 document = 回答 、 text = 问题 + 回答
        # 如果是 literature 文献数据，则 summary = 特殊抽取的文段摘要、document = 正文、text = document = 正文
        if source == "qa":
            text = f"问题: {summary}\n\n答案: {document}"
        else:
            text = document
        
        # 构建元数据
        metadata = {
            "summary": summary,
            "document": document,
            "source": source,
            "source_name": source_name,
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
        document = LegacyDocument(
            page_content=text,  # BM25基于完整文本
            metadata=metadata
        )
        
        documents.append(document)
    
    return documents


    
# 高级入库流水线
class IngestionPipeline:
    """高级入库流水线 - 使用多向量字段"""
    
    def __init__(self, data: DataConfig, kb: MedicalHybridKnowledgeBase):
        self.data_config = data
        self.kb = kb

    async def _insert_batches(self, documents: List[LegacyDocument]) -> int:
        """按 data_config.batch_size 分批，起 concurrency 个协程从同一个批次迭代器里取活。

        共享迭代器在单线程事件循环里是安全的，协程之间不会取到同一批；
        任一批失败就取消其余协程，不再继续写入。
        """
        batch_size = self.data_config.batch_size
        concurrency = self.data_config.concurrency
        batches = iter([documents[i:i + batch_size] for i in range(0, len(documents), batch_size)])
        logger.info(f"开始插入 {len(documents)} 个文档（batch_size={batch_size}, concurrency={concurrency}）...")

        bar = tqdm(total=len(documents), desc="入库")

        async def worker() -> int:
            inserted = 0
            for batch in batches:
                inserted += await self.kb.add_documents(batch)
                bar.update(len(batch))
            return inserted

        tasks = [asyncio.create_task(worker()) for _ in range(concurrency)]
        try:
            return sum(await asyncio.gather(*tasks))
        except BaseException:
            for task in tasks:
                task.cancel()
            raise
        finally:
            bar.close()

    async def run(self, raw_data: List[Dict[str, Any]]) -> bool:
        """运行高级入库流水线"""

        if not self.kb.sparse_vocab_ready:
            raise RuntimeError("请完成词表初始化，或者把稀疏向量交给Milvus管理")
        try:
            # 1. 初始化多向量字段集合
            logger.info("初始化多向量Milvus集合...")
            await self.kb._create_collection()

            # 2. 处理数据
            logger.info("预处理多向量字段文档...")
            documents = prepare_multi_vector_documents(data_config=self.data_config, raw_documents=raw_data)

            # 3. 批量插入：批内一起做 embedding、一次写入；最多 concurrency 个批次同时进行
            total_inserted = await self._insert_batches(documents)

            # 先封存再建索引：检索质量不再取决于 Milvus 自动封存的时机（v2 没有这一步）
            await self.kb.flush()
            logger.info(f"开始构建索引")
            await self.kb.build_index()

            logger.info(f"高级入库完成！总共插入 {total_inserted} 个文档")

            return True

        except Exception:
            logger.exception("高级入库流水线失败")
            return False
