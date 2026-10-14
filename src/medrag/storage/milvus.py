"""Milvus 知识库（v2 的单行多向量布局）

M0 只做迁移：表结构、检索与融合逻辑与 v2 一致；向量化改走 OpenAICompatEmbedding，
返回值由 LangChain Document 换成字段相同的 LegacyDocument。
M1 改造为 BaseVectorStorage 的实现 MilvusVectorStorage，表结构由 Chunk 生成（docs/architecture.md 第 9 节）。
"""
import asyncio
import logging
from typing import List

from pymilvus import (
    AnnSearchRequest,
    AsyncMilvusClient,
    DataType,
    Function,
    FunctionType,
    MilvusClient,
    RRFRanker,
    WeightedRanker,
)

from ..config.models import EmbeddingConfig, MilvusConfig, SearchRequest, SingleSearchRequest
from ..llm.openai_compat import OpenAICompatEmbedding
from ..types import LegacyDocument
from .backends import milvus_connection
from .milvus_io import insert_rows

logger = logging.getLogger(__name__)

AllFields = [
    "pk", "text",
    "summary", "document",
    "source", "source_name",
    "lt_doc_id", "chunk_id",
    "summary_dense", "text_dense", "text_sparse"
]

class MedicalHybridKnowledgeBase:
    """医疗混合知识库 - 支持多向量字段检索

    检索与写入走 AsyncMilvusClient；建表、建索引这类低频 DDL 仍用同步 MilvusClient
    （AsyncMilvusClient 没有 has_collection 等方法），由 to_thread 包装。
    """

    def __init__(self, milvus: MilvusConfig, embedding: EmbeddingConfig):

        self.milvus_config = milvus
        self.embedding_config = embedding

        # 创建多个嵌入模型实例
        self.summary_embedding = self._create_summary_embedding()
        self.text_embedding = self._create_text_embedding()

        # 向量存储实例
        # 未配置 milvus.uri 时连接参数指向 Milvus Lite 的本地文件（storage/backends.py）
        self._conn = milvus_connection(self.milvus_config)
        self.client = MilvusClient(**self._conn)
        self._aclient: AsyncMilvusClient | None = None
        self._loaded = False   # 本实例是否已确认集合处于 load 状态
        self.EMBEDDERS = {
            "summary_dense": self.summary_embedding,
            "text_dense": self.text_embedding
        }

        self._vocab = None
        if self.embedding_config.text_sparse.provider == "self":
            # 自己管理词表时还要创建 BM25 编码器；pkuseg 是可选依赖，只在这里导入
            from ..retrieval.sparse.bm25 import BM25SparseEmbedding
            from ..retrieval.sparse.vocab import BM25Vectorizer, Vocabulary

            self._vocab = Vocabulary.load(self.embedding_config.text_sparse.vocab_path_or_name)
            self._bm25 = BM25Vectorizer(
                vocab=self._vocab,
                domain_model=self.embedding_config.text_sparse.domain_model,
                k1=self.embedding_config.text_sparse.k1,
                b=self.embedding_config.text_sparse.b
            )
            self.EMBEDDERS["text_sparse"] = BM25SparseEmbedding(self._vocab, self._bm25)

    @property
    def collection_name(self) -> str:
        return self.milvus_config.collection_name

    @property
    def manage_sparse_self(self) -> bool:
        """稀疏向量是否由本项目自管理词表（否则交给 Milvus 托管 BM25）"""
        return self.embedding_config.text_sparse.provider == "self"

    @property
    def sparse_vocab_ready(self) -> bool:
        """自管理词表时，词表是否已经构建完成"""
        return not self.manage_sparse_self or self._vocab is not None

    @property
    def aclient(self) -> AsyncMilvusClient:
        """懒加载：底层是 grpc.aio，必须在事件循环内创建，首次访问一定发生在协程里"""
        if self._aclient is None:
            self._aclient = AsyncMilvusClient(**self._conn)
        return self._aclient

    def _create_summary_embedding(self) -> OpenAICompatEmbedding:
        """问题向量（summary_dense 字段）"""
        return OpenAICompatEmbedding(self.embedding_config.summary_dense)

    def _create_text_embedding(self) -> OpenAICompatEmbedding:
        """正文向量（text_dense 字段）"""
        return OpenAICompatEmbedding(self.embedding_config.text_dense)

    def _create_collection_sync(self):
        """ 使用原生 Milvus 客户端创建Collection"""
        assert self.embedding_config.summary_dense.dimension == self.embedding_config.text_dense.dimension, "多向量单行存储时，两个嵌入模型嵌入向量维度必须相同"
        dim = self.embedding_config.summary_dense.dimension
        # 集合已存在：drop_old=false 时直接沿用（追加写入），否则删掉重建；不存在则总是新建
        if self.client.has_collection(collection_name=self.milvus_config.collection_name):
            if not self.milvus_config.drop_old:
                return self.client
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
        if not self.manage_sparse_self:
            bm25_fn = Function(
                name="bm25_text_to_sparse",
                function_type=FunctionType.BM25,
                input_field_names=["text"],
                output_field_names=["text_sparse"],
            )
            schema.add_function(bm25_fn)

        self.client.create_collection(collection_name=self.milvus_config.collection_name, schema=schema)
        return self.client

    async def _create_collection(self):
        return await asyncio.to_thread(self._create_collection_sync)

    def _build_index_sync(self):
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
        if self.manage_sparse_self:
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

    async def build_index(self):
        await asyncio.to_thread(self._build_index_sync)

    async def flush(self):
        """把增长段封存落盘。

        批量写入结束后调用：未封存的增长段只能走临时索引，召回明显偏低（M0 回归实测 Hit@10 相差约 13pp），
        而 Milvus 何时自动封存不确定，不 flush 的话同一份数据的检索结果会随时机变化。
        """
        await asyncio.to_thread(self.client.flush, self.milvus_config.collection_name)

    @staticmethod
    def _to_text(value) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        return str(value)

    async def add_documents(self, documents: List[LegacyDocument]) -> int:
        """添加文档，自动处理多向量字段。摘要、正文的嵌入整批并发调用。"""
        summaries = [self._to_text(doc.metadata.get("summary", "")) for doc in documents]
        texts = [self._to_text(doc.page_content) for doc in documents]

        # 两路稠密向量整批、同时请求；稀疏向量是 CPU 密集的分词，放到线程里
        tasks = [
            self.EMBEDDERS["summary_dense"](summaries),
            self.EMBEDDERS["text_dense"](texts),
        ]
        if self.manage_sparse_self:
            tasks.append(asyncio.to_thread(self.EMBEDDERS["text_sparse"].embed_documents, texts))
        embedded = await asyncio.gather(*tasks)
        summary_dense, text_dense = embedded[0], embedded[1]
        text_sparse = embedded[2] if self.manage_sparse_self else None

        # 向量只放进 rows，不回写 doc.metadata：
        # documents 的生命周期覆盖整个入库流程，挂上去的向量一条都不会被回收。
        # 4096 维 × 2 路 × Python float 装箱 ≈ 262 KB/条，5 万条就是十几 GB。
        rows = []
        for i, doc in enumerate(documents):
            filtered = {k: v for k, v in doc.metadata.items() if k in AllFields}
            if not self.milvus_config.auto_id:
                # 如果不采用自动id，则默认id实现为quesiton的hash值，以便插入时覆盖重复数据
                filtered["pk"] = doc.metadata.get("hash_id", "")
            filtered["summary"] = summaries[i]
            filtered["text"] = texts[i]
            filtered["summary_dense"] = summary_dense[i]
            filtered["text_dense"] = text_dense[i]
            if text_sparse is not None:
                filtered["text_sparse"] = text_sparse[i]
            rows.append(filtered)

        # 交给 Milvus 之后立刻断开本批向量的引用，峰值只保留在途的几批
        del summary_dense, text_dense, text_sparse, embedded, tasks

        inserted = len(rows)
        try:
            await insert_rows(
                client=self.aclient,
                collection_name=self.milvus_config.collection_name,
                rows=rows,
                show_progress=False  # 小批量不显示进度条
            )
        finally:
            rows.clear()
        return inserted


    async def _encode_query(self, query: str, anns_field: str):
        if anns_field != "text_sparse":
            return await self.EMBEDDERS[anns_field].embed_query(query)
        if self.manage_sparse_self:
            # 自己管理的词表：分词是 CPU 密集的，放到线程里
            return await asyncio.to_thread(self.EMBEDDERS[anns_field].embed_query, query)
        return query  # 自动托管的BM25算法时，传入的查询不需要做任何处理

    async def _search(
        self,
        query: str, # 查询的问题
        single_search_request: SingleSearchRequest,
        collection_name: str,
        output_fields: list[str]
    ):
        """ Milvus 原生的查询单个问题 https://milvus.io/docs/zh/filtered-search.md """
        data = await self._encode_query(query=query, anns_field=single_search_request.anns_field)

        return await self.aclient.search(
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

    async def _build_ann_search_request(
        self,
        query: str,
        single_search_request: SingleSearchRequest
    ) -> AnnSearchRequest:
        """ 构建子 AnnSearchRequest 请求"""
        data = await self._encode_query(query=query, anns_field=single_search_request.anns_field)
        return AnnSearchRequest(
            data=[data],
            anns_field=single_search_request.anns_field,
            param={
                "metric_type": single_search_request.metric_type,
                "params": single_search_request.search_params
            },
            limit=single_search_request.limit,
            expr=single_search_request.expr
        )

    async def _hybrid_search(
        self,
        search: SearchRequest
    ):
        """ Milvus 原生混合查询 https://milvus.io/docs/zh/multi-vector-search.md"""
        # 多路子请求的编码互不依赖，并发执行
        anns = await asyncio.gather(*(
            self._build_ann_search_request(query=search.query, single_search_request=item)
            for item in search.requests
        ))
        if search.fuse.method == "rrf":
            rank = RRFRanker(search.fuse.k)
        else:
            rank = WeightedRanker(*search.fuse.weights)
        return await self.aclient.hybrid_search(
            collection_name=search.collection_name,
            reqs=anns,
            ranker=rank,
            limit=search.limit,
            output_fields=search.output_fields
        )

    def _ensure_loaded_sync(self, collection_name: str) -> None:
        state = self.client.get_load_state(collection_name).get("state")
        if str(state).endswith("Loaded") and "NotLoad" not in str(state):
            return
        self.client.load_collection(collection_name)

    async def ensure_loaded(self, collection_name: str | None = None) -> None:
        """检索前确认集合已 load。

        v2 只在入库建索引后 load 一次，默认集合一直处于 load 状态。Milvus 服务会记住这个状态，
        但 Milvus Lite 重启（换一个进程）后集合会回到 released，直接检索报 "call load() before search"。
        每个实例只检查一次，之后不再有额外开销。
        """
        if self._loaded:
            return
        await asyncio.to_thread(self._ensure_loaded_sync, collection_name or self.collection_name)
        self._loaded = True

    async def search(self, req: SearchRequest) -> List[LegacyDocument]:
        await self.ensure_loaded(req.collection_name)
        if len(req.requests) == 1:
            # 只有一个请求搜索，走普通的search
            outputs = (await self._search(
                req.query,
                req.requests[0],
                req.collection_name,
                req.output_fields
            ))[0]  # 批量中的第一条，这里先不支持批量查询
        else:
            # 有多个请求搜索，走混合search
            outputs = (await self._hybrid_search(req))[0]

        return [
            LegacyDocument(
                page_content=item.get("text", ""),
                metadata={
                    "pk": item.get("pk", ""),
                    "distance": item.get("distance", 99999),
                    "chunk_id": item.get("chunk_id", -1),
                    "summary": item.get("summary", ""),
                    "document": item.get("document", ""),
                    "source": item.get("source", ""),
                    "source_name": item.get("source_name", ""),
                    "lt_doc_id": item.get("lt_doc_id", ""),
                },
            )
            for item in outputs
        ]

    async def close(self):
        if self._aclient is not None:
            await self._aclient.close()
            self._aclient = None
        self.client.close()
        await asyncio.gather(self.summary_embedding.aclose(), self.text_embedding.aclose())
