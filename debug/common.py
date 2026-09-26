"""联调脚本的公共件。

注意导入顺序：本模块会先清理代理环境变量，再导入 MedicalRag。
所以每个脚本都要**先** `from common import ...`，再导入 MedicalRag 的东西。
"""
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# 1) 清理代理环境变量
#    - ALL_PROXY=socks://... 会让 httpx 在构造客户端时直接抛 ValueError，
#      import langchain_ollama 就崩（ollama 在 import 时会 new 一个 httpx.Client）；
#    - http(s)_proxy 会把 10.88.88.6 的请求也绕进代理，而 no_proxy 里没有它。
#    需要保留代理时设 MEDRAG_KEEP_PROXY=1。
# ---------------------------------------------------------------------------
if not os.environ.get("MEDRAG_KEEP_PROXY"):
    for name in ("ALL_PROXY", "all_proxy", "HTTP_PROXY", "http_proxy",
                 "HTTPS_PROXY", "https_proxy", "NO_PROXY", "no_proxy"):
        os.environ.pop(name, None)

# vLLM 不校验 key，但 openai 客户端要求非空
os.environ.setdefault("VLLM_API_KEY", "EMPTY")

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402

from MedicalRag.config.loader import ConfigLoader  # noqa: E402
from MedicalRag.config.models import (  # noqa: E402
    FusionSpec, SearchRequest, SingleSearchRequest,
)
from MedicalRag.core.KnowledgeBase import MedicalHybridKnowledgeBase  # noqa: E402

CONF_DIR = str(Path(__file__).parent / "conf")

# 少量语料，够验证检索与生成链路，不必拉整个数据集（data/*.jsonl 是 LFS 指针）
QA = [
    ("肚子一阵一阵地痛怎么办", "阵发性腹痛常见于肠痉挛、肠梗阻早期，建议禁食观察、腹部保暖，持续不缓解需就医。"),
    ("高血压有哪些常见症状", "高血压常见头痛、头晕、心悸、耳鸣，多数早期无明显症状，需定期测量血压。"),
    ("感冒和流感怎么区分", "流感起病急、高热明显、全身酸痛重；普通感冒以鼻塞流涕为主，发热轻。"),
    ("吃完饭就想吐是什么原因", "餐后恶心呕吐常见于胃炎、胃排空延迟、胆囊疾病，需结合腹痛部位判断。"),
    ("糖尿病人饮食要注意什么", "控制总热量，主食粗细搭配，定时定量，限制精制糖和饱和脂肪，配合监测血糖。"),
    ("头晕目眩可能是什么病", "头晕可由贫血、低血糖、颈椎病、前庭功能障碍、高血压引起，需查明原因。"),
    ("小孩发烧到几度需要吃退烧药", "腋温超过38.5摄氏度或明显不适可考虑退热药，重点是补液观察精神状态。"),
    ("长期失眠该怎么调整", "固定作息、睡前避免咖啡和屏幕、必要时行认知行为治疗，慎用镇静药物。"),
]

RAW_RECORDS = [{"question": q, "answer": a} for q, a in QA]


def load_config():
    """读 debug/conf/ 下的配置（本地 vLLM + medrag_debug 集合）"""
    return ConfigLoader(CONF_DIR).config


def build_kb(cfg):
    """创建知识库，并打上两个已知问题的补丁（等包侧修好后这里可以删）。

    1. Octen-Embedding-8B 不支持 Matryoshka，请求里不能带 dimensions，
       而 create_embedding_client 目前总会传；
    2. Agent 的 database_search 工具让 LLM 自己填 collection_name，默认值是
       medical_knowledge，不会是这里的 medrag_debug；LLM 还常常挑上 text_sparse
       + IP，而 Milvus 托管 BM25 下 metric 必须是 BM25，会报 MilvusException。
       所以包装 kb.search 把请求校正一遍，否则 Agent 里 database_search 永远
       取不到数据。其他调用方传的本来就是对的，不受影响。
    """
    kb = MedicalHybridKnowledgeBase(cfg.milvus, cfg.embedding)
    for embedder in kb.EMBEDDERS.values():
        embedder.dimensions = None

    raw_search = kb.search

    async def search_with_fixed_request(req):
        dense = [r for r in req.requests if r.anns_field != "text_sparse"]
        if not dense:
            dense = [SingleSearchRequest(anns_field="summary_dense", metric_type="COSINE",
                                         search_params={"ef": 64}, limit=10)]
        return await raw_search(req.model_copy(update={
            "collection_name": kb.collection_name,
            "requests": dense,
            # LLM 经常只挑一两个 output_fields，取不到 text 的话 Document.page_content
            # 就是空的，RAG 节点等于没拿到资料
            "output_fields": ["text", "summary", "document", "source", "source_name",
                              "lt_doc_id", "chunk_id"],
        }))

    kb.search = search_with_fixed_request
    return kb


def dense_search_config(cfg, limit: int = 3) -> SearchRequest:
    """两路稠密向量的混合检索配置。

    没有用 text_sparse：Milvus 托管 BM25 时 metric 必须是 BM25，而
    SingleSearchRequest.metric_type 目前只允许 COSINE / IP，表达不出来。
    自管理词表（provider: self）时可以正常用 IP，届时可换成包里的默认配置。
    """
    return SearchRequest(
        query="",
        collection_name=cfg.milvus.collection_name,
        requests=[
            SingleSearchRequest(anns_field="summary_dense", metric_type="COSINE",
                                search_params={"ef": 64}, limit=10),
            SingleSearchRequest(anns_field="text_dense", metric_type="COSINE",
                                search_params={"ef": 64}, limit=10),
        ],
        output_fields=["summary", "document", "source", "source_name",
                       "lt_doc_id", "chunk_id", "text"],
        fuse=FusionSpec(method="weighted", weights=[0.6, 0.4]),
        limit=limit,
    )


def drop_collection(kb):
    """删掉联调用的集合"""
    if kb.client.has_collection(kb.collection_name):
        kb.client.drop_collection(kb.collection_name)


async def clear_session(store, *session_ids):
    """清掉这些会话在 Redis 里的数据。

    故意不删 medrag:lock:{sid}：锁自己会过期，而删掉它会把另一个正在跑的进程
    持有的锁抢掉，那边 release 时就抛 LockNotOwnedError。
    """
    keys = []
    for sid in session_ids:
        keys += [
            f"medrag:chat:{sid}:history",
            f"medrag:chat:{sid}:summary",
            f"medrag:chat:{sid}:tokens",
            f"medrag:agent:{sid}:state",
        ]
    await store.r.delete(*keys)


# ---------------------------------------------------------------------------
# 假模型：用于不想占 GPU 的快速回归
# ---------------------------------------------------------------------------

# Agent 的若干节点要求结构化 JSON 输出，这一份同时满足 AskMess 与 SplitQuery
STRUCTURED_REPLY = (
    '{"need_ask": false, "questions": [], '
    '"need_split": false, "sub_query": [], "rewrite_query": "肚子痛"}'
)


class FakeLLM(GenericFakeChatModel):
    """带 usage_metadata 的假模型。

    bind_tools 直接返回自身：假模型不会真的触发工具调用。
    """

    reply: str = "这是一个回答"
    latency: float = 0.2   # 模拟一次 LLM 往返，用来观察并发收益

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        msg = AIMessage(
            content=self.reply,
            usage_metadata={"input_tokens": 10, "output_tokens": 6, "total_tokens": 16},
        )
        return ChatResult(generations=[ChatGeneration(message=msg)])

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        import asyncio
        await asyncio.sleep(self.latency)
        return self._generate(messages, stop, run_manager, **kwargs)

    def bind_tools(self, tools, **kwargs):
        return self


def fake_websearch(query, cnt):
    """只给 06_fake_fast.py 用的联网检索桩件，纯粹为了走通异步调用路径。

    真模型的脚本不要用它，见 has_websearch_credential()。
    """
    from langchain_core.documents import Document
    return [Document(page_content=f"[网络] 关于「{query}」的资料 {i}") for i in range(min(2, cnt))]


def has_websearch_credential() -> bool:
    """腾讯云联网检索的凭据在不在"""
    return bool(os.environ.get("TENCENTCLOUD_SECRET_ID") and os.environ.get("TENCENTCLOUD_SECRET_KEY"))


def apply_websearch_policy(agent_config):
    """没有凭据就关掉联网检索，而不是拿假数据顶。返回 (配置, 说明文字)。

    联网检索节点在 remain_doc_index 为空时会清掉本地检索到的文档，
    所以拿假数据顶会把本地库的真实检索结果冲掉，看不出效果。
    """
    if has_websearch_credential():
        return agent_config, "联网检索：已启用（腾讯云）"
    return (agent_config.model_copy(update={"network_search_enabled": False}),
            "联网检索：已关闭（未配置 TENCENTCLOUD_SECRET_ID / KEY），只用本地知识库")
