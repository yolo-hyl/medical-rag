"""M0 回归门槛：v2 基线与新实现在 A 层回归语料上的检索结果一致性

门槛（D17，见 docs/solution.md 研制路线中的 M0 与决策记录）：
- 硬门槛：`verify` 逐项一致——两个集合逐行比对数据、新旧代码的查询编码，都必须 0 差异；
- 软门槛：检索指标落在噪声范围内。Milvus 的 HNSW 近似检索加上并发写入，同一实现重跑的 top-10 重合率也只有约 91%，
  所以用 `compare --rerun` 传入新实现重跑一次的结果，以两次重跑的差异再加 2pp 作为容差；不传时检索指标只作参考。

用法（在仓库根目录执行）：

    # 1. 构造回归语料：new_qa_200 的 198 篇标准文档 + qa_50000 中随机 9,802 篇干扰文档（seed=42）
    python scripts/regression/m0_parity.py corpus --out runs/m0/corpus.jsonl

    # 2. 用这份语料构建自管理 BM25 词表（两边共用同一个词表文件）
    python scripts/regression/m0_parity.py vocab --corpus runs/m0/corpus.jsonl --out runs/m0/vocab.pkl.gz

    # 3. 分别跑新旧实现（旧实现取自 v2 基线 commit 的 worktree）
    git worktree add /tmp/v2base cff9b8e
    python scripts/regression/m0_parity.py run --impl new --corpus runs/m0/corpus.jsonl \
        --vocab runs/m0/vocab.pkl.gz --out runs/m0/new.json --base-url <embedding 服务> --model <模型> ...
    PYTHONPATH=/tmp/v2base/src python scripts/regression/m0_parity.py run --impl old ... --out runs/m0/old.json

    # 4. 对比
    python scripts/regression/m0_parity.py verify ...                     # 硬门槛（需要旧包在 PYTHONPATH 上）
    python scripts/regression/m0_parity.py run --impl new --out runs/m0/new_rerun.json ...   # 再跑一次，量出噪声
    python scripts/regression/m0_parity.py compare runs/m0/old.json runs/m0/new.json --rerun runs/m0/new_rerun.json

两边都使用 v2 默认的检索配置：summary_dense(COSINE) + text_sparse(IP) 加权融合 0.8 / 0.2，返回 10 条。
"""
import argparse
import asyncio
import json
import os
import random
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
QA_PATH = REPO / "data" / "qa_50000.jsonl"
EVAL_PATH = REPO / "data" / "eval" / "new_qa_200.jsonl"
CORPUS_SIZE = 10_000
SEED = 42


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------------------------------------------------------------------------
# 1. 语料
# ---------------------------------------------------------------------------
def cmd_corpus(args):
    gold = {r["question_hash"] for r in read_jsonl(EVAL_PATH)}
    rows = read_jsonl(QA_PATH)
    gold_rows, seen = [], set()
    for r in rows:
        # qa_50000 里有 57 个重复的 question_hash，标准文档只取第一条
        if r["question_hash"] in gold and r["question_hash"] not in seen:
            gold_rows.append(r)
            seen.add(r["question_hash"])
    others = [r for r in rows if r["question_hash"] not in gold]
    picked = random.Random(SEED).sample(others, CORPUS_SIZE - len(gold_rows))
    corpus = gold_rows + picked
    random.Random(SEED).shuffle(corpus)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in corpus:
            f.write(json.dumps({"question": r["question"], "answer": r["answer"]}, ensure_ascii=False) + "\n")
    print(f"回归语料 {len(corpus)} 篇（标准文档 {len(gold_rows)} 篇）→ {out}")


# ---------------------------------------------------------------------------
# 2. 词表
# ---------------------------------------------------------------------------
def cmd_vocab(args):
    from medrag.retrieval.sparse.vocab import BM25Vectorizer, Vocabulary

    corpus = read_jsonl(args.corpus)
    texts = [f"问题: {r['question']}\n\n答案: {r['answer']}" for r in corpus]   # 与入库时 QA 的 text 一致
    vocab = Vocabulary()
    vectorizer = BM25Vectorizer(vocab, domain_model="medicine")
    start = time.time()
    for toks in vectorizer.tokenize_parallel(texts, workers=args.workers, chunksize=64):
        vocab.add_document(toks)
    vocab.freeze()
    vocab.save(str(Path(args.out).resolve()))
    print(f"词表 {len(vocab.token2id)} 个词，{vocab.N} 篇文档，用时 {time.time() - start:.1f}s → {args.out}")


# ---------------------------------------------------------------------------
# 3. 入库 + 检索
# ---------------------------------------------------------------------------
def _configs(args):
    """按实现构造配置对象：两边的配置字段名不同（v2: provider/env_key_name，新: api_key_env）"""
    if args.impl == "old":
        from MedicalRag.config import models as m
        if not args.api_key_env:
            os.environ.setdefault("OPENAI_API_KEY", "EMPTY")   # LangChain 的 OpenAIEmbeddings 要求非空
        dense = m.DenseConfig(provider="openai", model=args.model, base_url=args.base_url,
                              env_key_name=args.api_key_env, dimension=args.dim)
    else:
        from medrag.config import models as m
        dense = m.DenseConfig(model=args.model, base_url=args.base_url, api_key_env=args.api_key_env,
                              dimension=args.dim, max_batch=args.batch_size)
    embedding = m.EmbeddingConfig(summary_dense=dense, text_dense=dense,
                                  text_sparse=m.SparseConfig(provider="self", vocab_path_or_name=str(Path(args.vocab).resolve())))
    milvus = m.MilvusConfig(uri=args.milvus_uri, collection_name=f"m0_parity_{args.impl}", drop_old=True, auto_id=False)
    data = m.DataConfig(batch_size=args.batch_size, concurrency=args.concurrency)
    search = m.SearchRequest(
        query="",
        collection_name=milvus.collection_name,
        requests=[
            m.SingleSearchRequest(anns_field="summary_dense", metric_type="COSINE", search_params={"ef": 64}, limit=10, expr=""),
            m.SingleSearchRequest(anns_field="text_sparse", metric_type="IP", search_params={"drop_ratio_search": 0.0}, limit=10, expr=""),
        ],
        output_fields=["summary", "document", "source", "source_name", "lt_doc_id", "chunk_id", "text"],
        fuse=m.FusionSpec(method="weighted", weights=[0.8, 0.2]),
        limit=10,
    )
    return embedding, milvus, data, search


def _impl_classes(impl):
    if impl == "old":
        from MedicalRag.core.IngestionPipeline import IngestionPipeline
        from MedicalRag.core.KnowledgeBase import MedicalHybridKnowledgeBase
    else:
        from medrag.ingest import IngestionPipeline
        from medrag.storage.milvus import MedicalHybridKnowledgeBase
    return IngestionPipeline, MedicalHybridKnowledgeBase


async def _ingest(args):
    IngestionPipeline, KB = _impl_classes(args.impl)
    embedding, milvus, data, _ = _configs(args)
    kb = KB(milvus, embedding)
    try:
        start = time.time()
        if not await IngestionPipeline(data, kb).run(read_jsonl(args.corpus)):
            raise RuntimeError("入库失败")
        print(f"[{args.impl}] 入库用时 {time.time() - start:.1f}s")
    finally:
        await kb.close()


async def _search(args):
    _, KB = _impl_classes(args.impl)
    embedding, milvus, _, search = _configs(args)
    kb = KB(milvus, embedding)
    try:
        results = {}
        for item in read_jsonl(EVAL_PATH):
            docs = await kb.search(search.model_copy(update={"query": item["new_question"]}))
            results[item["question_hash"]] = [d.metadata["pk"] for d in docs]
    finally:
        await kb.close()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"impl": args.impl, "results": results}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[{args.impl}] {len(results)} 题的 top-10 → {args.out}")


def wait_sealed_and_indexed(uri: str, collection: str, timeout: float = 600) -> None:
    """检索前先 flush 并等索引建完。

    刚写入的数据停留在未封存的增长段里，检索走临时索引、召回偏低，什么时候自动封存取决于 Milvus 的时机。
    不等这一步，两次入库同样数据的检索结果会因为封存时机不同而相差十几个百分点（M0 首次对比时实测）。
    """
    from pymilvus import MilvusClient

    client = MilvusClient(uri=uri)
    try:
        client.flush(collection)
        deadline = time.time() + timeout
        while time.time() < deadline:
            infos = [client.describe_index(collection, name) for name in client.list_indexes(collection)]
            if all(i["pending_index_rows"] == 0 and i["indexed_rows"] == i["total_rows"] > 0 for i in infos):
                return
            time.sleep(2)
        raise TimeoutError(f"{collection} 的索引在 {timeout}s 内没有建完")
    finally:
        client.close()


def cmd_run(args):
    if not args.skip_ingest:
        asyncio.run(_ingest(args))
    wait_sealed_and_indexed(args.milvus_uri, f"m0_parity_{args.impl}")
    asyncio.run(_search(args))


# ---------------------------------------------------------------------------
# 4. 对比
# ---------------------------------------------------------------------------
def _hits(results):
    n = len(results)
    hit = {k: sum(gold in pks[:k] for gold, pks in results.items()) / n for k in (1, 5, 10)}
    mrr = sum(1 / (pks.index(gold) + 1) if gold in pks else 0 for gold, pks in results.items()) / n
    return hit, mrr


def _overlap(a, b):
    return sum(len(set(a[q]) & set(b[q])) / max(1, len(a[q])) for q in a) / len(a)


def _load_results(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))["results"]


SLACK = 0.02   # 噪声之外再留的余量


def cmd_compare(args):
    """软门槛：检索指标的新旧差异不超过同一实现重跑的波动 + 2pp；硬门槛见 verify"""
    old, new = _load_results(args.old), _load_results(args.new)
    assert old.keys() == new.keys(), "两边的题目不一致"
    overlap = _overlap(old, new)
    identical = sum(old[q] == new[q] for q in old)
    (h_old, mrr_old), (h_new, mrr_new) = _hits(old), _hits(new)
    print(f"题目数：{len(old)}；top-10 集合重合率：{overlap:.2%}；排序完全一致：{identical}/{len(old)}")
    print(f"{'':8s}{'Hit@1':>8s}{'Hit@5':>8s}{'Hit@10':>8s}{'MRR':>8s}")
    for name, h, mrr in (("v2", h_old, mrr_old), ("medrag", h_new, mrr_new)):
        print(f"{name:8s}{h[1]:8.2%}{h[5]:8.2%}{h[10]:8.2%}{mrr:8.4f}")

    if not args.rerun:
        print("未提供 --rerun：检索指标只作参考，不做判定（硬门槛以 verify 为准）")
        return
    rerun = _load_results(args.rerun)
    assert rerun.keys() == new.keys(), "重跑结果的题目不一致"
    noise_overlap = _overlap(new, rerun)
    h_rerun, _ = _hits(rerun)
    tolerance = {k: abs(h_new[k] - h_rerun[k]) + SLACK for k in (1, 5, 10)}
    print(f"噪声底线（新实现重跑两次）：重合率 {noise_overlap:.2%}；"
          + "，".join(f"Hit@{k} 容差 ±{tolerance[k]:.2%}" for k in (1, 5, 10)))
    passed = overlap >= noise_overlap - SLACK and all(abs(h_old[k] - h_new[k]) <= tolerance[k] for k in (1, 5, 10))
    print("软门槛：" + ("通过" if passed else "未通过"))
    sys.exit(0 if passed else 1)


def cmd_verify(args):
    """排除 Milvus 的不确定性，直接比对代码行为：
    1) 两个集合逐行一致（文本、两路稠密向量、稀疏向量）；
    2) 198 个查询在新旧代码下的编码一致（需要旧包在 PYTHONPATH 上）。
    """
    from pymilvus import MilvusClient

    client = MilvusClient(uri=args.milvus_uri)
    fields = ["summary", "document", "text", "summary_dense", "text_dense", "text_sparse"]
    old_ids = sorted(r["pk"] for r in client.query("m0_parity_old", filter='pk != ""', output_fields=["pk"], limit=16384))
    new_ids = sorted(r["pk"] for r in client.query("m0_parity_new", filter='pk != ""', output_fields=["pk"], limit=16384))
    assert old_ids == new_ids, f"主键集合不同：旧 {len(old_ids)}，新 {len(new_ids)}"
    mismatch = []
    for i in range(0, len(old_ids), 500):
        ids = old_ids[i:i + 500]
        a = {r["pk"]: r for r in client.get("m0_parity_old", ids=ids, output_fields=fields)}
        b = {r["pk"]: r for r in client.get("m0_parity_new", ids=ids, output_fields=fields)}
        for pk in ids:
            ra, rb = a[pk], b[pk]
            for f in fields:
                va, vb = ra[f], rb[f]
                same = (max(abs(x - y) for x, y in zip(va, vb)) < 1e-6) if f.endswith("_dense") else va == vb
                if not same:
                    mismatch.append((pk, f))
    client.close()
    print(f"数据层：{len(old_ids)} 行，逐行比对不一致 {len(mismatch)} 处" + (f"，例如 {mismatch[:3]}" if mismatch else ""))

    async def encode_all():
        from MedicalRag.core.KnowledgeBase import MedicalHybridKnowledgeBase as Old
        from medrag.storage.milvus import MedicalHybridKnowledgeBase as New
        eo, mo, _, _ = _configs(argparse.Namespace(**{**vars(args), "impl": "old"}))
        en, mn, _, _ = _configs(argparse.Namespace(**{**vars(args), "impl": "new"}))
        ko, kn = Old(mo, eo), New(mn, en)
        bad = 0
        try:
            for item in read_jsonl(EVAL_PATH):
                q = item["new_question"]
                for f in ("summary_dense", "text_dense", "text_sparse"):
                    va, vb = await ko._encode_query(q, f), await kn._encode_query(q, f)
                    same = va == vb if isinstance(va, dict) else max(abs(x - y) for x, y in zip(va, vb)) < 1e-6
                    bad += not same
        finally:
            await ko.close()
            await kn.close()
        return bad

    bad = asyncio.run(encode_all())
    print(f"查询编码：198 题 × 3 路，不一致 {bad} 处")
    passed = not mismatch and bad == 0
    print("硬门槛（代码行为一致性）：" + ("通过" if passed else "未通过"))
    sys.exit(0 if passed else 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("corpus")
    p.add_argument("--out", default="runs/m0/corpus.jsonl")
    p.set_defaults(func=cmd_corpus)

    p = sub.add_parser("vocab")
    p.add_argument("--corpus", default="runs/m0/corpus.jsonl")
    p.add_argument("--out", default="runs/m0/vocab.pkl.gz")
    p.add_argument("--workers", type=int, default=8)
    p.set_defaults(func=cmd_vocab)

    p = sub.add_parser("run")
    p.add_argument("--impl", choices=["old", "new"], required=True)
    p.add_argument("--corpus", default="runs/m0/corpus.jsonl")
    p.add_argument("--vocab", default="runs/m0/vocab.pkl.gz")
    p.add_argument("--out", required=True)
    p.add_argument("--base-url", required=True, help="Embedding 服务的 OpenAI 兼容地址")
    p.add_argument("--model", required=True)
    p.add_argument("--api-key-env", default=None, help="存放 API 密钥的环境变量名；本地服务可以不填")
    p.add_argument("--dim", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=10, help="每批入库的文档数，DashScope 最多 10")
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--milvus-uri", default="http://localhost:19530")
    p.add_argument("--skip-ingest", action="store_true", help="集合已经建好时只跑检索")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("verify", help="逐行比对两个集合，并比对新旧代码的查询编码（需要旧包在 PYTHONPATH 上）")
    p.add_argument("--vocab", default="runs/m0/vocab.pkl.gz")
    p.add_argument("--base-url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--api-key-env", default=None)
    p.add_argument("--dim", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=10)
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--milvus-uri", default="http://localhost:19530")
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("compare", help="软门槛：检索指标的差异是否在噪声范围内")
    p.add_argument("old")
    p.add_argument("new")
    p.add_argument("--rerun", default=None, help="新实现重跑一次的结果，用来量出噪声；不传时只输出指标")
    p.set_defaults(func=cmd_compare)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
