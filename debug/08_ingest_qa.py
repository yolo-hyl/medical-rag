"""把 data/qa_50000.jsonl 入库到独立集合，供 07_web.py 用真实数据体验。

用法：
    python debug/08_ingest_qa.py --limit 1000   # 先入一小部分，看效果和速度
    python debug/08_ingest_qa.py                # 全量
然后：
    python debug/07_web.py      # 集合已存在时 07 直接用，不写样例也不删

集合名默认 medrag_debug。注意 03~06 跑完会删掉 medrag_debug，想保留就用 --collection 换个名字。
集合不存在时新建；已存在时看 storage.yaml 的 drop_old：true 删掉重建，false 在原集合上追加
（主键是问题的哈希，重复的问题会覆盖）。
"""
import argparse
import asyncio
import json
import time
from pathlib import Path

from common import build_kb, load_config

from MedicalRag.core.IngestionPipeline import IngestionPipeline

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "qa_50000.jsonl"
DEFAULT_COLLECTION = "medrag_debug"


def read_records(limit: int | None) -> list[dict]:
    with DATA_FILE.open(encoding="utf-8") as f:
        first = f.readline()
        if first.startswith("version https://git-lfs"):
            raise SystemExit(f"{DATA_FILE} 还是 Git LFS 指针，先执行: git lfs pull --include data/qa_50000.jsonl")
        records = [json.loads(first)]
        for line in f:
            if limit and len(records) >= limit:
                break
            if line.strip():
                records.append(json.loads(line))
    return records


async def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=None, help="只入库前 N 条，默认全量")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION, help=f"集合名，默认 {DEFAULT_COLLECTION}")
    parser.add_argument("--batch-size", type=int, default=None, help="每批入库的文档数，默认取 data.yaml")
    parser.add_argument("--concurrency", type=int, default=None, help="同时进行的批次数，默认取 data.yaml")
    args = parser.parse_args()

    cfg = load_config()
    data_update = {k: v for k, v in (("batch_size", args.batch_size), ("concurrency", args.concurrency)) if v}
    cfg = cfg.model_copy(update={
        "milvus": cfg.milvus.model_copy(update={"collection_name": args.collection}),
        "data": cfg.data.model_copy(update=data_update),
    })
    records = read_records(args.limit)
    print(f"读取 {len(records)} 条 → 集合 {args.collection}（{cfg.embedding.summary_dense.model}，"
          f"batch_size={cfg.data.batch_size}，concurrency={cfg.data.concurrency}）")

    kb = build_kb(cfg)
    t0 = time.time()
    try:
        ok = await IngestionPipeline(cfg.data, kb).run(records)
        cost = time.time() - t0
        if not ok:
            raise SystemExit("入库失败，看上面的日志")
        rows = kb.client.query(args.collection, output_fields=["count(*)"])[0]["count(*)"]
        print(f"入库完成：{rows} 条，用时 {cost:.0f}s（{len(records) / max(cost, 1e-6):.1f} 条/s）")
        if args.limit:
            print(f"按这个速度粗估（含建索引等固定开销，条数少时偏高）全量 50000 条约需 {50000 / len(records) * cost / 60:.0f} 分钟")
    finally:
        await kb.close()


if __name__ == "__main__":
    asyncio.run(main())
