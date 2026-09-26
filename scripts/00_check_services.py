"""
检查基础服务（Milvus / PostgreSQL / Redis / Neo4j）是否可用。

先执行 deploy/start.sh，再运行本脚本；全部通过时以 0 退出。
    python scripts/00_check_services.py
"""
import sys

from MedicalRag.config.loader import ConfigLoader
from MedicalRag.core.services import START_HINT, check_services


def main() -> int:
    config = ConfigLoader().config
    results = check_services(config)
    for r in results:
        mark = "✔" if r.ok else "✘"
        print(f"  {mark} {r.name:<9} {r.latency_ms:7.1f} ms  {r.detail}")

    failed = [r.name for r in results if not r.ok]
    if failed:
        print(f"\n不可用: {', '.join(failed)}。{START_HINT}")
        return 1
    print("\n全部服务可用")
    return 0


if __name__ == "__main__":
    sys.exit(main())
