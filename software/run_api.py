import os
import sys

import uvicorn

from MedicalRag.config.loader import ConfigLoader
from MedicalRag.core.services import START_HINT, check_services


def preflight() -> None:
    """启动前确认基础服务可用，避免服务起来之后才在请求中报错"""
    failed = [r for r in check_services(ConfigLoader().config) if not r.ok]
    if failed:
        for r in failed:
            print(f"[preflight] {r.name} 不可用: {r.detail}", file=sys.stderr)
        sys.exit(f"[preflight] {START_HINT}")


if __name__ == "__main__":
    # API keys must be set in the environment before starting.
    # Example: export DASHSCOPE_API_KEY=sk-...
    preflight()
    uvicorn.run(
        "MedicalRag.api.app:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8000")),
        workers=1,       # MUST be 1: session state lives in process memory
        log_level="info",
        reload=False,
    )
