"""scripts/ 的公共配置入口

示例脚本统一读取 scripts/conf/ 下的配置（与包内默认配置分开，改这里不会影响库本身）：

    from _common import config_loader, load_config
    cfg = load_config()

- 想换一套配置目录：设置环境变量 MEDRAG_CONFIG_DIR。
- 只想覆盖其中几段（例如连接 deploy/ 启动的服务）：设置 MEDRAG_CONFIG_OVERRIDES=deploy/storage.yaml。
- 脚本要在仓库根目录运行（python scripts/04_basic_rag.py）：数据路径 data/... 和降级实现的 .medrag/ 都相对当前目录。
"""
import os
from pathlib import Path
from typing import Iterable

from medrag.config import AppConfig, ConfigLoader

CONF_DIR = Path(__file__).resolve().parent / "conf"


def config_loader(overrides: Iterable[str] = ()) -> ConfigLoader:
    return ConfigLoader(os.environ.get("MEDRAG_CONFIG_DIR") or str(CONF_DIR), overrides=overrides)


def load_config(overrides: Iterable[str] = ()) -> AppConfig:
    return config_loader(overrides).config
