"""
配置加载器

配置按领域拆成多个 YAML 文件（storage / models / data / dialogue / agent），
加载时把各文件的顶层 key 合并成一份完整配置：

    ConfigLoader()                                   # 读取包内 config/ 目录下所有 *.yaml
    ConfigLoader("my_conf/")                         # 读取自己的配置目录
    ConfigLoader(overrides=["exp/models_qwen.yaml"]) # 默认配置 + 整段替换模型配置
"""
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

import yaml
from dotenv import load_dotenv

from .models import AppConfig

logger = logging.getLogger(__name__)

# 仓库根目录：src/MedicalRag/config/loader.py -> 上溯三级
REPO_ROOT = Path(__file__).resolve().parents[3]


def load_deploy_env() -> Optional[Path]:
    """
    加载 deploy/start.sh 生成的 .env（中间件密码等），已存在的环境变量优先。
    可通过环境变量 MEDRAG_ENV_FILE 指定其他路径。
    """
    env_file = Path(os.environ.get("MEDRAG_ENV_FILE", REPO_ROOT / "deploy" / ".env"))
    if env_file.exists():
        load_dotenv(env_file, override=False)
        return env_file
    logger.debug(f"未找到 {env_file}，跳过加载")
    return None


class ConfigLoader:
    """配置加载器"""

    _INDEX_PATTERN = re.compile(r"(.*?)\[(\d+)\]$")  # 用于解析 a.b[0].c

    def __init__(
        self,
        config_dir: Optional[str] = None,
        overrides: Iterable[str] = (),
    ):
        """
        Args:
            config_dir: 配置目录。默认取环境变量 MEDRAG_CONFIG_DIR，再退回包内 config/ 目录
            overrides: 额外的配置文件，按顺序整段覆盖同名的顶层 key
        """
        load_deploy_env()

        if config_dir is None:
            config_dir = os.environ.get("MEDRAG_CONFIG_DIR") or str(Path(__file__).parent)
        self.config_dir = Path(config_dir)
        if not self.config_dir.is_dir():
            raise FileNotFoundError(f"配置目录不存在: {self.config_dir}")

        self._dict: Dict[str, Any] = {}
        self._sources: Dict[str, Path] = {}  # 顶层 key -> 它来自哪个文件

        # 1) 目录内的所有 yaml：同一个顶层 key 只允许出现一次，避免两个文件悄悄互相覆盖
        for path in sorted(self.config_dir.glob("*.yaml")):
            for key, value in self._read_yaml(path).items():
                if key in self._sources:
                    raise ValueError(
                        f"配置项 '{key}' 重复定义：{self._sources[key].name} 与 {path.name}"
                    )
                self._dict[key] = value
                self._sources[key] = path

        # 2) overrides：整段替换同名顶层 key
        for override in overrides:
            path = Path(override)
            if not path.exists():
                raise FileNotFoundError(f"覆盖配置文件不存在: {path}")
            for key, value in self._read_yaml(path).items():
                self._dict[key] = value
                self._sources[key] = path

        if not self._dict:
            raise FileNotFoundError(f"配置目录 {self.config_dir} 下没有可用的 *.yaml")

        self._app_config = AppConfig(**self._dict)  # Pydantic 校验

    @staticmethod
    def _read_yaml(path: Path) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    @property
    def config(self) -> AppConfig:
        return self._app_config

    @property
    def as_dict(self) -> dict:
        """返回当前配置的 dict 形式（深拷贝）"""
        return self._app_config.model_dump()

    @property
    def sources(self) -> Dict[str, Path]:
        """每个顶层 key 来自哪个文件"""
        return dict(self._sources)

    # -------------------------------------------------------------------------
    # 公共方法：change
    # -------------------------------------------------------------------------
    def change(
        self,
        updates: Union[dict, List[tuple[str, Any]]],
        save: bool = False,
        save_path: str = "",
    ) -> AppConfig:
        """
        任意快捷更改配置的任意字段。
        支持两种更新形式：
          1) 嵌套 dict：{"llm": {"model": "qwen3:72b"}}
          2) 点路径：{"embedding.text_dense.model": "bge-m3:latest"}
             也支持列表下标： "foo.bar[0].baz": 123

        Args:
            updates: 变更内容
            save: 是否立即写回 YAML 文件（只写被修改的那几段，各自回到自己的来源文件）
            save_path: 写入目录，留空则写回原来的来源文件
        Returns:
            更新并校验后的 AppConfig
        """
        # 把点路径更新转成嵌套 dict
        if isinstance(updates, dict):
            upd_dict = self._expand_dot_paths(updates)
        else:
            # 支持传入 [("a.b", 1), ...]
            upd_dict = self._expand_dot_paths(dict(updates))

        # 深合并到现有 dict
        merged = self._deep_merge(self._dict, upd_dict)

        # 用 Pydantic 校验
        new_config = AppConfig(**merged)

        self._dict = merged
        self._app_config = new_config
        if save:
            self._save_yaml(changed_keys=list(upd_dict.keys()), save_path=save_path)

        return self._app_config

    # -------------------------------------------------------------------------
    # 工具方法
    # -------------------------------------------------------------------------
    def _expand_dot_paths(self, flat: dict) -> dict:
        """
        将 {"a.b[0].c": 1, "x.y": 2} 展开为嵌套 dict
        """
        root: dict = {}
        for key, value in flat.items():
            parts = key.split(".")
            cur = root
            for i, part in enumerate(parts):
                m = self._INDEX_PATTERN.match(part)
                if m:
                    # 处理带下标的部分，如 "items[0]"
                    name, idx = m.group(1), int(m.group(2))
                    if name not in cur or not isinstance(cur.get(name), list):
                        cur[name] = []
                    lst = cur[name]
                    # 确保列表长度足够
                    while len(lst) <= idx:
                        lst.append({})
                    if i == len(parts) - 1:
                        lst[idx] = value
                    else:
                        if not isinstance(lst[idx], dict):
                            lst[idx] = {}
                        cur = lst[idx]
                else:
                    if i == len(parts) - 1:
                        cur[part] = value
                    else:
                        if part not in cur or not isinstance(cur[part], dict):
                            cur[part] = {}
                        cur = cur[part]
        return root

    def _deep_merge(self, base: Any, patch: Any) -> Any:
        """
        递归合并：dict 深合并；list 位置覆盖；其余类型直接替换。
        """
        if isinstance(base, dict) and isinstance(patch, dict):
            out = dict(base)
            for k, v in patch.items():
                if k in out:
                    out[k] = self._deep_merge(out[k], v)
                else:
                    out[k] = v
            return out
        elif isinstance(base, list) and isinstance(patch, list):
            # 列表按索引覆盖：patch 的长度优先生效
            out = list(base)
            for i, v in enumerate(patch):
                if i < len(out):
                    out[i] = self._deep_merge(out[i], v)
                else:
                    out.append(v)
            return out
        else:
            return patch

    def _save_yaml(self, changed_keys: List[str], save_path: str = ""):
        """把被修改的顶层 key 写回各自的来源文件；save_path 指定时写到该目录下的同名文件。"""
        out_dir = Path(save_path) if save_path else None
        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)

        # 按来源文件分组，一个文件只写一次
        by_file: Dict[Path, List[str]] = {}
        for key in changed_keys:
            source = self._sources.get(key)
            if source is None:
                raise KeyError(f"配置项 '{key}' 没有来源文件，无法写回")
            by_file.setdefault(source, []).append(key)

        for source, keys in by_file.items():
            # 保留该文件原有的其他顶层 key，只更新变化的部分
            content = self._read_yaml(source) if source.exists() else {}
            for key in keys:
                content[key] = self._dict[key]
            target = out_dir / source.name if out_dir is not None else source
            with open(target, "w", encoding="utf-8") as f:
                yaml.safe_dump(content, f, allow_unicode=True, sort_keys=False)
            logger.info(f"配置 {keys} 已写入 {target}")
