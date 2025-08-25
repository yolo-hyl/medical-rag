"""
配置加载器
"""
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging
from .models import AppConfig
from typing import Dict, List, Optional, Literal, Any, Union
from pydantic import BaseModel, Field
import re

logger = logging.getLogger(__name__)

class ConfigLoader:
    """配置加载器"""

    _INDEX_PATTERN = re.compile(r"(.*?)\[(\d+)\]$")  # 用于解析 a.b[0].c

    def __init__(self, config_path: Optional[str] = None):
        """
        Args:
            config_path: 配置文件路径（不是目录）。默认使用当前文件同目录下的 app_config.yaml
        """
        if config_path is None:
            config_root = Path(__file__).parent
            self.config_path = config_root / "app_config.yaml"
        else:
            self.config_path = Path(config_path)

        if not self.config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")

        with open(self.config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        self._dict = raw
        self._app_config = AppConfig(**raw)  # Pydantic 校验

    @property
    def config(self) -> AppConfig:
        return self._app_config

    @property
    def as_dict(self) -> dict:
        """返回当前配置的 dict 形式（深拷贝）"""
        # pydantic v1
        return self._app_config.model_dump()

    # -------------------------------------------------------------------------
    # 公共方法：change
    # -------------------------------------------------------------------------
    def change(
        self, 
        updates: Union[dict, List[tuple[str, Any]]], 
        save: bool = False,
        save_path: str = ""
    ) -> AppConfig:
        """
        任意快捷更改配置的任意字段。
        支持两种更新形式：
          1) 嵌套 dict：{"llm": {"model": "qwen3:72b"}}
          2) 点路径：{"embedding.text_dense.model": "bge-m3:latest"}
             也支持列表下标： "foo.bar[0].baz": 123

        Args:
            updates: 变更内容
            save: 是否立即写回 YAML 文件
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

        # 持久化
        self._dict = merged
        self._app_config = new_config
        if save:
            self._save_yaml(save_path)

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

    def _save_yaml(self, save_path):
        with open(save_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self._dict, f, allow_unicode=True, sort_keys=False)