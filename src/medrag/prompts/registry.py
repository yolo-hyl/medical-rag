"""版本化提示词注册表（设计见 docs/architecture.md 第 11 节）

    templates/<name>/_meta.yaml        说明、变量定义、别名（stable / latest）
    templates/<name>/<version>.<lang>.yaml   一个版本一个文件

    registry = PromptRegistry.from_config(cfg.prompts)
    prompt = registry.get("basic_rag")          # 版本由配置决定，缺省用 stable 别名
    messages = prompt.render(all_document_str=docs, input=q)
    prompt.ref                                  # name@version.lang#sha，写进 trace 和评测记录

已发布（released）的版本不可修改：内容哈希登记在 prompts.lock，`check_lock()` 在单测里校验。
"""
from __future__ import annotations

import hashlib
import json
import string
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional

import yaml
from jinja2 import StrictUndefined
from jinja2.sandbox import SandboxedEnvironment

from ..config.models import PromptsConfig

BUILTIN_DIR = Path(__file__).parent / "templates"
LOCK_FILE = Path(__file__).parent / "prompts.lock"

_JINJA = SandboxedEnvironment(undefined=StrictUndefined, keep_trailing_newline=True, autoescape=False)


@dataclass(frozen=True)
class PromptRef:
    name: str
    version: str
    lang: str
    sha256: str

    def __str__(self) -> str:
        return f"{self.name}@{self.version}.{self.lang}#{self.sha256[:12]}"


@dataclass(frozen=True)
class PromptVersion:
    """一个提示词的一个版本（一个 YAML 文件）"""
    name: str
    version: str
    lang: str
    status: Literal["draft", "released", "deprecated"]
    format: Literal["fstring", "jinja2"]
    messages: tuple            # ((role, content), ...)
    params: Dict[str, Any]
    variables: Dict[str, dict]  # 来自 _meta.yaml
    parent: Optional[str] = None
    changelog: str = ""
    path: Optional[Path] = None

    @property
    def sha256(self) -> str:
        """只对影响模型输入的部分取哈希：语法、参数、消息；作者、说明之类的改动不算改版本"""
        payload = {"format": self.format, "params": self.params, "messages": [list(m) for m in self.messages]}
        return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()

    @property
    def ref(self) -> PromptRef:
        return PromptRef(self.name, self.version, self.lang, self.sha256)

    def template(self, role: str) -> str:
        """取某个角色的原始模板文本。

        给需要自己组装消息的调用方用（例如 apps/ 里用 LangChain 的 ChatPromptTemplate
        插入 MessagesPlaceholder）；fstring 语法的模板可以直接交给 ChatPromptTemplate。
        """
        for r, content in self.messages:
            if r == role:
                return content
        raise KeyError(f"提示词 {self.name}@{self.version} 没有 {role} 消息")

    def render(self, **variables: Any) -> List[Dict[str, str]]:
        """渲染成 OpenAI 格式的消息列表；缺少必填变量或传了未声明的变量都会报错"""
        declared = set(self.variables)
        unknown = set(variables) - declared
        if unknown:
            raise ValueError(f"提示词 {self.name} 未声明变量: {sorted(unknown)}")
        missing = {k for k, v in self.variables.items() if v.get("required", True)} - set(variables)
        if missing:
            raise ValueError(f"提示词 {self.name} 缺少变量: {sorted(missing)}")
        return [{"role": role, "content": self._render_text(content, variables)} for role, content in self.messages]

    def _render_text(self, content: str, variables: Dict[str, Any]) -> str:
        if self.format == "jinja2":
            return _JINJA.from_string(content).render(**variables)
        # fstring：只填该段出现过的字段，和 v2 的 str.format 行为一致
        fields = {f for _, f, _, _ in string.Formatter().parse(content) if f}
        return content.format(**{k: v for k, v in variables.items() if k in fields})


class PromptRegistry:
    def __init__(
        self,
        dirs: Iterable[Path | str] = (),
        versions: Optional[Dict[str, str]] = None,
        lang: str = "zh",
        include_builtin: bool = True,
    ):
        """dirs 中的目录优先级高于内置模板：同名提示词整体以先出现的目录为准"""
        self.dirs = [Path(d) for d in dirs] + ([BUILTIN_DIR] if include_builtin else [])
        self.versions = dict(versions or {})
        self.lang = lang
        self._cache: Dict[tuple, PromptVersion] = {}

    @classmethod
    def from_config(cls, config: PromptsConfig) -> "PromptRegistry":
        return cls(dirs=config.dirs, versions=config.versions, lang=config.lang)

    @classmethod
    @lru_cache(maxsize=1)
    def builtin(cls) -> "PromptRegistry":
        """只含内置模板、全部使用 stable 版本的共享实例；调用方没有注入注册表时使用"""
        return cls()

    # ---------- 查找 ----------
    def _prompt_dir(self, name: str) -> Path:
        for d in self.dirs:
            if (d / name / "_meta.yaml").exists():
                return d / name
        raise KeyError(f"找不到提示词 {name}（搜索目录：{[str(d) for d in self.dirs]}）")

    @staticmethod
    def _read(path: Path) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def meta(self, name: str) -> dict:
        return self._read(self._prompt_dir(name) / "_meta.yaml")

    def names(self) -> List[str]:
        found: List[str] = []
        for d in self.dirs:
            if d.is_dir():
                found += [p.parent.name for p in sorted(d.glob("*/_meta.yaml")) if p.parent.name not in found]
        return found

    def resolve_version(self, name: str, version: Optional[str] = None) -> str:
        """显式版本 > 配置中的版本 > stable 别名；别名在 _meta.yaml 的 aliases 里定义"""
        wanted = version or self.versions.get(name) or "stable"
        return self.meta(name).get("aliases", {}).get(wanted, wanted)

    def get(self, name: str, version: Optional[str] = None, lang: Optional[str] = None) -> PromptVersion:
        lang = lang or self.lang
        ver = self.resolve_version(name, version)
        key = (name, ver, lang)
        if key not in self._cache:
            path = self._prompt_dir(name) / f"{ver}.{lang}.yaml"
            if not path.exists():
                # 中英文严格区分：找不到对应语言就报错，不退回另一种语言
                raise KeyError(f"提示词 {name}@{ver} 没有 {lang} 版本：{path}")
            self._cache[key] = self._load(name, path)
        return self._cache[key]

    def _load(self, name: str, path: Path) -> PromptVersion:
        data = self._read(path)
        version, lang = path.name[: -len(".yaml")].rsplit(".", 1)
        if data.get("version", version) != version:
            raise ValueError(f"{path} 中的 version 与文件名不一致")
        return PromptVersion(
            name=name,
            version=version,
            lang=lang,
            status=data.get("status", "draft"),
            format=data.get("format", "jinja2"),
            messages=tuple((m["role"], m["content"]) for m in data["messages"]),
            params=data.get("params") or {},
            variables=self.meta(name).get("variables") or {},
            parent=data.get("parent"),
            changelog=data.get("changelog", ""),
            path=path,
        )

    def all_versions(self) -> List[PromptVersion]:
        out = []
        for name in self.names():
            for path in sorted(self._prompt_dir(name).glob("v*.*.yaml")):
                out.append(self._load(name, path))
        return out

    # ---------- 锁定已发布版本 ----------
    @staticmethod
    def _lock_key(p: PromptVersion) -> str:
        return f"{p.name}@{p.version}.{p.lang}"

    def check_lock(self, lock_file: Path = LOCK_FILE) -> List[str]:
        """返回问题列表；为空表示所有已发布版本都和 lock 一致"""
        lock = self._read(lock_file) if lock_file.exists() else {}
        problems = []
        for p in self.all_versions():
            if p.status != "released":
                continue
            key = self._lock_key(p)
            if key not in lock:
                problems.append(f"{key} 已发布但未登记到 {lock_file.name}，请执行 python -m medrag.prompts lock")
            elif lock[key] != p.sha256:
                problems.append(f"{key} 已发布后被修改（不允许）：请恢复原文，改动放到新版本里")
        return problems

    def write_lock(self, lock_file: Path = LOCK_FILE) -> Dict[str, str]:
        """把已发布版本的哈希登记进 lock；已登记的条目不会被覆盖"""
        lock = self._read(lock_file) if lock_file.exists() else {}
        for p in self.all_versions():
            if p.status == "released":
                lock.setdefault(self._lock_key(p), p.sha256)
        with open(lock_file, "w", encoding="utf-8") as f:
            f.write("# 已发布提示词的内容哈希，由 `python -m medrag.prompts lock` 维护，不要手改\n")
            yaml.safe_dump(dict(sorted(lock.items())), f, allow_unicode=True, sort_keys=False)
        return lock
