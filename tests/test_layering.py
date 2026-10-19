"""分层约束（docs/architecture.md 第 1 节第 3 条）：核心层不依赖 LangChain / LangGraph

允许依赖的位置：apps/（应用层）、llm/langchain_adapter.py（给应用层构造模型）、eval/scorers/（封装 ragas）。
"""
import ast
from pathlib import Path

import medrag

PKG = Path(medrag.__file__).parent
ALLOWED = ("apps/", "llm/langchain_adapter.py", "eval/scorers/")
FORBIDDEN = ("langchain", "langgraph", "langchain_core", "langchain_openai", "langchain_community")


def _imports(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from (a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            yield node.module


def test_core_does_not_import_langchain():
    offenders = []
    for path in PKG.rglob("*.py"):
        rel = path.relative_to(PKG).as_posix()
        if rel.startswith(ALLOWED):
            continue
        for mod in _imports(path):
            if mod.split(".")[0] in FORBIDDEN:
                offenders.append(f"{rel}: {mod}")
    assert offenders == []
