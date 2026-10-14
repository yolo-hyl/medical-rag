"""配置分离验证：不需要模型，也不需要 Milvus / Redis。

验证点：
1. 包内默认配置与 debug/conf 都能加载，缺省的段使用默认值；
2. overrides 只整段替换指定的顶层 key，其他段不受影响；
3. 同一目录下两个文件包含同一个顶层 key 时报错；
4. 记录了每个顶层 key 来自哪个文件。
"""
import shutil
import tempfile
from pathlib import Path

import yaml

from common import CONF_DIR  # noqa: F401  先导入以清理代理

from medrag.config import ConfigLoader

PKG_CONF = Path(__file__).resolve().parents[1] / "src" / "medrag" / "config"


def check_loadable():
    """包内默认配置、debug/conf 都能加载（v2 拆分前的 app_config.yaml 对比在 M0 改配置字段后已不适用）"""
    pkg = ConfigLoader(str(PKG_CONF)).config
    dbg = ConfigLoader(CONF_DIR).config
    print(f"1) 包内默认: llm={pkg.llm.model} embedding={pkg.embedding.text_dense.model} 提示词语言={pkg.prompts.lang}")
    print(f"   debug/conf: llm={dbg.llm.model}@{dbg.llm.base_url} 集合={dbg.milvus.collection_name}"
          f"（debug/conf 没有 prompts.yaml，使用默认: {dbg.prompts.versions or 'stable'}）")


def check_overrides():
    """overrides 整段替换"""
    tmp = Path(tempfile.mkdtemp())
    (tmp / "models_ollama.yaml").write_text(
        yaml.safe_dump({
            # Ollama 也走 OpenAI 兼容的 /v1 端点（D2），不需要密钥
            "llm": {"model": "qwen3:32b", "base_url": "http://localhost:11434/v1", "temperature": 0.5},
            "embedding": {
                "summary_dense": {"model": "bge-m3:latest", "base_url": "http://localhost:11434/v1", "dimension": 1024},
                "text_dense": {"model": "bge-m3:latest", "base_url": "http://localhost:11434/v1", "dimension": 1024},
                "text_sparse": {"provider": "Milvus"},
            },
        }, allow_unicode=True), encoding="utf-8")

    base = ConfigLoader(CONF_DIR).config
    ov = ConfigLoader(CONF_DIR, overrides=[str(tmp / "models_ollama.yaml")]).config
    print(f"2) llm 段被替换: {ov.llm.model}@{ov.llm.base_url}（原 {base.llm.model}@{base.llm.base_url}）")
    print(f"   text_sparse 是整段替换（vocab 回到默认值）: {ov.embedding.text_sparse.vocab_path_or_name}")
    print(f"   其他段不变: milvus={ov.milvus == base.milvus} data={ov.data == base.data} agent={ov.agent == base.agent}")
    shutil.rmtree(tmp)


def check_duplicate_key():
    """同一目录里重复的顶层 key 必须报错"""
    tmp = Path(tempfile.mkdtemp())
    for name in ("a.yaml", "b.yaml"):
        shutil.copy(PKG_CONF / "storage.yaml", tmp / name)
    try:
        ConfigLoader(str(tmp))
        print("3) 重复 key 没有报错（不符合预期）")
    except ValueError as e:
        print(f"3) 重复 key 报错: {e}")
    shutil.rmtree(tmp)


def show_sources():
    loader = ConfigLoader(CONF_DIR)
    print("4) 每个顶层 key 的来源:")
    for key, path in sorted(loader.sources.items()):
        print(f"   {key:20s} <- {path.name}")


if __name__ == "__main__":
    check_loadable()
    check_overrides()
    check_duplicate_key()
    show_sources()
