"""配置分离验证：不需要模型，也不需要 Milvus / Redis。

验证点：
1. 拆分后加载出的 AppConfig 与拆分前的 app_config.yaml 完全相等；
2. overrides 只整段替换指定的顶层 key，其他段不受影响；
3. 同一目录下两个文件包含同一个顶层 key 时报错；
4. 记录了每个顶层 key 来自哪个文件。
"""
import shutil
import subprocess
import tempfile
from pathlib import Path

import yaml

from common import CONF_DIR  # noqa: F401  先导入以清理代理

from MedicalRag.config.loader import ConfigLoader
from MedicalRag.config.models import AppConfig

PKG_CONF = Path(__file__).resolve().parents[1] / "src" / "MedicalRag" / "config"


def check_equal_to_old():
    """拆分前后 AppConfig 是否一致（从 git 里取回被删除的 app_config.yaml）"""
    proc = subprocess.run(
        ["git", "show", "f7fc50d:src/MedicalRag/config/app_config.yaml"],
        capture_output=True, text=True, cwd=PKG_CONF,
    )
    if proc.returncode != 0:
        print("1) 跳过：取不到拆分前的 app_config.yaml")
        return

    old = AppConfig(**yaml.safe_load(proc.stdout))
    new = ConfigLoader(str(PKG_CONF)).config
    d_old, d_new = old.model_dump(), new.model_dump()
    diff = {k: (d_old.get(k), v) for k, v in d_new.items() if d_old.get(k) != v}
    # 预期只在这次新增的字段上有差异
    print(f"1) 与拆分前完全相等（忽略新增字段）: {set(diff) <= {'agent', 'redis'}}")
    for key, (before, after) in diff.items():
        changed = {k: (before.get(k), v) for k, v in after.items() if before.get(k) != v}
        print(f"   {key} 段的差异: {changed}")


def check_overrides():
    """overrides 整段替换"""
    tmp = Path(tempfile.mkdtemp())
    (tmp / "models_ollama.yaml").write_text(
        yaml.safe_dump({
            "llm": {"provider": "ollama", "model": "qwen3:32b", "temperature": 0.5},
            "embedding": {
                "summary_dense": {"provider": "ollama", "model": "bge-m3:latest", "dimension": 1024},
                "text_dense": {"provider": "ollama", "model": "bge-m3:latest", "dimension": 1024},
                "text_sparse": {"provider": "Milvus"},
            },
        }, allow_unicode=True), encoding="utf-8")

    base = ConfigLoader(CONF_DIR).config
    ov = ConfigLoader(CONF_DIR, overrides=[str(tmp / "models_ollama.yaml")]).config
    print(f"2) llm 段被替换: {ov.llm.provider}/{ov.llm.model}（原 {base.llm.provider}/{base.llm.model}）")
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
    check_equal_to_old()
    check_overrides()
    check_duplicate_key()
    show_sources()
