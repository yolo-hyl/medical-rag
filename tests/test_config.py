"""配置：包内默认配置能加载、各段可选、覆盖文件整段替换"""
import pytest

from medrag.config import AppConfig, ConfigLoader


def test_package_defaults_load():
    cfg = ConfigLoader().config
    assert cfg.llm is not None and cfg.llm.api_key_env == "DASHSCOPE_API_KEY"
    assert cfg.embedding.text_dense.max_batch == 10
    assert cfg.prompts.lang == "zh"


def test_sections_are_optional():
    cfg = AppConfig()
    assert cfg.llm is None and cfg.embedding is None
    assert cfg.milvus.collection_name == "medical_knowledge"


def test_override_replaces_whole_section(tmp_path):
    override = tmp_path / "models_local.yaml"
    override.write_text("llm:\n  model: Qwen3-8B\n  base_url: http://127.0.0.1:8001/v1\n", encoding="utf-8")
    cfg = ConfigLoader(overrides=[str(override)]).config
    assert cfg.llm.model == "Qwen3-8B" and cfg.llm.api_key_env is None
    assert cfg.llm.api_key() == "EMPTY"   # 本地服务不校验密钥


def test_duplicate_top_level_key_is_rejected(tmp_path):
    (tmp_path / "a.yaml").write_text("llm: {model: a}\n", encoding="utf-8")
    (tmp_path / "b.yaml").write_text("llm: {model: b}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="重复定义"):
        ConfigLoader(str(tmp_path))
