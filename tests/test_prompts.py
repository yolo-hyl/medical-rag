"""提示词注册表：v1 与 v2 原文逐字一致、已发布版本锁定、渲染校验、版本与语言解析"""
import hashlib

import pytest

from medrag.config import PromptsConfig
from medrag.prompts import PromptRegistry

# v2（cff9b8e）prompts/templates.py 中各模板原文的 sha256，由基线代码直接计算得到
V2_TEMPLATE_SHA256 = {
    "basic_rag": ("cccd8ffaf2fa8177a52f2f27ded1679383e721d9f75b9c0ec2d4085e305d76a5", "f662f30461ca3ccde3bd6a35e835f6b98c7ffcec41e14ab372fa88fcf1bb542d"),
    "dialogue_rag": ("a494408cfcba0d136385693edc29a9c6a9fffad248d264ce2998a91ef57b8c57", "3ec5f9af8ea7f92c5e135dce8b977bec247e2e0a4c36bd68425d3e37e443d577"),
    "agent_answer": ("c16de64fbf5edd828bd7a2205db26dfae33f08ea58529d61bc42a10567f13d48", "2e865e96d2aa31e0c79fb0a21c7d25b3d7326be6eb1213d349034053faa18bdd"),
    "rewriter": ("6b3cb3d6b8ab74cb2ad123cce3a35fd11404ae5bf831a450ba1983000c6df65a", "a4bb36cb315f8716eecb32153986095d92a811595e08fabf4feae7536aecdb93"),
    "summary": ("020ccb15809e02bc960ba6867e1cc09f1adc5c7080779d6d480621b37e87d95a", "c8bccef2a5122be509630f84196e1333c1013e2df1a1fb4cea00455a946cc781"),
    "call_db": ("593510ff79ca316ee2fcb6a2bb1f63632568d8af2904b816d2fd69c876834623", "c57cb581e2f00068c8f2894195b9d8350fc8a67e4bf8d95d71c73b0a38630264"),
    "web_router": ("165deab76d7ca00d2cb81a969f59037d39347472db9c10d740775be62455242c", "6855de0bbc52487e68c2db133f0cc362f9d9668aa2e764728dfb7edb9d1a8230"),
    "call_web": ("3eda968cc044ef7d9e6580af96117c386694db66ea41a79cc21e0cc94aba0257", "c5a26d637c07af1081f38e3658d2ef16a13894b7bb1593f6f84deb0cbb344c2c"),
    "judge_rag": ("036ba955f0210a2bf8b03c0ded9efb8324c9d08f1c76fa963f2437665fd6fdc3", "7a627d9879f17f0397fa4e5156d1ff8dbd00da53fa350b3e67abe4240c3588b0"),
    "ask_user": ("843d291a54f8fe0257b57600d0d974bde38599b4e384bc69b5080bce6563e605", "b272de35bad51c3d60d402f9e967a691648e2a7f3a5a2ea1a931f171ab099675"),
    "follow_ask": ("6c9f981019b790cf09fd6ae8e7c16bbdf0e4341a25aaca3414354f346cadf19e", "f03417a45106690064476e428161d6030c7e7ee3534563a9aeacb35aafda1b92"),
    "extract_user_info": ("69042294843b64c76f1c7a0ab027d43dc9a8a0748ad8c3eaaafc0c8aa3274a4e", "a1c6a6f427da9fbab1cde5d522d526cfe76358ea25a1063388f99db0045f59ed"),
    "handle_query": ("554d414d9ce2bd38da4a7463b5bdafff557ee48992d255a220f6c578c813583f", "e883687c899ee4863dfed8c77e2894676f7812cc27a8c03224643ba3d9fc63de"),
    "update_background": ("bb63fe763cce0c69515c9799409e1185d74002762d7fc3f3235023b5481593af", "8e8e0add25cec0c938c3533491e0805db47b6734e572cc270ca97824d4a48447"),
    "medical_qa_annotation": ("dfce5b6166bd1d83cd6100d731b8bfd47b4ea2810c66f3ebc7a6f5d856b967e1", "61d65c4dfe38d3885696afd989c78bc3cb4c905793a7f18f1a6280ab5b5a075b"),
}


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@pytest.mark.parametrize("name", sorted(V2_TEMPLATE_SHA256))
def test_v1_matches_v2_verbatim(name):
    prompt = PromptRegistry.builtin().get(name, "v1")
    system_sha, user_sha = V2_TEMPLATE_SHA256[name]
    assert _sha(prompt.template("system")) == system_sha
    assert _sha(prompt.template("user")) == user_sha


def test_released_versions_are_locked():
    assert PromptRegistry.builtin().check_lock() == []


def test_render_fstring_matches_str_format():
    prompt = PromptRegistry.builtin().get("basic_rag")
    messages = prompt.render(all_document_str="## 文档1：\n资料", input="头痛怎么办")
    assert messages[0] == {"role": "system", "content": prompt.template("system")}
    assert messages[1]["content"] == prompt.template("user").format(all_document_str="## 文档1：\n资料", input="头痛怎么办")


def test_render_validates_variables():
    prompt = PromptRegistry.builtin().get("basic_rag")
    with pytest.raises(ValueError, match="缺少变量"):
        prompt.render(input="头痛怎么办")
    with pytest.raises(ValueError, match="未声明变量"):
        prompt.render(all_document_str="", input="", extra="x")


def _write_prompt(root, name, files: dict, aliases=None):
    d = root / name
    d.mkdir(parents=True)
    (d / "_meta.yaml").write_text(
        f"name: {name}\nvariables:\n  question: {{required: true}}\n  contexts: {{required: false}}\n"
        f"aliases: {aliases or {'stable': 'v1'}}\n", encoding="utf-8")
    for fname, text in files.items():
        (d / fname).write_text(text, encoding="utf-8")


def test_custom_dir_overrides_builtin_and_versions_from_config(tmp_path):
    jinja_v2 = (
        "version: v2\nstatus: draft\nformat: jinja2\nmessages:\n"
        "  - role: user\n    content: |-\n      {% for c in contexts %}[{{ loop.index }}] {{ c }}\n      {% endfor %}问题：{{ question }}\n"
    )
    _write_prompt(tmp_path, "basic_rag", {
        "v1.zh.yaml": "version: v1\nstatus: draft\nformat: fstring\nmessages:\n  - role: user\n    content: 自定义{question}\n",
        "v2.zh.yaml": jinja_v2,
    })
    registry = PromptRegistry.from_config(PromptsConfig(dirs=[str(tmp_path)], versions={"basic_rag": "v2"}))
    prompt = registry.get("basic_rag")
    assert prompt.version == "v2" and prompt.format == "jinja2"
    assert prompt.render(question="Q", contexts=["a", "b"])[0]["content"] == "[1] a\n[2] b\n问题：Q"
    assert registry.get("basic_rag", "stable").template("user") == "自定义{question}"
    # 自定义目录里没有的提示词仍然取内置的
    assert registry.get("rewriter").version == "v1"


def test_jinja2_missing_variable_is_an_error(tmp_path):
    _write_prompt(tmp_path, "p", {"v1.zh.yaml": "version: v1\nformat: jinja2\nmessages:\n  - role: user\n    content: '{{ contexts[0] }}'\n"})
    with pytest.raises(Exception):
        PromptRegistry(dirs=[tmp_path], include_builtin=False).get("p").render(question="q")


def test_language_does_not_fall_back(tmp_path):
    with pytest.raises(KeyError, match="没有 en 版本"):
        PromptRegistry.builtin().get("basic_rag", lang="en")


def test_lock_detects_modified_release(tmp_path):
    _write_prompt(tmp_path, "p", {"v1.zh.yaml": "version: v1\nstatus: released\nformat: fstring\nmessages:\n  - role: user\n    content: 原文{question}\n"})
    registry = PromptRegistry(dirs=[tmp_path], include_builtin=False)
    lock = tmp_path / "prompts.lock"
    registry.write_lock(lock)
    assert registry.check_lock(lock) == []
    (tmp_path / "p" / "v1.zh.yaml").write_text(
        "version: v1\nstatus: released\nformat: fstring\nmessages:\n  - role: user\n    content: 改过{question}\n", encoding="utf-8")
    problems = PromptRegistry(dirs=[tmp_path], include_builtin=False).check_lock(lock)
    assert problems and "已发布后被修改" in problems[0]
