"""提示词命令行（M0 先提供 list / show / check / lock；diff / new / release 随评测框架在 M3 补齐）

    python -m medrag.prompts list
    python -m medrag.prompts show basic_rag [--version v1] [--lang zh]
    python -m medrag.prompts check
    python -m medrag.prompts lock
"""
import argparse
import sys

from .registry import PromptRegistry


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m medrag.prompts")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="列出提示词及各版本")
    show = sub.add_parser("show", help="查看某个版本的原始模板")
    show.add_argument("name")
    show.add_argument("--version")
    show.add_argument("--lang", default="zh")
    sub.add_parser("check", help="校验已发布版本未被修改")
    sub.add_parser("lock", help="把新发布的版本登记进 prompts.lock")
    args = parser.parse_args(argv)

    registry = PromptRegistry()
    if args.cmd == "list":
        by_name = {}
        for p in registry.all_versions():
            by_name.setdefault(p.name, []).append(f"{p.version}.{p.lang}({p.status})")
        for name in registry.names():
            aliases = registry.meta(name).get("aliases", {})
            print(f"{name:24s} {' '.join(by_name.get(name, []))}  aliases={aliases}")
    elif args.cmd == "show":
        p = registry.get(args.name, args.version, args.lang)
        print(f"# {p.ref}  status={p.status}  format={p.format}  params={p.params}")
        for role, content in p.messages:
            print(f"\n----- {role} -----\n{content}")
    elif args.cmd == "check":
        problems = registry.check_lock()
        for msg in problems:
            print(msg)
        if problems:
            return 1
        print("所有已发布版本与 prompts.lock 一致")
    elif args.cmd == "lock":
        lock = registry.write_lock()
        print(f"prompts.lock 共登记 {len(lock)} 个版本")
    return 0


if __name__ == "__main__":
    sys.exit(main())
