#!/usr/bin/env python3
"""只给 hasenbusch_operation 的 runner / force-check 增加 matpc_type 参数。"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path


ROOT = Path("/home/jyg/stout_smear/PyQUDA_stout_smeared_force/PyQUDA")


def replace_once(text: str, old: str, new: str, path: Path) -> str:
    if old not in text:
        raise SystemExit(f"pattern not found in {path}: {old[:120]!r}")
    return text.replace(old, new, 1)


def backup(path: Path, stamp: str) -> None:
    path.with_name(path.name + f".codex_bak_matpc_runner_{stamp}").write_bytes(path.read_bytes())


def patch_runner(path: Path) -> None:
    text = path.read_text()
    if "--matpc-type" in text:
        return
    text = replace_once(
        text,
        '    parser.add_argument(\n'
        '        "--cancel-constdet",\n'
        '        choices=["yes", "no"],\n'
        '        default="no",\n'
        '        help="是否按 Chroma const-det cancel 去掉 clover trlog/sigma-trace action 贡献。",\n'
        "    )\n",
        '    parser.add_argument(\n'
        '        "--cancel-constdet",\n'
        '        choices=["yes", "no"],\n'
        '        default="no",\n'
        '        help="是否按 Chroma const-det cancel 去掉 clover trlog/sigma-trace action 贡献。",\n'
        "    )\n"
        '    parser.add_argument(\n'
        '        "--matpc-type",\n'
        '        choices=["even_even_asymmetric", "odd_odd_asymmetric", "even_even", "odd_odd"],\n'
        '        default="even_even_asymmetric",\n'
        '        help="Hasenbusch shifted-clover MatPC 类型；Chroma SEOPREC 对齐优先测试 odd_odd。",\n'
        "    )\n",
        path,
    )
    text = replace_once(
        text,
        '        "cancel_constdet": args.cancel_constdet == "yes",\n',
        '        "cancel_constdet": args.cancel_constdet == "yes",\n'
        '        "matpc_type": args.matpc_type,\n',
        path,
    )
    text = replace_once(
        text,
        '        ("cancel_constdet", config["cancel_constdet"]),\n',
        '        ("cancel_constdet", config["cancel_constdet"]),\n'
        '        ("matpc_type", config["matpc_type"]),\n',
        path,
    )
    text = text.replace(
        '            ratio_force_mode=config["ratio_force_mode"],\n'
        '            label=f"has{idx}_{mu:g}_to_{has_mu[idx + 1]:g}",\n',
        '            ratio_force_mode=config["ratio_force_mode"],\n'
        '            matpc_type=config["matpc_type"],\n'
        '            label=f"has{idx}_{mu:g}_to_{has_mu[idx + 1]:g}",\n',
    )
    text = text.replace(
        '        cancel_sample_scale=config["cancel_sample_scale"],\n'
        '        label="has_cancel",\n',
        '        cancel_sample_scale=config["cancel_sample_scale"],\n'
        '        matpc_type=config["matpc_type"],\n'
        '        label="has_cancel",\n',
    )
    path.write_text(text)


def patch_force_check(path: Path) -> None:
    text = path.read_text()
    if "--matpc-type" in text:
        return
    text = replace_once(
        text,
        '    parser.add_argument(\n'
        '        "--cancel-constdet",\n'
        '        choices=["yes", "no"],\n'
        '        default="no",\n'
        '        help="是否按 Chroma const-det cancel 去掉 clover trlog/sigma-trace action 贡献。",\n'
        "    )\n",
        '    parser.add_argument(\n'
        '        "--cancel-constdet",\n'
        '        choices=["yes", "no"],\n'
        '        default="no",\n'
        '        help="是否按 Chroma const-det cancel 去掉 clover trlog/sigma-trace action 贡献。",\n'
        "    )\n"
        '    parser.add_argument(\n'
        '        "--matpc-type",\n'
        '        choices=["even_even_asymmetric", "odd_odd_asymmetric", "even_even", "odd_odd"],\n'
        '        default="even_even_asymmetric",\n'
        '        help="Hasenbusch shifted-clover MatPC 类型；Chroma SEOPREC 对齐优先测试 odd_odd。",\n'
        "    )\n",
        path,
    )
    text = replace_once(
        text,
        "        cancel_constdet=args.cancel_constdet,\n"
        "    )\n",
        "        cancel_constdet=args.cancel_constdet,\n"
        "        matpc_type=args.matpc_type,\n"
        "    )\n",
        path,
    )
    path.write_text(text)


def main() -> None:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    runner_files = [
        ROOT / "tests/hasenbusch_operation/scripts/run_hmc_hasenbusch_12x32.py",
        ROOT / "tests/hasenbusch_operation/scripts/run_hmc_hasenbusch_12x32_split_scales.py",
    ]
    force_file = ROOT / "tests/hasenbusch_operation/scripts/force_check_hasenbusch_12x32.py"
    for path in [*runner_files, force_file]:
        backup(path, stamp)
    for path in runner_files:
        patch_runner(path)
    patch_force_check(force_file)
    print(f"patched runner matpc_type support, backup stamp {stamp}")


if __name__ == "__main__":
    main()
