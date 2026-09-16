#!/usr/bin/env python3
"""给 p100 Hasenbusch 实验路径增加 matpc_type 参数。

用途：
- 备份并修改 PyQUDA 的实验性 CloverWilsonHasenbuschTwistAction。
- 备份并修改 hasenbusch_operation 下的 runner / force-check 脚本。
- 打开 symmetric MatPC 路径，为后续 Chroma SEOPREC 对齐 force 做准备。

运行位置：
/home/jyg/stout_smear/PyQUDA_stout_smeared_force/PyQUDA

输出：
每个被修改文件旁边生成 .codex_bak_matpc_param_<timestamp> 备份。
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path


ROOT = Path("/home/jyg/stout_smear/PyQUDA_stout_smeared_force/PyQUDA")


def replace_once(text: str, old: str, new: str, path: Path) -> str:
    """替换唯一片段；找不到时立即失败，避免静默生成半补丁。"""
    if old not in text:
        raise SystemExit(f"pattern not found in {path}: {old[:120]!r}")
    return text.replace(old, new, 1)


def backup(path: Path, stamp: str) -> None:
    """保留原文件，便于按时间戳回滚。"""
    backup_path = path.with_name(path.name + f".codex_bak_matpc_param_{stamp}")
    backup_path.write_bytes(path.read_bytes())


def patch_action(path: Path) -> None:
    text = path.read_text()
    text = replace_once(
        text,
        '        cancel_sample_scale: float = 1.0,\n        label: str = "",\n',
        '        cancel_sample_scale: float = 1.0,\n'
        '        matpc_type: str = "even_even_asymmetric",\n'
        '        label: str = "",\n',
        path,
    )
    text = replace_once(
        text,
        "        self.cancel_sample_scale = float(cancel_sample_scale)\n"
        "        if self.determinant_ratio and self.mu_prec is None:\n",
        "        self.cancel_sample_scale = float(cancel_sample_scale)\n"
        "        matpc_map = {\n"
        '            "even_even_asymmetric": QudaMatPCType.QUDA_MATPC_EVEN_EVEN_ASYMMETRIC,\n'
        '            "odd_odd_asymmetric": QudaMatPCType.QUDA_MATPC_ODD_ODD_ASYMMETRIC,\n'
        '            "even_even": QudaMatPCType.QUDA_MATPC_EVEN_EVEN,\n'
        '            "odd_odd": QudaMatPCType.QUDA_MATPC_ODD_ODD,\n'
        "        }\n"
        "        if matpc_type not in matpc_map:\n"
        "            getLogger().critical(\n"
        '                "matpc_type must be even_even_asymmetric, odd_odd_asymmetric, even_even or odd_odd", ValueError\n'
        "            )\n"
        "        self.matpc_type_label = matpc_type\n"
        "        self.matpc_type = matpc_map[matpc_type]\n"
        "        if self.determinant_ratio and self.mu_prec is None:\n",
        path,
    )
    text = replace_once(
        text,
        "        self.invert_param.matpc_type = QudaMatPCType.QUDA_MATPC_EVEN_EVEN_ASYMMETRIC\n",
        "        self.invert_param.matpc_type = self.matpc_type\n",
        path,
    )
    text = replace_once(
        text,
        "        # PyQUDA 当前使用 even-even asymmetric；保留 odd-odd 分支便于后续对照 Chroma 文档公式。\n"
        "        if self.invert_param.matpc_type == QudaMatPCType.QUDA_MATPC_EVEN_EVEN_ASYMMETRIC:\n"
        "            return QudaParity.QUDA_EVEN_PARITY\n"
        "        if self.invert_param.matpc_type == QudaMatPCType.QUDA_MATPC_ODD_ODD_ASYMMETRIC:\n"
        "            return QudaParity.QUDA_ODD_PARITY\n"
        '        getLogger().critical("pc_pair only supports asymmetric even-even or odd-odd matpc", NotImplementedError)\n',
        "        # Chroma 的 SEOPREC Hasenbusch 作用在 odd checkerboard；这里允许显式选择 symmetric/asymmetric。\n"
        "        if self.invert_param.matpc_type in (\n"
        "            QudaMatPCType.QUDA_MATPC_EVEN_EVEN,\n"
        "            QudaMatPCType.QUDA_MATPC_EVEN_EVEN_ASYMMETRIC,\n"
        "        ):\n"
        "            return QudaParity.QUDA_EVEN_PARITY\n"
        "        if self.invert_param.matpc_type in (\n"
        "            QudaMatPCType.QUDA_MATPC_ODD_ODD,\n"
        "            QudaMatPCType.QUDA_MATPC_ODD_ODD_ASYMMETRIC,\n"
        "        ):\n"
        "            return QudaParity.QUDA_ODD_PARITY\n"
        '        getLogger().critical("unsupported matpc_type for Hasenbusch force probe", NotImplementedError)\n',
        path,
    )
    path.write_text(text)


def patch_runner(path: Path) -> None:
    text = path.read_text()
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
    action_files = [
        ROOT / "pyquda/action/clover_hasenbusch.py",
        ROOT / "pyquda_core/pyquda/action/clover_hasenbusch.py",
    ]
    runner_files = [
        ROOT / "tests/hasenbusch_operation/scripts/run_hmc_hasenbusch_12x32.py",
        ROOT / "tests/hasenbusch_operation/scripts/run_hmc_hasenbusch_12x32_split_scales.py",
    ]
    force_file = ROOT / "tests/hasenbusch_operation/scripts/force_check_hasenbusch_12x32.py"
    files = [*action_files, *runner_files, force_file]

    for path in files:
        backup(path, stamp)
    for path in action_files:
        patch_action(path)
    for path in runner_files:
        patch_runner(path)
    patch_force_check(force_file)

    print(f"patched matpc_type support, backup stamp {stamp}")
    for path in files:
        print(path)


if __name__ == "__main__":
    main()
