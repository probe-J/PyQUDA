from pathlib import Path
from datetime import datetime


ROOT = Path("/home/jyg/stout_smear/PyQUDA_stout_smeared_force/PyQUDA")
FILES = [
    ROOT / "tests/hasenbusch_operation/scripts/run_hmc_hasenbusch_12x32.py",
    ROOT / "tests/hasenbusch_operation/scripts/force_check_hasenbusch_12x32.py",
]


def main() -> None:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for path in FILES:
        backup = path.with_name(path.name + f".codex_bak_pc_pair_runner_{stamp}")
        text = path.read_text(encoding="utf-8")
        backup.write_text(text, encoding="utf-8")
        text = text.replace(
            'choices=["quda_detratio", "pair"],',
            'choices=["quda_detratio", "pair", "pc_pair"],',
        )
        text = text.replace(
            "help=\"ratio force 实现：默认使用 QUDA detratio；pair 仅用于调试 PC bilinear force。\",",
            "help=\"ratio force 实现：默认 QUDA detratio；pair/pc_pair 用于调试 bilinear 和 PC block force。\",",
        )
        text = text.replace(
            "help=\"ratio force 实现：默认 QUDA detratio；pair 用于调试新增 bilinear 接口。\",",
            "help=\"ratio force 实现：默认 QUDA detratio；pair/pc_pair 用于调试 bilinear 和 PC block force。\",",
        )
        path.write_text(text, encoding="utf-8")
    print(f"patched runner pc_pair, backup stamp={stamp}")


if __name__ == "__main__":
    main()
