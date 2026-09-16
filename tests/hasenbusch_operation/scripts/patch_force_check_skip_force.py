#!/usr/bin/env python3
"""给 Hasenbusch force-check 增加 action-only 模式。"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path


PATH = Path(
    "/home/jyg/stout_smear/PyQUDA_stout_smeared_force/PyQUDA/tests/hasenbusch_operation/scripts/force_check_hasenbusch_12x32.py"
)


def replace_once(text: str, old: str, new: str) -> str:
    if old not in text:
        raise SystemExit(f"pattern not found: {old[:120]!r}")
    return text.replace(old, new, 1)


def main() -> None:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    PATH.with_name(PATH.name + f".codex_bak_skip_force_{stamp}").write_bytes(PATH.read_bytes())
    text = PATH.read_text()
    if "--skip-force" in text:
        print("skip-force already present")
        return

    text = replace_once(
        text,
        '    parser.add_argument(\n'
        '        "--eps",\n'
        '        default="1e-6,3e-6,1e-5,3e-5",\n'
        '        help="逗号分隔的正负扰动步长列表。",\n'
        "    )\n",
        '    parser.add_argument(\n'
        '        "--eps",\n'
        '        default="1e-6,3e-6,1e-5,3e-5",\n'
        '        help="逗号分隔的正负扰动步长列表。",\n'
        "    )\n"
        '    parser.add_argument(\n'
        '        "--skip-force",\n'
        '        action="store_true",\n'
        '        help="只计算固定 pseudofermion 下的 action finite-difference，不调用 fermionForce。",\n'
        "    )\n",
    )

    old_block = """    zero_mom = LatticeMom(latt_info)
    force_mom = LatticeMom(latt_info)
    hmc.loadGauge(gauge0)
    hmc.loadMom(zero_mom)
    hmc.fermionForce(1.0)
    hmc.saveMom(force_mom)

    force_dot_direction = kinetic_inner_product(hmc, latt_info, force_mom, direction)
    host_force_dot_direction = host_inner_product(force_mom, direction)
    host_weighted_force_dot_direction = host_weighted_inner_product(force_mom, direction)
    direction_norm = kinetic_norm(hmc, direction)
    force_norm = kinetic_norm(hmc, force_mom)
    logger.info(f"force_dot_direction = {force_dot_direction:.17g}")
    logger.info(f"host_force_dot_direction = {host_force_dot_direction:.17g}")
    logger.info(f"host_weighted_force_dot_direction = {host_weighted_force_dot_direction:.17g}")
    logger.info(f"direction_norm = {direction_norm:.17g}")
    logger.info(f"force_norm = {force_norm:.17g}")
    if direction_norm <= 0:
        raise RuntimeError(
            "force check direction_norm <= 0; resident momentum flags or save/load mom are inconsistent"
        )
"""
    new_block = """    direction_norm = kinetic_norm(hmc, direction)
    if direction_norm <= 0:
        raise RuntimeError(
            "force check direction_norm <= 0; resident momentum flags or save/load mom are inconsistent"
        )
    if args.skip_force:
        # symmetric MatPC 先做 action-only probe；force 分支实现后再关闭 skip-force 做闭合检查。
        force_dot_direction = float("nan")
        host_force_dot_direction = float("nan")
        host_weighted_force_dot_direction = float("nan")
        force_norm = float("nan")
        logger.info("skip_force = true")
        logger.info(f"direction_norm = {direction_norm:.17g}")
    else:
        zero_mom = LatticeMom(latt_info)
        force_mom = LatticeMom(latt_info)
        hmc.loadGauge(gauge0)
        hmc.loadMom(zero_mom)
        hmc.fermionForce(1.0)
        hmc.saveMom(force_mom)

        force_dot_direction = kinetic_inner_product(hmc, latt_info, force_mom, direction)
        host_force_dot_direction = host_inner_product(force_mom, direction)
        host_weighted_force_dot_direction = host_weighted_inner_product(force_mom, direction)
        force_norm = kinetic_norm(hmc, force_mom)
        logger.info(f"force_dot_direction = {force_dot_direction:.17g}")
        logger.info(f"host_force_dot_direction = {host_force_dot_direction:.17g}")
        logger.info(f"host_weighted_force_dot_direction = {host_weighted_force_dot_direction:.17g}")
        logger.info(f"direction_norm = {direction_norm:.17g}")
        logger.info(f"force_norm = {force_norm:.17g}")
"""
    text = replace_once(text, old_block, new_block)
    PATH.write_text(text)
    print(f"patched skip-force, backup stamp {stamp}")


if __name__ == "__main__":
    main()
