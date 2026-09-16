#!/usr/bin/env python3
"""L2 注入级 momAction 自检：直接比较 QUDA 动能和 QUDA 内积推出的动能。

输入：
  - 12x32 cfg1000 组态
  - ChromaRNG 生成的 momentum
输出：
  - rng_results/check_mom_action_12x32.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent.parent))

from pyquda_utils import core, io
from pyquda.hmc import HMC, STSForceGrad
from pyquda.chroma_rng import ChromaRNG
from pyquda_comm import getMPIComm


LATT_SIZE = (12, 12, 12, 32)
SEED = 20260910
GAUGE_PATH = (
    "/home/jyg/stout_smear/test_stout_smesr/2ml_1ms_lime/"
    "beta6.20_mu-0.2770_ms-0.2400_L12x32_cfg_1000.lime"
)


def expected_mom_action(mom) -> float:
    """按 QUDA MomAction kernel 的 MILC 公式直接计算全局动能。"""
    data = mom.data
    local = (
        0.5 * (data[..., 6] ** 2 + data[..., 7] ** 2 + data[..., 8] ** 2)
        + data[..., 0] ** 2
        + data[..., 1] ** 2
        + data[..., 2] ** 2
        + data[..., 3] ** 2
        + data[..., 4] ** 2
        + data[..., 5] ** 2
        - 4.0
    )
    return float(getMPIComm().allreduce(float(local.sum())))


def main() -> int:
    rank = getMPIComm().rank
    outdir = SCRIPT_DIR.parent / "rng_results"
    cache = outdir / ".cache" / "check_mom_action_12x32"
    cache.mkdir(parents=True, exist_ok=True)

    core.init(None, list(LATT_SIZE), resource_path=str(cache), enable_force_monitor=False)
    latt_info = core.LatticeInfo(list(LATT_SIZE), t_boundary=-1, anisotropy=1.0)

    # momAction 只依赖 resident momentum，不需要构造 fermion action。
    hmc = HMC(latt_info, [], STSForceGrad(1))
    gauge = io.readQIOGauge(GAUGE_PATH)

    rng = ChromaRNG(LATT_SIZE, precision="double")
    rng.set_seed(SEED)
    mom = rng.make_momentum(latt_info)
    hmc.initialize(SEED, gauge, mom)

    direct = hmc.momAction()
    expected = expected_mom_action(mom)

    # 重新注入一次，确认 direct 结果稳定。
    hmc.loadMom(mom)
    direct_again = hmc.momAction()

    result = {
        "run_id": "check_mom_action_12x32",
        "seed": SEED,
        "latt_size": list(LATT_SIZE),
        "gauge_path": GAUGE_PATH,
        "direct_mom_action": direct,
        "direct_mom_action_again": direct_again,
        "expected_mom_action": expected,
        "difference": direct - expected,
        "relative_difference": abs(direct - expected) / max(abs(expected), 1.0),
    }

    if rank == 0:
        out = outdir / "check_mom_action_12x32.json"
        out.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
