#!/usr/bin/env python3
"""24x64 gauge force 有限差分自检。

用途：在 PyQUDA 内部检验 GaugeAction.force 是否是 gauge action 的正确导数。
方法：沿随机动量方向 P 做正负微小规范场更新，比较中心差分 dS/dt 与
force 场和 P 的 Frobenius 内积。该检查不依赖 Chroma force monitor。
"""

import argparse
import json
import sys
from pathlib import Path

import numpy


SCRIPT_DIR = Path(__file__).resolve().parent
OP_DIR = SCRIPT_DIR.parent
PROJECT_DIR = OP_DIR.parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from pyquda.action import GaugeAction
from pyquda.hmc import HMC, STSForceGrad
from pyquda.field import LatticeMom
from pyquda_comm import getMPIComm
from pyquda_utils import core, io
from pyquda_utils.hmc_param import symanzikOneLoopGaugeLoopParam as loopParam


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gauge-path", required=True)
    p.add_argument("--latt", nargs=4, type=int, default=[24, 24, 24, 64])
    p.add_argument("--t-boundary", type=int, default=-1, choices=[-1, 1])
    p.add_argument("--beta", type=float, default=6.2)
    p.add_argument("--u0", type=float, default=0.855453)
    p.add_argument("--seed", type=int, default=20260910)
    p.add_argument("--eps", type=float, default=1e-5)
    p.add_argument("--run-id", required=True)
    return p.parse_args()


def packed_mom_inner(a: LatticeMom, b: LatticeMom) -> float:
    """计算两个反厄米无迹动量场的 Frobenius 内积。

    LatticeMom 的前 6 个数是三个独立非对角复数分量的实部和虚部，
    后 3 个数是对角虚部。反厄米矩阵的成对非对角元素贡献因子 2。
    """
    da = a.data.get()
    db = b.data.get()
    off_a = da[..., :6]
    off_b = db[..., :6]
    diag_a = da[..., 6:9]
    diag_b = db[..., 6:9]
    local = float(2.0 * (off_a * off_b).sum() + (diag_a * diag_b).sum())
    return float(getMPIComm().allreduce(local))


def main():
    args = parse_args()
    rank = getMPIComm().rank
    cache = OP_DIR / ".cache" / "gauge_force_fd" / args.run_id
    cache.mkdir(parents=True, exist_ok=True)
    core.init(None, args.latt, resource_path=str(cache), enable_force_monitor=False)

    li = core.LatticeInfo(args.latt, t_boundary=args.t_boundary, anisotropy=1.0)
    gauge = io.readQIOGauge(args.gauge_path)
    monomials = [GaugeAction(li, loopParam(args.u0), args.beta)]
    hmc = HMC(li, monomials, STSForceGrad(1))
    hmc.initialize(args.seed, gauge)

    # 生成一个确定 seed 的 QUDA 动量方向，并保存 host 副本用于内积。
    hmc.gaussMom()
    mom = LatticeMom(li)
    hmc.saveMom(mom)

    # 在初始规范场上通过正常 HMC resident-momentum 路径计算 gauge force。
    # 直接调用 GaugeAction.force(dt, mom) 会临时关闭 make_resident_gauge，
    # 破坏后续 computeGaugeLoopTraceQuda 依赖的 extended resident gauge。
    hmc.loadGauge(gauge)
    hmc.loadMom(LatticeMom(li))
    current = hmc
    while current is not None:
        current.gaugeForce(1.0)
        current = current.hmc_inner
    force = LatticeMom(li)
    hmc.saveMom(force)
    force_inner = packed_mom_inner(force, mom)

    # 中心差分：U(+eps)=exp(+eps P)U, U(-eps)=exp(-eps P)U。
    hmc.loadGauge(gauge)
    hmc.loadMom(mom)
    action_zero = hmc.gaugeAction()
    hmc.updateGauge(args.eps)
    action_plus = hmc.gaugeAction()

    hmc.loadGauge(gauge)
    hmc.loadMom(mom)
    hmc.updateGauge(-args.eps)
    action_minus = hmc.gaugeAction()

    fd_derivative = (action_plus - action_minus) / (2.0 * args.eps)
    forward_derivative = (action_plus - action_zero) / args.eps
    result = {
        "run_id": args.run_id,
        "gauge_path": args.gauge_path,
        "latt_size": args.latt,
        "seed": args.seed,
        "eps": args.eps,
        "action_zero": action_zero,
        "action_plus": action_plus,
        "action_minus": action_minus,
        "central_fd_derivative": fd_derivative,
        "forward_fd_derivative": forward_derivative,
        "force_momentum_inner": force_inner,
        "ratio_central_fd_to_force_inner": fd_derivative / force_inner if force_inner != 0 else None,
        "ratio_force_inner_to_central_fd": force_inner / fd_derivative if fd_derivative != 0 else None,
    }

    if rank == 0:
        out = OP_DIR / "rng_results" / f"{args.run_id}.json"
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(out)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
