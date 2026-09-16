#!/usr/bin/env python3
"""12x32 纯 stout 映射 force 有限差分检查。

用途：不经过 fermion action，直接检查 `V = stout(U)` 的链式导数。
脚本固定一条随机 momentum 扰动方向和一个随机 smeared-link 标量泛函
`S(V)=Re sum conj(sigma) * V`，比较 `[S(U+eps)-S(U-eps)]/(2eps)`
与 `computeStoutForceQuda` 回传到 thin-link momentum 后的方向内积。

主要输入：cfg、t_boundary、eps 列表、随机 seed。
主要输出：`hasenbusch_operation/csv/stout_map_force_check_<run_id>.csv`。
环境前提：p100 上先 source `set_PyQUDA_env.sh`；只写 hasenbusch_operation 内部目录。
"""

import argparse
import csv
import sys
from pathlib import Path

import cupy as cp

OP_DIR = Path(__file__).resolve().parents[1]
SCRIPT_DIR = OP_DIR / "scripts"
CSV_DIR = OP_DIR / "csv"
CACHE_ROOT = OP_DIR / ".cache" / "stout_map_force_check"
TESTS_DIR = OP_DIR.parent
PROJECT_DIR = TESTS_DIR.parent

sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(TESTS_DIR))

from pyquda.enum_quda import QudaVerbosity
from pyquda.field import LatticeGauge, LatticeMom
from pyquda_utils import core, io

import run_hmc_hasenbusch_12x32 as hmc_runner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", choices=["1000", "10001"], default="1000")
    parser.add_argument("--t-boundary", type=int, choices=[-1, 1], default=-1)
    parser.add_argument("--seed", type=int, default=10086)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--eps", default="1e-4,3e-5,1e-5")
    parser.add_argument("--gauge-path", default=None)
    parser.add_argument("--stout-dim", type=int, choices=[3, 4], default=4)
    return parser.parse_args()


def make_config(args: argparse.Namespace) -> dict:
    # 复用 HMC runner 的 lattice/gauge 参数，确保边界和组态路径口径一致。
    runner_args = argparse.Namespace(
        cfg=args.cfg,
        gauge_path=args.gauge_path,
        t_boundary=args.t_boundary,
        trajectory_length=1.0,
        stop=1,
        start=0,
        warm=0,
        seed=args.seed,
        run_id=args.run_id,
        nsteps_inner=1,
        nsteps_outer=1,
        placement="all_outer",
        integrator_inner="O2Nf1Ng0V",
        integrator_outer="O2Nf1Ng0V",
        hasenbusch_mu="0,0.0005",
        hasenbusch_force_sign=1.0,
        hasenbusch_force_coeff_scale=1.0,
        hasenbusch_force_output_scale=1.0,
        sym_odd_clover_scale=1.0,
        hasenbusch_force_dagger="yes",
        monomial_subset="strange_only",
        ratio_force_source="phi",
        ratio_force_mode="pc_quda_norm",
        cancel_force_mode="pc_quda_norm",
        cancel_sample_scale=1.0,
        cancel_constdet="yes",
        matpc_type="odd_odd",
        stout_dim=args.stout_dim,
        stout_rho=0.125,
        stout_n_smear=1,
    )
    return hmc_runner.base_config(runner_args)


def copy_mom_like(latt_info, data) -> LatticeMom:
    mom = LatticeMom(latt_info)
    mom.data = data.copy()
    return mom


def kinetic_inner_product(hmc, latt_info, a: LatticeMom, b: LatticeMom) -> float:
    # 沿用 QUDA momAction 的动量度量，避免手写 9 分量权重导致符号/系数误判。
    plus = copy_mom_like(latt_info, a.data + b.data)
    minus = copy_mom_like(latt_info, a.data - b.data)
    hmc.loadMom(plus)
    k_plus = hmc.momAction()
    hmc.loadMom(minus)
    k_minus = hmc.momAction()
    return (k_plus - k_minus) / 2.0


def random_sigma_like(gauge: LatticeGauge, seed: int) -> LatticeGauge:
    # 随机 general-link force，不投影到 su(3)，用于测试 stout map 的完整 Jacobian。
    cp.random.seed(seed)
    sigma = LatticeGauge(gauge.latt_info)
    real = cp.random.standard_normal(gauge.data.shape, dtype=cp.float64)
    imag = cp.random.standard_normal(gauge.data.shape, dtype=cp.float64)
    sigma.data = real + 1j * imag
    return sigma


def scalar_functional(sigma: LatticeGauge, smeared: LatticeGauge) -> float:
    # S(V)=Re sum_ij conj(sigma_ij) V_ij；其对 V 的 general-link 导数就是 sigma。
    return float(cp.real(cp.sum(cp.conj(sigma.data) * smeared.data)).get())


def stout_at_shift(hmc, gauge0: LatticeGauge, direction: LatticeMom, eps: float) -> LatticeGauge:
    # 使用真实 HMC 的 loadGaugeMomSmeared 路径，保留当前 AP 边界处理方式。
    hmc.loadGauge(gauge0)
    hmc.loadMom(direction)
    hmc.updateGauge(eps)
    hmc.loadGaugeMomSmeared()
    out = LatticeGauge(gauge0.latt_info)
    out.data = hmc.smeared.data.copy()
    return out


def stout_reverse_projection(hmc, latt_info, gauge0: LatticeGauge, sigma: LatticeGauge) -> LatticeMom:
    # 使用真实 HMC.loadGaugeMom 路径，把 smeared-link general force 回传到 thin-link momentum。
    zero = LatticeMom(latt_info)
    hmc.loadGauge(gauge0)
    hmc.loadMom(zero)
    hmc.loadGaugeMomSmeared()
    hmc.force_v2.data = sigma.data.copy()
    hmc.loadGaugeMom()
    out = LatticeMom(latt_info)
    hmc.saveMom(out)
    return out


def main() -> int:
    args = parse_args()
    config = make_config(args)
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)

    core.init(None, config["latt_size"], resource_path=str(CACHE_ROOT / args.run_id), enable_force_monitor=True)
    latt_info = core.LatticeInfo(config["latt_size"], t_boundary=config["t_boundary"], anisotropy=1.0)
    hmc = hmc_runner.make_hmc(latt_info, config, core.getLogger() if hasattr(core, "getLogger") else None)
    hmc.setFermionVerbosity(QudaVerbosity.QUDA_SILENT)

    gauge0 = io.readQIOGauge(str(config["gauge_path"]))
    hmc.initialize(config["seed"], gauge0)
    hmc.gaussMom()
    direction = LatticeMom(latt_info)
    hmc.saveMom(direction)
    hmc.loadGauge(gauge0)
    hmc.loadGaugeMomSmeared()
    sigma = random_sigma_like(hmc.smeared, args.seed + 17)
    force_mom = stout_reverse_projection(hmc, latt_info, gauge0, sigma)
    force_dot = kinetic_inner_product(hmc, latt_info, force_mom, direction)

    rows = []
    for eps_text in args.eps.split(","):
        eps = float(eps_text)
        s_plus = scalar_functional(sigma, stout_at_shift(hmc, gauge0, direction, eps))
        s_minus = scalar_functional(sigma, stout_at_shift(hmc, gauge0, direction, -eps))
        finite_difference = (s_plus - s_minus) / (2.0 * eps)
        rows.append(
            {
                "run_id": args.run_id,
                "cfg": args.cfg,
                "t_boundary": args.t_boundary,
                "stout_dim": args.stout_dim,
                "stout_dir_ignore": config["stout_dir_ignore"],
                "eps": eps,
                "S_plus": s_plus,
                "S_minus": s_minus,
                "finite_difference": finite_difference,
                "force_dot_direction": force_dot,
                "fd_minus_force_dot": finite_difference - force_dot,
                "fd_plus_force_dot": finite_difference + force_dot,
                "relative_error_same_sign": abs(finite_difference - force_dot)
                / max(abs(finite_difference), abs(force_dot), 1.0),
                "relative_error_opposite_sign": abs(finite_difference + force_dot)
                / max(abs(finite_difference), abs(force_dot), 1.0),
            }
        )

    out_file = CSV_DIR / f"stout_map_force_check_{args.run_id}.csv"
    with out_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(out_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
