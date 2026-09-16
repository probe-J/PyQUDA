#!/usr/bin/env python3
"""12x32 stout reverse 输入/输出口径诊断。

用途：在 fermion force 的 smeared-link 层已经闭合后，固定同一组 pseudofermion
和同一条 thin-link 扰动方向，比较不同 `computeStoutForceQuda` 输入/输出解释下的
thin-link force 投影。输出用于判断 QUDA stout reverse 期望的是 general-link force，
还是已经左乘 smeared link 的 momentum-like force。

主要输入：cfg、monomial subset、Hasenbusch force mode、eps。
主要输出：hasenbusch_operation/csv/stout_reverse_convention_<run_id>.csv。
运行前提：p100 上先 source set_PyQUDA_env.sh；脚本只写 hasenbusch_operation 内部。
"""

import argparse
import csv
import sys
from pathlib import Path
from types import SimpleNamespace

import cupy as cp

OP_DIR = Path(__file__).resolve().parents[1]
SCRIPT_DIR = OP_DIR / "scripts"
CSV_DIR = OP_DIR / "csv"
CACHE_ROOT = OP_DIR / ".cache" / "stout_reverse_convention"
TESTS_DIR = OP_DIR.parent
PROJECT_DIR = TESTS_DIR.parent

sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(TESTS_DIR))

from pyquda import quda
from pyquda.enum_quda import QudaVerbosity
from pyquda.field import LatticeGauge, LatticeMom
from pyquda.quda import loadGaugeQuda
from pyquda_utils import core, io

import run_hmc_hasenbusch_12x32 as hmc_runner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", choices=["1000", "10001"], default="1000")
    parser.add_argument("--t-boundary", type=int, default=-1, choices=[-1, 1])
    parser.add_argument("--seed", type=int, default=10086)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--eps", type=float, default=3e-5)
    parser.add_argument("--monomial-subset", choices=["full", "ratio_only", "no_ratio"], default="full")
    parser.add_argument("--ratio-force-mode", choices=["pc_quda_norm", "pc_quda_norm_chroma4"], default="pc_quda_norm")
    parser.add_argument("--cancel-force-mode", choices=["pc_quda_norm", "pc_constdet"], default="pc_quda_norm")
    parser.add_argument("--cancel-constdet", choices=["yes", "no"], default="yes")
    parser.add_argument("--matpc-type", choices=["odd_odd", "even_even"], default="odd_odd")
    parser.add_argument("--tol-override", type=float, default=1e-10)
    parser.add_argument("--maxiter-override", type=int, default=5000)
    return parser.parse_args()


def make_config(args: argparse.Namespace) -> dict:
    # 复用 HMC runner，保证 force 结构和真实 HMC/force_check 完全一致。
    runner_args = SimpleNamespace(
        cfg=args.cfg,
        gauge_path=None,
        t_boundary=args.t_boundary,
        trajectory_length=1.0,
        stop=1,
        start=0,
        warm=0,
        seed=args.seed,
        run_id=args.run_id,
        nsteps_inner=1,
        nsteps_outer=56,
        placement="all_outer",
        integrator_inner="O2Nf1Ng0V",
        integrator_outer="O2Nf1Ng0V",
        hasenbusch_mu="0,0.0005,0.0015,0.0035,0.0075,0.0155",
        hasenbusch_force_sign=1.0,
        hasenbusch_force_coeff_scale=1.0,
        hasenbusch_force_output_scale=1.0,
        hasenbusch_force_dagger="yes",
        ratio_force_source="phi",
        ratio_force_mode=args.ratio_force_mode,
        cancel_force_mode=args.cancel_force_mode,
        cancel_sample_scale=1.0,
        sym_odd_clover_scale=1.0,
        cancel_constdet=args.cancel_constdet,
        matpc_type=args.matpc_type,
        monomial_subset=args.monomial_subset,
    )
    config = hmc_runner.base_config(runner_args)
    config["tol"] = args.tol_override
    config["maxiter"] = args.maxiter_override
    return config


def iter_fermion_monomials(hmc):
    for monomial in getattr(hmc, "fermion_monomials", []):
        yield monomial
    inner = getattr(hmc, "hmc_inner", None)
    if inner is not None:
        yield from iter_fermion_monomials(inner)


def copy_mom_like(latt_info, data) -> LatticeMom:
    mom = LatticeMom(latt_info)
    mom.data = data.copy()
    return mom


def kinetic_inner_product(hmc, latt_info, a: LatticeMom, b: LatticeMom) -> float:
    # 沿用 QUDA momAction 的动量度量，避免手写 9 分量权重导致误判。
    plus = copy_mom_like(latt_info, a.data + b.data)
    minus = copy_mom_like(latt_info, a.data - b.data)
    hmc.loadMom(plus)
    k_plus = hmc.momAction()
    hmc.loadMom(minus)
    k_minus = hmc.momAction()
    return (k_plus - k_minus) / 2.0


def ah_traceless(matrix):
    ah = (matrix - matrix.conj().swapaxes(axis1=-2, axis2=-1)) / 2
    return ah - (ah.trace(axis1=-2, axis2=-1) / 3)[..., None, None] * cp.eye(3)


def matrix_to_mom(latt_info, matrix) -> LatticeMom:
    # 把反厄米无迹矩阵按 PyQUDA LatticeMom 9 分量约定压缩。
    mom = LatticeMom(latt_info)
    mat = ah_traceless(matrix)
    mom.data[..., 0] = mat[..., 0, 1].real
    mom.data[..., 1] = mat[..., 0, 1].imag
    mom.data[..., 2] = mat[..., 0, 2].real
    mom.data[..., 3] = mat[..., 0, 2].imag
    mom.data[..., 4] = mat[..., 1, 2].real
    mom.data[..., 5] = mat[..., 1, 2].imag
    mom.data[..., 6] = mat[..., 0, 0].imag
    mom.data[..., 7] = mat[..., 1, 1].imag
    mom.data[..., 8] = mat[..., 2, 2].imag
    return mom


def project_with_link(latt_info, gauge: LatticeGauge, force: LatticeGauge) -> LatticeMom:
    # 当前 HMC.loadGaugeMom 的口径：U_thin * force 后取反厄米无迹。
    return matrix_to_mom(latt_info, cp.einsum("...ab,...bc->...ac", gauge.data, force.data))


def load_resident_gauge(hmc, gauge: LatticeGauge) -> None:
    hmc.gauge_param.use_resident_gauge = 0
    loadGaugeQuda(gauge.data_ptrs, hmc.gauge_param)
    hmc.gauge_param.use_resident_gauge = 1


def stout_reverse(hmc, thin_gauge: LatticeGauge, sigma: LatticeGauge) -> LatticeGauge:
    # 直接调用 QUDA stout force reverse，避免走 HMC.loadGaugeMom 的固定投影。
    out = LatticeGauge(thin_gauge.latt_info)
    out.data = sigma.data.copy()
    load_resident_gauge(hmc, thin_gauge)
    smear_param = quda.QudaGaugeSmearParam()
    smear_param.smear_type = 1
    smear_param.rho = 0.125
    smear_param.dir_ignore = 4
    hmc.gauge_param.use_resident_force = 0
    hmc.gauge_param.return_result_force = 1
    hmc.gauge_param.overwrite_force = 0
    quda.computeStoutForceQuda(out.data_ptrs, hmc.gauge_param, smear_param)
    hmc.gauge_param.use_resident_force = 1
    hmc.gauge_param.return_result_force = 0
    return out


def fermion_action_at_shift(hmc, gauge0: LatticeGauge, direction: LatticeMom, eps: float) -> float:
    hmc.loadGauge(gauge0)
    hmc.loadMom(direction)
    hmc.updateGauge(eps)
    return hmc.fermionAction()


def main() -> int:
    args = parse_args()
    config = make_config(args)
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)

    core.init(None, config["latt_size"], resource_path=str(CACHE_ROOT / args.run_id), enable_force_monitor=True)
    latt_info = core.LatticeInfo(config["latt_size"], t_boundary=config["t_boundary"], anisotropy=1.0)
    hmc = hmc_runner.make_hmc(latt_info, config, core.getLogger() if hasattr(core, "getLogger") else None)
    hmc.setFermionVerbosity(QudaVerbosity.QUDA_SILENT)

    gauge = io.readQIOGauge(str(config["gauge_path"]))
    hmc.initialize(config["seed"], gauge)
    hmc.samplePhi()
    gauge0 = LatticeGauge(latt_info)
    hmc.saveGauge(gauge0)
    hmc.gaussMom()
    direction = LatticeMom(latt_info)
    hmc.saveMom(direction)

    s_plus = fermion_action_at_shift(hmc, gauge0, direction, args.eps)
    s_minus = fermion_action_at_shift(hmc, gauge0, direction, -args.eps)
    finite_difference = (s_plus - s_minus) / (2.0 * args.eps)

    hmc.loadGauge(gauge0)
    hmc.loadMom(LatticeMom(latt_info))
    hmc.loadGaugeMomSmeared()
    thin = LatticeGauge(latt_info)
    thin.data = hmc.gauge.data.copy()
    smeared = LatticeGauge(latt_info)
    smeared.data = hmc.smeared.data.copy()
    raw_sigma = LatticeGauge(latt_info)
    raw_sigma.data = 0
    for monomial in iter_fermion_monomials(hmc):
        monomial.force_v2(1.0, raw_sigma)

    smeared_left = LatticeGauge(latt_info)
    smeared_left.data = cp.einsum("...ab,...bc->...ac", smeared.data, raw_sigma.data)
    smeared_left_ah = LatticeGauge(latt_info)
    smeared_left_ah.data = ah_traceless(smeared_left.data)

    rows = []
    for input_name, sigma in [
        ("raw_general_force", raw_sigma),
        ("smeared_left_force", smeared_left),
        ("smeared_left_ah_traceless", smeared_left_ah),
    ]:
        reversed_force = stout_reverse(hmc, thin, sigma)
        projections = [
            ("project_Uthin_times_returned", project_with_link(latt_info, thin, reversed_force)),
            ("direct_returned_ah_traceless", matrix_to_mom(latt_info, reversed_force.data)),
        ]
        for output_name, mom in projections:
            force_dot = kinetic_inner_product(hmc, latt_info, mom, direction)
            rows.append(
                {
                    "run_id": args.run_id,
                    "cfg": config["cfg"],
                    "subset": config["monomial_subset"],
                    "eps": args.eps,
                    "input_convention": input_name,
                    "output_projection": output_name,
                    "S_plus": s_plus,
                    "S_minus": s_minus,
                    "finite_difference": finite_difference,
                    "force_dot_direction": force_dot,
                    "fd_plus_force_dot": finite_difference + force_dot,
                    "relative_error": abs(finite_difference + force_dot)
                    / max(abs(finite_difference), abs(force_dot), 1.0),
                }
            )

    out_file = CSV_DIR / f"stout_reverse_convention_{args.run_id}.csv"
    with out_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(out_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
