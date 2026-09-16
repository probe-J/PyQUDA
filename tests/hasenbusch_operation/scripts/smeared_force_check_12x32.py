#!/usr/bin/env python3
"""12x32 smeared-link 层 force finite-difference 检查。

用途：绕过 stout reverse，只在已经 stout-smeared 的 gauge 上比较
  [S(U(+eps)) - S(U(-eps))] / (2 eps)
和 fermion monomial 返回的 general-link force 投影。

主要输入：cfg、monomial subset、Hasenbusch force mode、eps 列表。
主要输出：hasenbusch_operation/csv/smeared_force_check_*.csv。
运行前提：p100 上先 source set_PyQUDA_env.sh；脚本只写 hasenbusch_operation 内部。
"""

import argparse
import csv
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import cupy as cp

OP_DIR = Path(__file__).resolve().parents[1]
SCRIPT_DIR = OP_DIR / "scripts"
CSV_DIR = OP_DIR / "csv"
CACHE_ROOT = OP_DIR / ".cache" / "smeared_force_check"
TESTS_DIR = OP_DIR.parent
PROJECT_DIR = TESTS_DIR.parent

sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(TESTS_DIR))

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
    parser.add_argument("--eps", default="1e-4,3e-5,1e-5")
    parser.add_argument("--gauge-path", default=None)
    parser.add_argument("--stout-dim", type=int, choices=[3, 4], default=4)
    parser.add_argument("--monomial-subset", choices=["full", "no_ratio", "no_cancel", "cancel_only", "ratio_only", "strange_only", "logdet_only"], default="full")
    parser.add_argument("--ratio-force-mode", choices=["quda_detratio", "pair", "pc_pair", "pc_quda_twist", "pc_quda_norm", "pc_quda_norm_chroma4"], default="pc_quda_norm")
    parser.add_argument("--cancel-force-mode", choices=["quda_tmclover", "pc_pair", "pc_constdet", "pc_quda_twist", "pc_quda_norm"], default="pc_quda_norm")
    parser.add_argument("--cancel-constdet", choices=["yes", "no"], default="yes")
    parser.add_argument("--cancel-sample-scale", type=float, default=1.0)
    parser.add_argument("--matpc-type", choices=["even_even_asymmetric", "odd_odd_asymmetric", "even_even", "odd_odd"], default="odd_odd")
    parser.add_argument("--tol-override", type=float, default=1e-10)
    parser.add_argument("--maxiter-override", type=int, default=5000)
    return parser.parse_args()


def make_config(args: argparse.Namespace) -> dict:
    # 复用 HMC runner，保证 monomial 构造与真实 HMC/force_check 一致。
    runner_args = SimpleNamespace(
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
        cancel_sample_scale=args.cancel_sample_scale,
        sym_odd_clover_scale=1.0,
        cancel_constdet=args.cancel_constdet,
        matpc_type=args.matpc_type,
        monomial_subset=args.monomial_subset,
        stout_dim=args.stout_dim,
        stout_rho=0.125,
        stout_n_smear=1,
    )
    config = hmc_runner.base_config(runner_args)
    config["tol"] = args.tol_override
    config["maxiter"] = args.maxiter_override
    return config


def iter_fermion_monomials(hmc):
    # 同时遍历 outer 和 inner，避免后续切换布局时漏项。
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
    # 使用 QUDA momAction 定义反厄米 momentum 内积，保持与已有 force_check 一致。
    plus = copy_mom_like(latt_info, a.data + b.data)
    minus = copy_mom_like(latt_info, a.data - b.data)
    hmc.loadMom(plus)
    k_plus = hmc.momAction()
    hmc.loadMom(minus)
    k_minus = hmc.momAction()
    return (k_plus - k_minus) / 2.0


def kinetic_norm(mom: LatticeMom) -> float:
    return math.sqrt(float((mom.data * mom.data).sum()))


def load_resident_gauge(hmc, gauge: LatticeGauge) -> None:
    # 直接加载给定 gauge，不触发 HMC.loadGauge 的 t_boundary 再处理。
    hmc.gauge_param.use_resident_gauge = 0
    loadGaugeQuda(gauge.data_ptrs, hmc.gauge_param)
    hmc.gauge_param.use_resident_gauge = 1


def project_general_force_to_mom(latt_info, gauge: LatticeGauge, force: LatticeGauge) -> LatticeMom:
    # 与 HMC.loadGaugeMom 中 U * force_v2 -> anti-Hermitian traceless momentum 的投影保持一致。
    mom = LatticeMom(latt_info)
    uf = cp.einsum("...ab,...bc->...ac", gauge.data, force.data)
    uf_ah = (uf - uf.conj().swapaxes(axis1=-2, axis2=-1)) / 2
    uf_tlah = uf_ah - (uf_ah.trace(axis1=-2, axis2=-1) / 3)[..., None, None] * cp.eye(3)
    mom.data[..., 0] = uf_tlah[..., 0, 1].real
    mom.data[..., 1] = uf_tlah[..., 0, 1].imag
    mom.data[..., 2] = uf_tlah[..., 0, 2].real
    mom.data[..., 3] = uf_tlah[..., 0, 2].imag
    mom.data[..., 4] = uf_tlah[..., 1, 2].real
    mom.data[..., 5] = uf_tlah[..., 1, 2].imag
    mom.data[..., 6] = uf_tlah[..., 0, 0].imag
    mom.data[..., 7] = uf_tlah[..., 1, 1].imag
    mom.data[..., 8] = uf_tlah[..., 2, 2].imag
    return mom


def smeared_action(hmc) -> float:
    # 不调用 hmc.fermionAction，避免重新做 thin->smeared 与 stout reverse。
    total = 0.0
    for monomial in iter_fermion_monomials(hmc):
        total += monomial.action()
    return total


def action_at_shift(hmc, smeared0: LatticeGauge, direction: LatticeMom, eps: float) -> float:
    load_resident_gauge(hmc, smeared0)
    hmc.loadMom(direction)
    hmc.updateGauge(eps)
    return smeared_action(hmc)


def main() -> int:
    args = parse_args()
    config = make_config(args)
    eps_values = [float(item) for item in args.eps.split(",") if item.strip()]
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)

    core.init(None, config["latt_size"], resource_path=str(CACHE_ROOT / args.run_id), enable_force_monitor=True)
    latt_info = core.LatticeInfo(config["latt_size"], t_boundary=config["t_boundary"], anisotropy=1.0)
    hmc = hmc_runner.make_hmc(latt_info, config, core.getLogger() if hasattr(core, "getLogger") else None)
    hmc.setFermionVerbosity(QudaVerbosity.QUDA_SILENT)

    gauge = io.readQIOGauge(str(config["gauge_path"]))
    hmc.initialize(config["seed"], gauge)
    hmc.samplePhi()

    hmc.loadGaugeMomSmeared()
    smeared0 = LatticeGauge(latt_info)
    smeared0.data = hmc.smeared.data.copy()

    # 直接 smeared 层诊断绕过了常规 HMC force 路径；先显式建立 resident momentum。
    hmc.loadMom(LatticeMom(latt_info))
    hmc.gaussMom()
    direction = LatticeMom(latt_info)
    hmc.saveMom(direction)

    load_resident_gauge(hmc, smeared0)
    hmc.force_v2.data = 0
    for monomial in iter_fermion_monomials(hmc):
        monomial.force_v2(1.0, hmc.force_v2)
    force_mom = project_general_force_to_mom(latt_info, smeared0, hmc.force_v2)
    force_dot = kinetic_inner_product(hmc, latt_info, force_mom, direction)

    rows = []
    for eps in eps_values:
        s_plus = action_at_shift(hmc, smeared0, direction, eps)
        s_minus = action_at_shift(hmc, smeared0, direction, -eps)
        fd = (s_plus - s_minus) / (2 * eps)
        rows.append(
            {
                "run_id": args.run_id,
                "cfg": config["cfg"],
                "subset": config["monomial_subset"],
                "eps": eps,
                "S_plus": s_plus,
                "S_minus": s_minus,
                "finite_difference": fd,
                "force_dot_direction": force_dot,
                "fd_plus_force_dot": fd + force_dot,
                "relative_error_opposite_sign": abs(fd + force_dot) / max(abs(fd), abs(force_dot), 1.0),
                "direction_norm": kinetic_norm(direction),
                "force_norm": kinetic_norm(force_mom),
            }
        )

    out_file = CSV_DIR / f"smeared_force_check_{args.run_id}.csv"
    with out_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(out_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
