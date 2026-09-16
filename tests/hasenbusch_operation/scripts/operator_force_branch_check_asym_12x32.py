#!/usr/bin/env python3
"""12x32 Hasenbusch asymmetric MatPC 分支 force 诊断。

用途：固定 pseudofermion 解出的 X/Y 后，分别检查
`-<X, M^dag Y>` 与 `-<Y, M X>` 两个 cancel 分支的有限差分，
并和 asymmetric e/o PC 显式导数写出的 momentum force 内积比较。

主要输入：cfg、t_boundary、seed、eps、branch_force_mode。
主要输出：hasenbusch_operation/csv/operator_force_branch_asym_<run_id>.csv。
运行前提：p100 上先 source set_PyQUDA_env.sh；脚本只写 hasenbusch_operation 内部目录。
"""

import argparse
import csv
import sys
from pathlib import Path
from types import SimpleNamespace


OP_DIR = Path(__file__).resolve().parents[1]
SCRIPT_DIR = OP_DIR / "scripts"
CSV_DIR = OP_DIR / "csv"
CACHE_ROOT = OP_DIR / ".cache" / "operator_force_branch_asym"
TESTS_DIR = OP_DIR.parent
PROJECT_DIR = TESTS_DIR.parent

sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(TESTS_DIR))

import cupy as cp

from pyquda.enum_quda import QudaDagType, QudaVerbosity
from pyquda.field import LatticeFermion, LatticeGauge, LatticeMom
from pyquda_utils import core, io

import run_hmc_hasenbusch_12x32 as hmc_runner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", choices=["1000", "10001"], default="1000")
    parser.add_argument("--t-boundary", type=int, choices=[-1, 1], default=-1)
    parser.add_argument("--seed", type=int, default=10086)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--eps", type=float, default=3e-5)
    parser.add_argument(
        "--hasenbusch-mu",
        default="0,0.0005,0.0015,0.0035,0.0075,0.0155",
        help="逗号分隔的 Hasenbusch shift 链；最后一个 shift 用于 cancel monomial。",
    )
    parser.add_argument(
        "--matpc-type",
        choices=["even_even_asymmetric", "odd_odd_asymmetric"],
        default="odd_odd_asymmetric",
        help="asymmetric MatPC 类型；Chroma SEOPREC 对齐优先测试 odd_odd_asymmetric。",
    )
    parser.add_argument(
        "--branch-force-mode",
        choices=["eoprec", "eoprec_quda_twist", "constdet"],
        default="eoprec_quda_twist",
        help="分支 force 展开：完整 e/o PC、补 QUDA active twist、或只含 const-det D-link。",
    )
    parser.add_argument(
        "--force-sign",
        type=float,
        choices=[-1.0, 1.0],
        default=-1.0,
        help="传给单个 branch helper 的整体符号；默认对应 cancel force 的负号。",
    )
    return parser.parse_args()


def make_config(args: argparse.Namespace) -> dict:
    # 与 force_check 使用同一构造，只隔离 cancel monomial 和 asymmetric MatPC。
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
        hasenbusch_mu=args.hasenbusch_mu,
        hasenbusch_force_sign=1.0,
        hasenbusch_force_coeff_scale=1.0,
        hasenbusch_force_output_scale=1.0,
        hasenbusch_force_dagger="yes",
        monomial_subset="cancel_only",
        ratio_force_source="phi",
        ratio_force_mode="pc_pair",
        cancel_force_mode="pc_quda_twist",
        cancel_sample_scale=1.0,
        cancel_constdet="yes",
        matpc_type=args.matpc_type,
    )
    return hmc_runner.base_config(runner_args)


def get_cancel_action(hmc):
    for monomial in getattr(hmc, "fermion_monomials", []):
        if getattr(monomial, "label", "") == "has_cancel":
            return monomial
    inner = getattr(hmc, "hmc_inner", None)
    if inner is not None:
        return get_cancel_action(inner)
    raise RuntimeError("has_cancel monomial not found")


def copy_fermion(latt_info, src: LatticeFermion) -> LatticeFermion:
    out = LatticeFermion(latt_info)
    out.data = src.data.copy()
    return out


def copy_mom_like(latt_info, data) -> LatticeMom:
    mom = LatticeMom(latt_info)
    mom.data = data.copy()
    return mom


def kinetic_inner_product(hmc, latt_info, a: LatticeMom, b: LatticeMom) -> float:
    # 沿用 force_check 的 momentum 度量，避免手写 9 分量权重。
    plus = copy_mom_like(latt_info, a.data + b.data)
    minus = copy_mom_like(latt_info, a.data - b.data)
    hmc.loadMom(plus)
    k_plus = hmc.momAction()
    hmc.loadMom(minus)
    k_minus = hmc.momAction()
    return (k_plus - k_minus) / 2.0


def signed_bilinear_at_shift(hmc, action, gauge0, direction, eps, chi, psi, dagger) -> float:
    # 每次从 thin gauge 出发，扰动后走 HMC 的 stout-smear action 路径。
    hmc.loadGauge(gauge0)
    hmc.loadMom(direction)
    hmc.updateGauge(eps)
    hmc.loadGaugeMomSmeared()
    action.updateClover()
    out = LatticeFermion(action.latt_info)
    action._mat(out, psi, dagger, action.mu)
    # QUDA MatPC 半格输入/输出均通过 even_ptr 暴露；这里比较 half-vector 本身。
    value = cp.vdot(chi.even.data.reshape(-1), out.even.data.reshape(-1))
    return -float(value.real.get())


def branch_force_dot(hmc, action, gauge0, direction, chi, psi, dagger, label, mode, sign) -> tuple[float, list]:
    # 只写一个 branch 的 general-link force，再走 stout force reverse 和 momentum 投影。
    zero_mom = LatticeMom(action.latt_info)
    hmc.loadGauge(gauge0)
    hmc.loadMom(zero_mom)
    hmc.loadGaugeMomSmeared()
    hmc.force_v2.data = 0
    action.gauge_param.use_resident_force = 0
    action.gauge_param.return_result_force = 1
    action.gauge_param.overwrite_force = 0
    action.clear_force_term_history()
    action.updateClover()
    if mode == "constdet":
        action._add_eoprec_constdet_deriv_force(hmc.force_v2, chi, psi, dagger, sign, label)
    else:
        action._add_eoprec_deriv_force(
            hmc.force_v2,
            chi,
            psi,
            dagger,
            sign,
            label,
            include_quda_twist=(mode == "eoprec_quda_twist"),
        )
    terms = [dict(item) for item in action.force_term_history]
    action.gauge_param.use_resident_force = 1
    action.gauge_param.return_result_force = 0
    hmc.loadGaugeMom()
    out = LatticeMom(action.latt_info)
    hmc.saveMom(out)
    return kinetic_inner_product(hmc, action.latt_info, out, direction), terms


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
    action = get_cancel_action(hmc)

    # 固定 X=(MdagM)^-1 phi 与 Y=M X，后续只改变 gauge。
    action.updateClover()
    action._solve(action.phi, action.mu, compute_action=False)
    x = copy_fermion(latt_info, action.quark[0])
    action._mat(action.y, action.quark[0], QudaDagType.QUDA_DAG_NO, action.mu)
    y = copy_fermion(latt_info, action.y)

    gauge0 = LatticeGauge(latt_info)
    hmc.saveGauge(gauge0)
    hmc.gaussMom()
    direction = LatticeMom(latt_info)
    hmc.saveMom(direction)

    rows = []
    for name, chi, psi, dagger in (
        ("minus_Mdag_XY", x, y, QudaDagType.QUDA_DAG_YES),
        ("minus_M_YX", y, x, QudaDagType.QUDA_DAG_NO),
    ):
        s_plus = signed_bilinear_at_shift(hmc, action, gauge0, direction, args.eps, chi, psi, dagger)
        s_minus = signed_bilinear_at_shift(hmc, action, gauge0, direction, -args.eps, chi, psi, dagger)
        finite_difference = (s_plus - s_minus) / (2.0 * args.eps)
        force_dot, terms = branch_force_dot(
            hmc, action, gauge0, direction, chi, psi, dagger, name, args.branch_force_mode, args.force_sign
        )
        rows.append(
            {
                "run_id": args.run_id,
                "cfg": args.cfg,
                "matpc_type": args.matpc_type,
                "branch_force_mode": args.branch_force_mode,
                "branch": name,
                "eps": args.eps,
                "signed_bilinear_plus": s_plus,
                "signed_bilinear_minus": s_minus,
                "finite_difference": finite_difference,
                "force_dot_direction": force_dot,
                "fd_plus_force_dot": finite_difference + force_dot,
                "fd_minus_force_dot": finite_difference - force_dot,
                "term_labels": "|".join(item["raw_label"] for item in terms),
            }
        )

    out_file = CSV_DIR / f"operator_force_branch_asym_{args.run_id}.csv"
    with out_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(out_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
