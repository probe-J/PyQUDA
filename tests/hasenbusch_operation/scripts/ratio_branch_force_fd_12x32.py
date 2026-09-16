#!/usr/bin/env python3
"""12x32 Hasenbusch ratio 分支 force 有限差分诊断。

用途：固定每个 ratio monomial 的 pseudofermion、X、Y 后，把 Chroma ratio
四个高层分支分别做 action 有限差分，并和 `_add_sym_oddodd_quda_norm_deriv_force`
生成的 general-link force 投影比较。

主要输入：cfg、seed、eps、Hasenbusch shift 链、可选 ratio monomial index。
主要输出：`hasenbusch_operation/csv/ratio_branch_force_fd_<run_id>.csv`。
运行前提：p100 上先 `source set_PyQUDA_env.sh`；脚本只写 hasenbusch_operation 内部。
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
CACHE_ROOT = OP_DIR / ".cache" / "ratio_branch_force_fd"
TESTS_DIR = OP_DIR.parent
PROJECT_DIR = TESTS_DIR.parent

sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(TESTS_DIR))

from pyquda.enum_quda import QudaDagType, QudaVerbosity
from pyquda.field import LatticeFermion, LatticeGauge, LatticeMom
from pyquda_utils import core, io

import run_hmc_hasenbusch_12x32 as hmc_runner
from chroma_rng_adapter import attach_chroma_rng


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", choices=["1000", "10001"], default="1000")
    parser.add_argument("--t-boundary", type=int, choices=[-1, 1], default=-1)
    parser.add_argument("--seed", type=int, default=10086)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--eps", type=float, default=3e-5)
    parser.add_argument("--hasenbusch-mu", default="0,0.0005,0.0015,0.0035,0.0075,0.0155")
    parser.add_argument("--ratio-index", type=int, default=-1, help="-1 表示检查全部 ratio monomial。")
    parser.add_argument("--tol-override", type=float, default=1e-10)
    parser.add_argument("--maxiter-override", type=int, default=5000)
    return parser.parse_args()


def make_config(args: argparse.Namespace) -> dict:
    # 与现有 force_check 保持同一构造，只隔离 ratio monomial 和 odd_odd symmetric MatPC。
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
        sym_odd_clover_scale=1.0,
        hasenbusch_force_dagger="yes",
        monomial_subset="ratio_only",
        ratio_force_source="phi",
        ratio_force_mode="pc_quda_norm",
        cancel_force_mode="pc_quda_norm",
        cancel_sample_scale=1.0,
        cancel_constdet="yes",
        matpc_type="odd_odd",
    )
    config = hmc_runner.base_config(runner_args)
    config["tol"] = args.tol_override
    config["maxiter"] = args.maxiter_override
    return config


def copy_fermion(latt_info, src: LatticeFermion) -> LatticeFermion:
    out = LatticeFermion(latt_info)
    out.data = src.data.copy()
    return out


def copy_mom_like(latt_info, data) -> LatticeMom:
    mom = LatticeMom(latt_info)
    mom.data = data.copy()
    return mom


def kinetic_inner_product(hmc, latt_info, a: LatticeMom, b: LatticeMom) -> float:
    # 使用 QUDA momAction 定义 momentum 度量，避免手写反厄米矩阵分量权重。
    plus = copy_mom_like(latt_info, a.data + b.data)
    minus = copy_mom_like(latt_info, a.data - b.data)
    hmc.loadMom(plus)
    k_plus = hmc.momAction()
    hmc.loadMom(minus)
    k_minus = hmc.momAction()
    return (k_plus - k_minus) / 2.0


def component_value(action, left: LatticeFermion, right: LatticeFermion, dagger: QudaDagType, mu: float) -> float:
    # 计算 <left, M(mu) right> 或 <left, M(mu)^dag right> 的实部。
    old_mu = action.invert_param.mu
    action.invert_param.mu = mu
    out = LatticeFermion(action.latt_info)
    action._mat(out, right, dagger, mu)
    action.invert_param.mu = old_mu
    return float(cp.vdot(left.even.data.reshape(-1), out.even.data.reshape(-1)).real.get())


def load_shifted_smeared(hmc, gauge0: LatticeGauge, direction: LatticeMom, eps: float) -> None:
    # 从同一 thin gauge 出发，沿同一 momentum 方向扰动，再加载 stout-smeared gauge。
    hmc.loadGauge(gauge0)
    hmc.loadMom(direction)
    hmc.updateGauge(eps)
    hmc.loadGaugeMomSmeared()


def branch_force_dot(
    hmc,
    action,
    gauge0: LatticeGauge,
    direction: LatticeMom,
    left: LatticeFermion,
    right: LatticeFermion,
    dagger: QudaDagType,
    mu: float,
    branch_sign: float,
    label: str,
    shift_scale: float,
) -> tuple[float, float, float, object]:
    """单独计算一个高层 ratio 分支的 force 投影。

    注意：ratio 的 D_prec 分支必须临时把 `invert_param.mu` 切到 `mu_prec`。
    如果底层 helper 误读 `self.mu`，这里会直接暴露为 fd/force 不一致。
    """
    zero_mom = LatticeMom(action.latt_info)
    hmc.loadGauge(gauge0)
    hmc.loadMom(zero_mom)
    hmc.loadGaugeMomSmeared()
    hmc.force_v2.data = 0

    old_mu = action.invert_param.mu
    action.invert_param.mu = mu
    action.sym_shift_scale = shift_scale
    action.gauge_param.use_resident_force = 0
    action.gauge_param.return_result_force = 1
    action.gauge_param.overwrite_force = 0
    action.updateClover()
    action._add_sym_oddodd_quda_norm_deriv_force(hmc.force_v2, left, right, dagger, branch_sign, label)
    action.gauge_param.use_resident_force = 1
    action.gauge_param.return_result_force = 0
    action.invert_param.mu = old_mu
    action.sym_shift_scale = 1.0

    general_data = hmc.force_v2.data.get().copy()
    probe_link = hmc.smeared.data[0, 0, 0, 0, 0, 0].get().copy()
    print("PYQUDA_LINKPROBE", *[format(value, ".17g") for value in probe_link.reshape(-1)], flush=True)
    general_raw_sq = float((general_data * general_data.conj()).real.sum())

    hmc.loadGaugeMom()
    force_mom = LatticeMom(action.latt_info)
    hmc.saveMom(force_mom)
    data = force_mom.data.get()
    offdiag = 2.0 * (data[..., :6] ** 2).sum()
    diag = (data[..., 6:9] ** 2).sum()
    raw_sq = float(offdiag + diag)
    return (
        kinetic_inner_product(hmc, action.latt_info, force_mom, direction),
        float(cp.linalg.norm(force_mom.data).get()),
        raw_sq,
        general_raw_sq,
        general_data,
    )


def prepare_ratio_fields(hmc, action, gauge0: LatticeGauge):
    # 在基准 gauge 上固定 Chroma ratio 公式中的 phi、X、Y。
    hmc.loadGauge(gauge0)
    hmc.loadGaugeMomSmeared()
    action.updateClover()
    action._mat(action.rhs, action.phi, QudaDagType.QUDA_DAG_YES, action.mu_prec)
    action._solve(action.rhs, action.mu, compute_action=False)
    action._mat(action.y, action.quark[0], QudaDagType.QUDA_DAG_NO, action.mu)
    hmc.loadGaugeMom()
    return (
        copy_fermion(action.latt_info, action.phi),
        copy_fermion(action.latt_info, action.quark[0]),
        copy_fermion(action.latt_info, action.y),
    )


def relative_error(a: float, b: float) -> float:
    return abs(a - b) / max(abs(a), abs(b), 1.0)


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
    rng_handle = attach_chroma_rng(hmc, seed=config["seed"], precision="double", use_chroma_accept=False)
    hmc.gaussMom()
    hmc.samplePhi()

    gauge0 = LatticeGauge(latt_info)
    hmc.saveGauge(gauge0)
    hmc.gaussMom()
    direction = LatticeMom(latt_info)
    hmc.saveMom(direction)

    rows = []
    ratio_actions = [
        action for action in hmc.fermion_monomials if getattr(action, "determinant_ratio", False)
    ]
    for local_index, action in enumerate(ratio_actions):
        if args.ratio_index >= 0 and local_index != args.ratio_index:
            continue
        label = getattr(action, "label", f"ratio_{local_index}")
        phi, x, y = prepare_ratio_fields(hmc, action, gauge0)
        branches = [
            ("Dprec_phi_X_PLUS", +1.0, phi, x, QudaDagType.QUDA_DAG_NO, action.mu_prec),
            ("D_X_Y_MINUS", -1.0, x, y, QudaDagType.QUDA_DAG_YES, action.mu),
            ("D_Y_X_PLUS", -1.0, y, x, QudaDagType.QUDA_DAG_NO, action.mu),
            ("Dprec_X_phi_MINUS", +1.0, x, phi, QudaDagType.QUDA_DAG_YES, action.mu_prec),
        ]
        for branch_name, branch_sign, left, right, dagger, mu in branches:
            hmc.loadGauge(gauge0)
            hmc.loadGaugeMomSmeared()
            action.updateClover()
            shift_bilinear = component_value(action, left, right, dagger, mu) - component_value(action, left, right, dagger, 0.0)
            hmc.loadGaugeMom()
            values = []
            for eps in (args.eps, -args.eps):
                load_shifted_smeared(hmc, gauge0, direction, eps)
                action.updateClover()
                values.append(component_value(action, left, right, dagger, mu))
                hmc.loadGaugeMom()
            finite_difference_raw = (values[0] - values[1]) / (2.0 * args.eps)
            finite_difference_signed = branch_sign * finite_difference_raw
            base_dot, base_norm, base_raw_sq, base_general_raw_sq, base_general_data = branch_force_dot(
                hmc, action, gauge0, direction, left, right, dagger, mu, branch_sign, branch_name, 0.0
            )
            base_repeat_dot, base_repeat_norm, base_repeat_raw_sq, base_repeat_general_raw_sq, base_repeat_general_data = branch_force_dot(
                hmc, action, gauge0, direction, left, right, dagger, mu, branch_sign, branch_name, 0.0
            )
            force_dot, force_norm, force_raw_sq, general_raw_sq, full_general_data = branch_force_dot(
                hmc, action, gauge0, direction, left, right, dagger, mu, branch_sign, branch_name, 1.0
            )
            shift_general_data = full_general_data - base_general_data
            shift_general_raw_sq = float((shift_general_data * shift_general_data.conj()).real.sum())
            base_repeat_diff_data = base_repeat_general_data - base_general_data
            base_repeat_diff_raw_sq = float((base_repeat_diff_data * base_repeat_diff_data.conj()).real.sum())
            probe_mat = shift_general_data[0, 0, 0, 0, 0, 0]
            print("PYQUDA_SHIFTPROBE", branch_name, *[format(value, ".17g") for value in probe_mat.reshape(-1)], flush=True)
            left_norm_sq = float(cp.vdot(left.data.reshape(-1), left.data.reshape(-1)).real.get())
            right_norm_sq = float(cp.vdot(right.data.reshape(-1), right.data.reshape(-1)).real.get())
            rows.append(
                {
                    "run_id": args.run_id,
                    "cfg": config["cfg"],
                    "seed": args.seed,
                    "ratio_index": local_index,
                    "monomial_label": label,
                    "branch": branch_name,
                    "mu_for_branch": mu,
                    "shift_bilinear": shift_bilinear,
                    "left_norm_sq": left_norm_sq,
                    "right_norm_sq": right_norm_sq,
                    "branch_sign": branch_sign,
                    "eps": args.eps,
                    "value_plus": values[0],
                    "value_minus": values[1],
                    "finite_difference_raw": finite_difference_raw,
                    "finite_difference_signed": finite_difference_signed,
                    "force_dot_direction": force_dot,
                    "base_force_dot_direction": base_dot,
                    "base_force_norm": base_norm,
                    "base_force_raw_sq": base_raw_sq,
                    "base_general_force_raw_sq": base_general_raw_sq,
                    "force_raw_sq": force_raw_sq,
                    "general_force_raw_sq": general_raw_sq,
                    "shift_general_force_raw_sq": shift_general_raw_sq,
                    "base_repeat_diff_raw_sq": base_repeat_diff_raw_sq,
                    "fd_minus_force_dot": finite_difference_signed - force_dot,
                    "fd_plus_force_dot": finite_difference_signed + force_dot,
                    "relative_error_same_sign": relative_error(finite_difference_signed, force_dot),
                    "relative_error_opposite_sign": relative_error(finite_difference_signed, -force_dot),
                    "force_norm": force_norm,
                }
            )

    out_file = CSV_DIR / f"ratio_branch_force_fd_{args.run_id}.csv"
    with out_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(out_file)
    rng_handle.detach()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
