#!/usr/bin/env python3
"""12x32 thin-link stout force 投影口径诊断。

用途：固定 pseudofermion 和随机扰动方向后，对同一个 computeStoutForceQuda
返回的 thin-link general force，分别使用不同 thin gauge 口径投影成 momentum，
再与 action finite-difference 做比较。

输出：hasenbusch_operation/csv/thin_projection_probe_*.csv。
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
CACHE_ROOT = OP_DIR / ".cache" / "thin_projection_probe"
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
    parser.add_argument("--eps", default="1e-4,3e-5,1e-5")
    parser.add_argument("--gauge-path", default=None)
    parser.add_argument("--stout-dim", type=int, choices=[3, 4], default=4)
    parser.add_argument("--ratio-force-mode", default="pc_quda_norm")
    parser.add_argument("--cancel-force-mode", default="pc_quda_norm")
    parser.add_argument("--cancel-constdet", choices=["yes", "no"], default="yes")
    parser.add_argument("--matpc-type", default="odd_odd")
    parser.add_argument("--tol-override", type=float, default=1e-10)
    parser.add_argument("--maxiter-override", type=int, default=5000)
    return parser.parse_args()


def make_config(args: argparse.Namespace) -> dict:
    # 复用正式 HMC runner 的 monomial 构造，避免诊断脚本与实验脚本分叉。
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
        monomial_subset="full",
        ratio_force_source="phi",
        ratio_force_mode=args.ratio_force_mode,
        cancel_force_mode=args.cancel_force_mode,
        cancel_sample_scale=1.0,
        sym_odd_clover_scale=1.0,
        cancel_constdet=args.cancel_constdet,
        matpc_type=args.matpc_type,
        stout_dim=args.stout_dim,
        stout_rho=0.125,
        stout_n_smear=1,
    )
    config = hmc_runner.base_config(runner_args)
    config["tol"] = args.tol_override
    config["maxiter"] = args.maxiter_override
    return config


def iter_fermion_monomials(hmc):
    # 同时遍历 outer 和 inner，保持与嵌套 HMC 布局兼容。
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
    # 采用 QUDA momentum action 定义的内积，避免手写 9 分量权重带来二义性。
    plus = copy_mom_like(latt_info, a.data + b.data)
    minus = copy_mom_like(latt_info, a.data - b.data)
    hmc.loadMom(plus)
    k_plus = hmc.momAction()
    hmc.loadMom(minus)
    k_minus = hmc.momAction()
    return (k_plus - k_minus) / 2.0


def kinetic_norm(mom: LatticeMom) -> float:
    return math.sqrt(float((mom.data * mom.data).sum()))


def project_general_force_to_mom(latt_info, gauge: LatticeGauge, force: LatticeGauge) -> LatticeMom:
    # 与 HMC.loadGaugeMom 当前实现一致：U * returned_force 后取反厄米无迹部分。
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


def action_at_shift(hmc, gauge0: LatticeGauge, direction: LatticeMom, eps: float) -> float:
    # 使用 QUDA updateGauge 沿相同 resident momentum 方向扰动 thin gauge。
    hmc.loadGauge(gauge0)
    hmc.loadMom(direction)
    hmc.updateGauge(eps)
    return hmc.fermionAction()


def compute_fat_force(hmc, gauge0: LatticeGauge) -> LatticeGauge:
    # 固定 pseudofermion 后只计算一次 smeared-link force，供所有边界组合复用。
    hmc.loadGauge(gauge0)
    # loadGaugeMomSmeared 会先从 resident momentum 读回 host；手动 probe 连续调用时
    # 需要显式补一个 resident momentum，避免前一次读回后 resident 被 QUDA 清空。
    hmc.loadMom(LatticeMom(gauge0.latt_info))
    hmc.loadGaugeMomSmeared()
    hmc.force_v2.data = 0
    for monomial in iter_fermion_monomials(hmc):
        monomial.force_v2(1.0, hmc.force_v2)
    force = LatticeGauge(gauge0.latt_info)
    force.data = hmc.force_v2.data.copy()
    return force


def returned_stout_force(
    hmc,
    gauge0: LatticeGauge,
    fat_force: LatticeGauge,
    reverse_gauge: str,
    input_ap_t: bool,
    output_ap_t: bool,
) -> LatticeGauge:
    # 穷举 reverse 的 Jacobian 评估点以及输入/输出 AP 边界映射，定位唯一闭合口径。
    hmc.force_v2.data = fat_force.data.copy()
    if input_ap_t:
        hmc.force_v2.setAntiPeriodicT()

    if reverse_gauge == "resident_ap":
        hmc.loadGauge(gauge0)
    elif reverse_gauge == "physical":
        hmc.gauge_param.use_resident_gauge = 0
        loadGaugeQuda(gauge0.data_ptrs, hmc.gauge_param)
        hmc.gauge_param.use_resident_gauge = 1
    else:
        raise ValueError(f"Unknown reverse_gauge: {reverse_gauge}")
    smear_param = quda.QudaGaugeSmearParam()
    smear_param.smear_type = 1
    smear_param.n_steps = hmc.stout_n_smear
    smear_param.rho = hmc.stout_rho
    smear_param.dir_ignore = hmc.stout_dir_ignore
    smear_param.smear_anisotropy = hmc.stout_smear_anisotropy
    hmc.gauge_param.use_resident_force = 0
    hmc.gauge_param.return_result_force = 1
    hmc.gauge_param.overwrite_force = 0
    quda.computeStoutForceQuda(hmc.force_v2.data_ptrs, hmc.gauge_param, smear_param)
    if output_ap_t:
        hmc.force_v2.setAntiPeriodicT()
    force = LatticeGauge(gauge0.latt_info)
    force.data = hmc.force_v2.data.copy()
    return force


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

    gauge0 = LatticeGauge(latt_info)
    hmc.saveGauge(gauge0)
    gauge0_ap = gauge0.copy()
    if config["t_boundary"] == -1:
        gauge0_ap.setAntiPeriodicT()

    hmc.gaussMom()
    direction = LatticeMom(latt_info)
    hmc.saveMom(direction)
    projections = {}
    fat_force = compute_fat_force(hmc, gauge0)
    for reverse_gauge in ("resident_ap", "physical"):
        for input_ap_t in (False, True):
            for output_ap_t in (False, True):
                thin_force = returned_stout_force(
                    hmc,
                    gauge0,
                    fat_force,
                    reverse_gauge,
                    input_ap_t,
                    output_ap_t,
                )
                prefix = (
                    f"reverse={reverse_gauge}:input_ap={int(input_ap_t)}:"
                    f"output_ap={int(output_ap_t)}"
                )
                projections[f"{prefix}:project=physical_U"] = project_general_force_to_mom(
                    latt_info, gauge0, thin_force
                )
                projections[f"{prefix}:project=resident_ap_U"] = project_general_force_to_mom(
                    latt_info, gauge0_ap, thin_force
                )

    rows = []
    force_dots = {
        name: kinetic_inner_product(hmc, latt_info, mom, direction)
        for name, mom in projections.items()
    }
    force_norms = {name: kinetic_norm(mom) for name, mom in projections.items()}
    for eps in eps_values:
        s_plus = action_at_shift(hmc, gauge0, direction, eps)
        s_minus = action_at_shift(hmc, gauge0, direction, -eps)
        fd = (s_plus - s_minus) / (2 * eps)
        for name, force_dot in force_dots.items():
            rows.append(
                {
                    "run_id": args.run_id,
                    "cfg": config["cfg"],
                    "t_boundary": config["t_boundary"],
                    "projection": name,
                    "eps": eps,
                    "S_plus": s_plus,
                    "S_minus": s_minus,
                    "finite_difference": fd,
                    "force_dot_direction": force_dot,
                    "fd_plus_force_dot": fd + force_dot,
                    "relative_error_opposite_sign": abs(fd + force_dot) / max(abs(fd), abs(force_dot), 1.0),
                    "direction_norm": kinetic_norm(direction),
                    "force_norm": force_norms[name],
                }
            )

    out_file = CSV_DIR / f"thin_projection_probe_{args.run_id}.csv"
    with out_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(out_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
