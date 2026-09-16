#!/usr/bin/env python3
"""12x32 Hasenbusch ratio 四项 bilinear 有限差分诊断。

用途：固定每个 ratio monomial 的 phi、X、Y 后，分别计算 Chroma ratio force
四个 bilinear 分量对 thin gauge 扰动的有限差分，帮助判断 PyQUDA/QUDA 的
pair-force 项组合和归一化是否与 action 变分一致。

输出：hasenbusch_operation/csv/ratio_component_fd_<run_id>.csv。
环境：运行前 source set_PyQUDA_env.sh；脚本只写 hasenbusch_operation 内部。
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
CACHE_ROOT = OP_DIR / ".cache" / "ratio_component_fd"
TESTS_DIR = OP_DIR.parent
PROJECT_DIR = TESTS_DIR.parent

sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(TESTS_DIR))

from pyquda.enum_quda import QudaDagType, QudaVerbosity
from pyquda.field import LatticeFermion, LatticeGauge, LatticeMom
from pyquda_utils import core, io

import run_hmc_hasenbusch_12x32 as hmc_runner


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", choices=["1000", "10001"], default="1000")
    parser.add_argument("--t-boundary", type=int, choices=[-1, 1], default=-1)
    parser.add_argument("--seed", type=int, default=10086)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--eps", type=float, default=3e-5)
    parser.add_argument("--hasenbusch-mu", default="0,0.0005,0.0015,0.0035,0.0075,0.0155")
    parser.add_argument("--tol-override", type=float, default=1e-10)
    parser.add_argument("--maxiter-override", type=int, default=5000)
    return parser.parse_args()


def make_config(args):
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


def copy_fermion(latt_info, src):
    out = LatticeFermion(latt_info)
    out.data = src.data.copy()
    return out


def component_value(action, left, right, dagger, mu):
    old_mu = action.invert_param.mu
    action.invert_param.mu = mu
    out = LatticeFermion(action.latt_info)
    action._mat(out, right, dagger, mu)
    action.invert_param.mu = old_mu
    return float(cp.vdot(left.even.data.reshape(-1), out.even.data.reshape(-1)).real.get())


def load_shifted_smeared(hmc, gauge0, direction, eps):
    hmc.loadGauge(gauge0)
    hmc.loadMom(direction)
    hmc.updateGauge(eps)
    hmc.loadGaugeMomSmeared()


def main():
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

    rows = []
    for mono_index, action in enumerate(hmc.fermion_monomials):
        if not getattr(action, "determinant_ratio", False):
            continue
        label = getattr(action, "label", f"ratio_{mono_index}")

        # 在基准 gauge 上固定 Chroma 公式中的 phi、X、Y。
        hmc.loadGauge(gauge0)
        hmc.loadGaugeMomSmeared()
        action.updateClover()
        action._mat(action.rhs, action.phi, QudaDagType.QUDA_DAG_YES, action.mu_prec)
        action._solve(action.rhs, action.mu, compute_action=False)
        action._mat(action.y, action.quark[0], QudaDagType.QUDA_DAG_NO, action.mu)
        hmc.loadGaugeMom()
        phi = copy_fermion(latt_info, action.phi)
        x = copy_fermion(latt_info, action.quark[0])
        y = copy_fermion(latt_info, action.y)

        components = [
            ("Dprec_phi_X_PLUS", +1.0, phi, x, QudaDagType.QUDA_DAG_NO, action.mu_prec),
            ("D_X_Y_MINUS", -1.0, x, y, QudaDagType.QUDA_DAG_YES, action.mu),
            ("D_Y_X_PLUS", -1.0, y, x, QudaDagType.QUDA_DAG_NO, action.mu),
            ("Dprec_X_phi_MINUS", +1.0, x, phi, QudaDagType.QUDA_DAG_YES, action.mu_prec),
        ]
        for component, chroma_sign, left, right, dagger, mu in components:
            vals = []
            for eps in (args.eps, -args.eps):
                load_shifted_smeared(hmc, gauge0, direction, eps)
                action.updateClover()
                vals.append(component_value(action, left, right, dagger, mu))
                hmc.loadGaugeMom()
            fd_raw = (vals[0] - vals[1]) / (2.0 * args.eps)
            rows.append(
                {
                    "run_id": args.run_id,
                    "cfg": config["cfg"],
                    "monomial_index": mono_index,
                    "monomial_label": label,
                    "component": component,
                    "chroma_sign": chroma_sign,
                    "eps": args.eps,
                    "value_plus": vals[0],
                    "value_minus": vals[1],
                    "finite_difference_raw": fd_raw,
                    "finite_difference_chroma_signed": chroma_sign * fd_raw,
                }
            )

    out_file = CSV_DIR / f"ratio_component_fd_{args.run_id}.csv"
    with out_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(out_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
