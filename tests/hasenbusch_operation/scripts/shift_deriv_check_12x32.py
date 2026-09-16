#!/usr/bin/env python3
"""12x32 Hasenbusch twist-shift 导数专项检查。

用途：固定左右费米子场后，用 M(mu)-M(0) 数值隔离 twist 项，并将其
规范场有限差分与 _add_symmetric_shift_deriv_force 的 force 投影比较。
该检查避免 ratio force 中 base 项大数相消对诊断精度的掩盖。
"""

import argparse
import csv
import sys
from pathlib import Path
from types import SimpleNamespace

import cupy as cp

OP_DIR = Path(__file__).resolve().parent.parent
SCRIPT_DIR = OP_DIR / "scripts"
CSV_DIR = OP_DIR / "csv"
CACHE_ROOT = OP_DIR / ".cache" / "shift_deriv_check"
TESTS_DIR = OP_DIR.parent
PROJECT_DIR = TESTS_DIR.parent

sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(TESTS_DIR))

from pyquda.enum_quda import QudaDagType, QudaVerbosity
from pyquda.field import LatticeGauge, LatticeMom
from pyquda_utils import core, io

import run_hmc_hasenbusch_12x32 as hmc_runner


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", choices=["1000", "10001"], default="1000")
    parser.add_argument("--t-boundary", type=int, choices=[-1, 1], default=-1)
    parser.add_argument("--seed", type=int, default=10086)
    parser.add_argument("--mu", type=float, default=0.0155)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--run-id", required=True)
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
        stout_n_smear=1,
        stout_rho=0.0,
        hasenbusch_mu="0,0.0005,0.0015,0.0035,0.0075,0.0155",
        hasenbusch_force_sign=1.0,
        hasenbusch_force_coeff_scale=1.0,
        hasenbusch_force_output_scale=1.0,
        sym_odd_clover_scale=1.0,
        hasenbusch_force_dagger="yes",
        monomial_subset="cancel_only",
        ratio_force_source="phi",
        ratio_force_mode="pc_quda_norm",
        cancel_force_mode="pc_quda_norm",
        cancel_sample_scale=1.0,
        cancel_constdet="yes",
        matpc_type="odd_odd",
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


def kinetic_inner_product(hmc, latt_info, a, b):
    plus = LatticeMom(latt_info)
    minus = LatticeMom(latt_info)
    plus.data = a.data + b.data
    minus.data = a.data - b.data
    hmc.loadMom(plus)
    k_plus = hmc.momAction()
    hmc.loadMom(minus)
    k_minus = hmc.momAction()
    return (k_plus - k_minus) / 2.0


def shift_bilinear_at_gauge(hmc, action, gauge0, direction, eps, chi, psi, dagger, mu):
    hmc.loadGauge(gauge0)
    hmc.loadMom(direction)
    hmc.updateGauge(eps)
    hmc.loadGaugeMomSmeared()
    action.updateClover()

    out_mu = type(psi)(action.latt_info)
    out_zero = type(psi)(action.latt_info)
    action._mat(out_mu, psi, dagger, mu)
    action._mat(out_zero, psi, dagger, 0.0)
    diff = out_mu.even.data - out_zero.even.data
    value = cp.vdot(chi.even.data.reshape(-1), diff.reshape(-1))
    hmc.loadGaugeMom()
    return float(value.real.get())


def base_bilinear_at_gauge(hmc, action, gauge0, direction, eps, chi, psi, dagger):
    hmc.loadGauge(gauge0)
    hmc.loadMom(direction)
    hmc.updateGauge(eps)
    hmc.loadGaugeMomSmeared()
    action.updateClover()
    out = type(psi)(action.latt_info)
    action._mat(out, psi, dagger, 0.0)
    value = cp.vdot(chi.even.data.reshape(-1), out.even.data.reshape(-1))
    hmc.loadGaugeMom()
    return float(value.real.get())


def shift_force_projection(hmc, action, gauge0, direction, chi, psi, dagger, mu):
    zero_mom = LatticeMom(action.latt_info)
    hmc.loadGauge(gauge0)
    hmc.loadMom(zero_mom)
    hmc.loadGaugeMomSmeared()
    hmc.force_v2.data = 0

    old_mu = action.invert_param.mu
    action.invert_param.mu = mu
    action.gauge_param.use_resident_force = 0
    action.gauge_param.return_result_force = 1
    action.gauge_param.overwrite_force = 0
    action.updateClover()
    branch_label = "shift-only diagnostic"
    raw_shift_label = f"{branch_label}: quda shifted i_mu_over_2_gamma5_A_oo prime"
    action.set_force_term_filter(raw_shift_label)
    action._add_sym_oddodd_quda_norm_deriv_force(
        hmc.force_v2,
        chi,
        psi,
        dagger,
        +1.0,
        branch_label,
    )
    action.set_force_term_filter(None)
    action.gauge_param.use_resident_force = 1
    action.gauge_param.return_result_force = 0
    action.invert_param.mu = old_mu

    hmc.loadGaugeMom()
    force = LatticeMom(action.latt_info)
    hmc.saveMom(force)
    return kinetic_inner_product(hmc, action.latt_info, force, direction), float(cp.linalg.norm(force.data).get())


def base_force_projection(hmc, action, gauge0, direction, chi, psi, dagger):
    zero_mom = LatticeMom(action.latt_info)
    hmc.loadGauge(gauge0)
    hmc.loadMom(zero_mom)
    hmc.loadGaugeMomSmeared()
    hmc.force_v2.data = 0
    action.gauge_param.use_resident_force = 0
    action.gauge_param.return_result_force = 1
    action.gauge_param.overwrite_force = 0
    action.updateClover()
    action._add_sym_oddodd_quda_norm_deriv_force(
        hmc.force_v2,
        chi,
        psi,
        dagger,
        +1.0,
        "base-only diagnostic",
    )
    action.gauge_param.use_resident_force = 1
    action.gauge_param.return_result_force = 0
    hmc.loadGaugeMom()
    force = LatticeMom(action.latt_info)
    hmc.saveMom(force)
    offdiag = 2.0 * (force.data[..., :6] * direction.data[..., :6]).sum()
    diag = (force.data[..., 6:9] * direction.data[..., 6:9]).sum()
    return (
        kinetic_inner_product(hmc, action.latt_info, force, direction),
        float(offdiag + diag),
        float(cp.linalg.norm(force.data).get()),
    )


def main():
    args = parse_args()
    config = make_config(args)
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    core.init(
        None,
        config["latt_size"],
        resource_path=str(CACHE_ROOT / args.run_id),
        enable_force_monitor=True,
    )
    latt_info = core.LatticeInfo(
        config["latt_size"], t_boundary=config["t_boundary"], anisotropy=1.0
    )
    hmc = hmc_runner.make_hmc(
        latt_info, config, core.getLogger() if hasattr(core, "getLogger") else None
    )
    hmc.setFermionVerbosity(QudaVerbosity.QUDA_SILENT)
    gauge = io.readQIOGauge(str(config["gauge_path"]))
    hmc.initialize(config["seed"], gauge)
    hmc.samplePhi()
    action = get_cancel_action(hmc)

    chi = type(action.phi)(latt_info)
    psi = type(action.phi)(latt_info)
    chi.data = action.phi.data.copy()
    psi.data = action.eta.data.copy()

    gauge0 = LatticeGauge(latt_info)
    hmc.saveGauge(gauge0)
    hmc.gaussMom()
    direction = LatticeMom(latt_info)
    hmc.saveMom(direction)

    rows = []
    for name, dagger in (
        ("PLUS", QudaDagType.QUDA_DAG_NO),
        ("MINUS", QudaDagType.QUDA_DAG_YES),
    ):
        value_plus = shift_bilinear_at_gauge(
            hmc, action, gauge0, direction, args.eps, chi, psi, dagger, args.mu
        )
        value_minus = shift_bilinear_at_gauge(
            hmc, action, gauge0, direction, -args.eps, chi, psi, dagger, args.mu
        )
        finite_difference = (value_plus - value_minus) / (2.0 * args.eps)
        force_dot, force_norm = shift_force_projection(
            hmc, action, gauge0, direction, chi, psi, dagger, args.mu
        )
        rel_same = abs(finite_difference - force_dot) / max(
            abs(finite_difference), abs(force_dot), 1.0
        )
        rel_opposite = abs(finite_difference + force_dot) / max(
            abs(finite_difference), abs(force_dot), 1.0
        )
        rows.append(
            {
                "run_id": args.run_id,
                "cfg": args.cfg,
                "seed": args.seed,
                "mu": args.mu,
                "branch": name,
                "eps": args.eps,
                "shift_bilinear_plus": value_plus,
                "shift_bilinear_minus": value_minus,
                "finite_difference": finite_difference,
                "force_dot_direction": force_dot,
                "force_norm": force_norm,
                "relative_error_same_sign": rel_same,
                "relative_error_opposite_sign": rel_opposite,
            }
        )

    for name, dagger in (
        ("BASE_PLUS", QudaDagType.QUDA_DAG_NO),
        ("BASE_MINUS", QudaDagType.QUDA_DAG_YES),
    ):
        value_plus = base_bilinear_at_gauge(
            hmc, action, gauge0, direction, args.eps, chi, psi, dagger
        )
        value_minus = base_bilinear_at_gauge(
            hmc, action, gauge0, direction, -args.eps, chi, psi, dagger
        )
        finite_difference = (value_plus - value_minus) / (2.0 * args.eps)
        force_dot, host_force_dot, force_norm = base_force_projection(
            hmc, action, gauge0, direction, chi, psi, dagger
        )
        rel_kinetic = abs(finite_difference - force_dot) / max(
            abs(finite_difference), abs(force_dot), 1.0
        )
        rel_host = abs(finite_difference - host_force_dot) / max(
            abs(finite_difference), abs(host_force_dot), 1.0
        )
        rows.append(
            {
                "run_id": args.run_id,
                "cfg": args.cfg,
                "seed": args.seed,
                "mu": 0.0,
                "branch": name,
                "eps": args.eps,
                "shift_bilinear_plus": value_plus,
                "shift_bilinear_minus": value_minus,
                "finite_difference": finite_difference,
                "force_dot_direction": force_dot,
                "force_norm": force_norm,
                "relative_error_same_sign": rel_kinetic,
                "relative_error_opposite_sign": rel_host,
            }
        )

    out_file = CSV_DIR / f"shift_deriv_check_{args.run_id}.csv"
    with out_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(out_file)


if __name__ == "__main__":
    main()
