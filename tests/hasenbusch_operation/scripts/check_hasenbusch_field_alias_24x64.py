#!/usr/bin/env python3
"""24x64 Hasenbusch ratio/cancel 字段别名诊断。

用途：在同一次运行中构造 ratio 1 与 cancel，输出对象 id、设备指针、
场范数和固定 odd site 分量，检查是否存在共享内存或状态污染。
"""

import argparse
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

SCRIPT_DIR = Path(__file__).resolve().parent
OP_DIR = SCRIPT_DIR.parent
PROJECT_DIR = OP_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_DIR))

import run_hmc_hasenbusch_12x32 as hr
import run_rng_compare as rr


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gauge-path", required=True)
    p.add_argument("--latt", type=int, nargs=4, default=[24, 24, 24, 64])
    p.add_argument("--ml-bare", type=float, required=True)
    p.add_argument("--ms-bare", type=float, required=True)
    p.add_argument("--t-boundary", type=int, default=-1, choices=[1, -1])
    p.add_argument("--seed", type=int, default=20260910)
    p.add_argument("--tol", type=float, default=1e-10)
    p.add_argument("--maxiter", type=int, default=5000)
    p.add_argument("--run-id", required=True)
    return p.parse_args()


def make_config(args):
    runner_args = SimpleNamespace(
        plan="B",
        arm="chroma",
        latt=args.latt,
        cfg="custom",
        gauge_path=args.gauge_path,
        ml_bare=args.ml_bare,
        ms_bare=args.ms_bare,
        t_boundary=args.t_boundary,
        seed=args.seed,
        rng_precision="double",
        ntraj=1,
        maxiter=args.maxiter,
        tol=args.tol,
        traj_length=1.0,
        chroma_traj_length=None,
        nsteps_inner=1,
        nsteps_outer=1,
        integrator_inner="O2Nf1Ng0V",
        integrator_outer="O2Nf1Ng0V",
        placement="cancel_inner",
        hasenbusch_mu="0,0.0005,0.0015,0.0035,0.0075,0.0155",
        matpc_type="odd_odd",
        ratio_force_mode="pc_quda_norm",
        cancel_force_mode="pc_quda_norm",
        cancel_sample_scale=1.0,
        cancel_constdet="yes",
        monomial_subset="full",
        strange_matpc_type="odd_odd",
        strange_action_rational=True,
        use_chroma_accept=False,
        dump_modes="",
        run_id=args.run_id,
    )
    return rr.build_config(runner_args)


def iter_monomials(hmc):
    current = hmc
    while current is not None:
        for monomial in current.fermion_monomials:
            yield monomial
        current = current.hmc_inner


def prepare_ratio(monomial):
    monomial._mat(monomial.rhs, monomial.phi, 1, monomial.mu_prec)
    monomial._solve(monomial.rhs, monomial.mu, compute_action=False)
    monomial._mat(monomial.y, monomial.quark[0], 0, monomial.mu)


def prepare_cancel(monomial):
    monomial._solve(monomial.phi, monomial.mu, compute_action=False)
    monomial._mat(monomial.y, monomial.quark[0], 0, monomial.mu)


def array_pointer(array):
    if hasattr(array, "__cuda_array_interface__"):
        return array.__cuda_array_interface__["data"][0]
    if hasattr(array, "__array_interface__"):
        return array.__array_interface__["data"][0]
    return getattr(array, "ptr", None)


def describe(label, monomial, latt_info):
    from pyquda.enum_quda import QudaDagType
    if getattr(monomial, "determinant_ratio", False):
        prepare_ratio(monomial)
    else:
        prepare_cancel(monomial)

    coords = latt_info.coordinate()
    coords = coords.get() if hasattr(coords, "get") else coords
    odd_coords = coords[:, 1].reshape(4, -1)
    x_flat = monomial.quark[0].even.data.reshape(-1, 12).get()
    y_flat = monomial.y.even.data.reshape(-1, 12).get()
    phi_flat = monomial.phi.even.data.reshape(-1, 12).get()
    eta_flat = monomial.eta.even.data.reshape(-1, 12).get()

    print("FIELD_ALIAS", label)
    print("id_monomial", id(monomial))
    print("id_y", id(monomial.y))
    print("ptr_y", array_pointer(monomial.y.data))
    print("ptr_y_even", array_pointer(monomial.y.even_ptr))
    print("ptr_quark", array_pointer(monomial.quark[0].data))
    print("ptr_phi", array_pointer(monomial.phi.data))
    print("norm_x", float(monomial.quark[0].norm2()))
    print("norm_y", float(monomial.y.norm2()))
    print("norm_phi", float(monomial.phi.norm2()))
    for site in range(5):
        coord = tuple(int(v) for v in odd_coords[:, site])
        print(
            "PROBE",
            label,
            "site", site,
            "coord", *coord,
            "X", *x_flat[site, :4],
            "Y", *y_flat[site, :4],
            "PHI", *phi_flat[site, :4],
            "ETA", *eta_flat[site, :4],
        )


def main():
    args = parse_args()
    config = make_config(args)
    outdir = OP_DIR / "rng_results" / ".cache" / args.run_id
    outdir.mkdir(parents=True, exist_ok=True)

    from pyquda_utils import core, io
    from chroma_rng_adapter import attach_chroma_rng

    core.init(None, config["latt_size"], resource_path=str(outdir), enable_force_monitor=False)
    latt_info = core.LatticeInfo(
        config["latt_size"], t_boundary=config["t_boundary"], anisotropy=1.0
    )
    logger = logging.getLogger(args.run_id)
    logger.setLevel(logging.INFO)
    hmc = hr.make_hmc(latt_info, config, logger)
    gauge = io.readQIOGauge(str(config["gauge_path"]))
    hmc.initialize(args.seed, gauge)
    handle = attach_chroma_rng(hmc, seed=args.seed, precision="double", use_chroma_accept=False)
    hmc.gaussMom()
    hmc.samplePhi()

    ratio_counter = 0
    ratio1 = None
    cancel = None
    for monomial in iter_monomials(hmc):
        if getattr(monomial, "determinant_ratio", False):
            if ratio_counter == 0:
                ratio1 = monomial
            ratio_counter += 1
        elif getattr(monomial, "constdet", False):
            cancel = monomial

    hmc.loadGauge(gauge)
    hmc.loadGaugeMomSmeared()
    describe("ratio1", ratio1, latt_info)
    describe("cancel", cancel, latt_info)
    handle.detach()


if __name__ == "__main__":
    main()
