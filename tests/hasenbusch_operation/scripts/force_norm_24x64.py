#!/usr/bin/env python3
"""24x64 HMC 初始 force 范数诊断。

用途：在同一初始组态和同 seed 采样后，逐 monomial 输出 PyQUDA force 的
全格点 Frobenius 平方和，并按 Chroma force_monitors.cc 的定义换算为
F_sq、F_avg、F_max，便于与 Chroma XML 中的 ForcesByMonomial 直接比较。
"""

import argparse
import json
import numpy
import sys
from pathlib import Path
from types import SimpleNamespace

import cupy as cp
from mpi4py import MPI


SCRIPT_DIR = Path(__file__).resolve().parent
OP_DIR = SCRIPT_DIR.parent
PROJECT_DIR = OP_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_DIR))

import run_hmc_hasenbusch_12x32 as hr
import run_rng_compare as rr
from pyquda.field import LatticeMom
from pyquda.enum_quda import QudaInverterType
from pyquda_comm import getMPIComm
from pyquda_utils import core, io


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gauge-path", required=True)
    p.add_argument("--latt", nargs=4, type=int, default=[24, 24, 24, 64])
    p.add_argument("--ml-bare", type=float, required=True)
    p.add_argument("--ms-bare", type=float, required=True)
    p.add_argument("--t-boundary", type=int, default=-1, choices=[-1, 1])
    p.add_argument("--seed", type=int, default=20260910)
    p.add_argument("--tol", type=float, default=1e-6)
    p.add_argument("--maxiter", type=int, default=5000)
    p.add_argument("--traj-length", type=float, default=0.1)
    p.add_argument("--nsteps-inner", type=int, default=2)
    p.add_argument("--nsteps-outer", type=int, default=14)
    p.add_argument("--run-id", required=True)
    p.add_argument("--only-gauge", action="store_true")
    p.add_argument("--monomial-subset", default="full")
    p.add_argument("--clover-csw", type=float, default=None)
    p.add_argument("--mu-scale", type=float, default=None)
    p.add_argument("--sym-shift-scale", type=float, default=1.0)
    p.add_argument("--solver-type", choices=["cg", "gcr", "bicgstab"], default="cg")
    p.add_argument("--ratio-index", type=int, default=-1, help="-1 表示全部 ratio；0 表示 ratio 1")
    p.add_argument("--combine-ratios", action="store_true",
                   help="将全部 ratio action 的 force 累加后输出合力")
    p.add_argument(
        "--ratio-force-mode",
        default="pc_quda_norm",
        choices=["quda_detratio", "pair", "pc_pair", "pc_quda_twist", "pc_quda_norm"],
    )
    p.add_argument(
        "--cancel-force-mode",
        default="pc_quda_norm",
        choices=["quda_tmclover", "pc_pair", "pc_constdet", "pc_quda_twist", "pc_quda_norm"],
    )
    return p.parse_args()


def build_config(args):
    base_args = SimpleNamespace(
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
        traj_length=args.traj_length,
        chroma_traj_length=None,
        nsteps_inner=args.nsteps_inner,
        nsteps_outer=args.nsteps_outer,
        integrator_inner="STSForceGrad",
        integrator_outer="STSForceGrad",
        placement="cancel_inner",
        hasenbusch_mu="0,0.0005,0.0015,0.0035,0.0075,0.0155",
        matpc_type="odd_odd",
        ratio_force_mode=args.ratio_force_mode,
        cancel_force_mode=args.cancel_force_mode,
        cancel_sample_scale=1.0,
        cancel_constdet="yes",
        strange_matpc_type="odd_odd",
        strange_action_rational=True,
        monomial_subset="full",
        use_chroma_accept=False,
        dump_modes="",
        run_id=args.run_id,
    )
    config = rr.build_config(base_args)
    config["monomial_subset"] = args.monomial_subset
    if args.clover_csw is not None:
        config["clover_csw"] = args.clover_csw
    if args.mu_scale is not None:
        config["mu_scale"] = args.mu_scale
    return config


def monomial_label(m):
    name = type(m).__name__
    if name == "CloverWilsonLogDetAction":
        return "light_logdet"
    if name == "CloverWilsonHasenbuschTwistAction":
        if getattr(m, "determinant_ratio", False):
            return f"has_ratio_mu{m.mu_prec:g}_to_{m.mu:g}"
        return f"has_cancel_mu{m.mu:g}"
    if name == "CloverWilsonAction":
        return "strange"
    return getattr(m, "label", name)


def iter_fermion_monomials(hmc):
    for m in hmc.fermion_monomials:
        yield m
    if hmc.hmc_inner is not None:
        yield from iter_fermion_monomials(hmc.hmc_inner)


def chroma_force_stats(mom, global_volume):
    """按 Chroma force_monitors.cc 的口径计算 force 统计量。

    Chroma 对每个方向先做 taproj，再定义
    f2 = -2 Re Tr(F F)。PyQUDA 的 LatticeMom 已经是反厄米无迹代数元，
    因此每个 link 的 f2 等于 2 倍 Frobenius 平方。最终：
    F_sq = sum(f2)/(4V)，F_avg = sum(sqrt(f2))/(4V)，F_max = max(sqrt(f2))。
    """
    data = mom.data.get()
    offdiag = data[..., :6]
    diag = data[..., 6:9]
    link_norm_sq = 2.0 * (offdiag**2).sum(axis=-1) + (diag**2).sum(axis=-1)
    link_chroma_f2 = 2.0 * link_norm_sq
    link_f1 = numpy.sqrt(link_chroma_f2)
    comm = getMPIComm()
    total_norm_sq = float(comm.allreduce(float(link_norm_sq.sum())))
    per_direction_norm_sq = [
        float(comm.allreduce(float(link_norm_sq[mu].sum()))) for mu in range(4)
    ]
    total_f1 = float(comm.allreduce(float(link_f1.sum())))
    max_f1 = float(comm.allreduce(float(link_f1.max()), op=MPI.MAX))
    return {
        "force_frobenius_sq": total_norm_sq,
        "chroma_F_sq": total_norm_sq / (2.0 * global_volume),
        "chroma_F_sq_per_direction": [
            2.0 * value / global_volume for value in per_direction_norm_sq
        ],
        "chroma_F_avg": total_f1 / (4.0 * global_volume),
        "chroma_F_max": max_f1,
    }


def print_force_probe(label: str, mom, latt_info):
    """输出固定全局坐标的 QUDA 10 分量动量，供 Chroma CANCELFORCE 对拍。"""
    coords = latt_info.coordinate()
    coords = coords.get() if hasattr(coords, "get") else coords
    data = mom.data.get()
    probes = [(1, 0, 0, 0), (3, 0, 0, 0), (5, 0, 0, 0), (7, 0, 0, 0), (9, 0, 0, 0)]
    print("FORCE_PROBE", label, "shape", data.shape)
    for slot in range(2):
        flat_coords = coords[:, slot].reshape(4, -1)
        for site, probe in enumerate(probes):
            matches = numpy.nonzero((flat_coords == numpy.asarray(probe).reshape(4, 1)).all(axis=0))[0]
            if len(matches) == 0:
                continue
            index = int(matches[0])
            for mu in range(4):
                values = data[mu, slot].reshape(-1, 10)[index]
                print(
                    "FORCE_PROBE", label, "site", site, "coord", *probe,
                    "mu", mu,
                    "M01", *values[0:2],
                    "M02", *values[2:4],
                    "M12", *values[4:6],
                    "D0", values[6],
                    "D1", values[7],
                    "D2", values[8],
                )


def main():
    args = parse_args()
    rank = getMPIComm().rank
    config = build_config(args)
    cache = OP_DIR / ".cache" / "force_norm_24x64" / args.run_id
    cache.mkdir(parents=True, exist_ok=True)
    core.init(None, config["latt_size"], resource_path=str(cache), enable_force_monitor=False)
    li = core.LatticeInfo(config["latt_size"], t_boundary=config["t_boundary"], anisotropy=1.0)
    logger = core.getLogger() if hasattr(core, "getLogger") else __import__("logging").getLogger(args.run_id)
    hmc = hr.make_hmc(li, config, logger)
    for monomial in iter_fermion_monomials(hmc):
        if hasattr(monomial, "sym_shift_scale"):
            monomial.sym_shift_scale = args.sym_shift_scale
        if hasattr(monomial, "invert_param"):
            monomial.invert_param.inv_type = (
                QudaInverterType.QUDA_GCR_INVERTER
                if args.solver_type == "gcr"
                else QudaInverterType.QUDA_BICGSTAB_INVERTER
                if args.solver_type == "bicgstab"
                else QudaInverterType.QUDA_CG_INVERTER
            )
            monomial.invert_param.gcrNkrylov = 20
    gauge = io.readQIOGauge(str(config["gauge_path"]))
    hmc.initialize(args.seed, gauge)

    from chroma_rng_adapter import attach_chroma_rng

    handle = None
    if not args.only_gauge:
        handle = attach_chroma_rng(hmc, seed=args.seed, precision="double", use_chroma_accept=False)
        hmc.gaussMom()
        hmc.samplePhi()

    zero_mom = LatticeMom(li)
    force_mom = LatticeMom(li)
    combined_mom = LatticeMom(li)
    combined_mom.data = 0
    global_volume = int(numpy.prod(config["latt_size"]))
    records = []
    if not args.only_gauge:
        ratio_counter = 0
        for monomial in iter_fermion_monomials(hmc):
            label = monomial_label(monomial)
            if args.ratio_index >= 0:
                if not label.startswith("has_ratio_"):
                    continue
                if ratio_counter != args.ratio_index:
                    ratio_counter += 1
                    continue
                ratio_counter += 1
            hmc.loadGauge(gauge)
            hmc.loadMom(zero_mom)
            hmc.loadGaugeMomSmeared()
            monomial.force_v2(1.0, hmc.force_v2)
            hmc.loadGaugeMom()
            hmc.saveMom(force_mom)
            if args.combine_ratios and label.startswith("has_ratio_"):
                combined_mom.data += force_mom.data
            if "cancel" in label or label.startswith("has_ratio_"):
                print_force_probe(label, force_mom, li)
            record = {"label": label}
            record.update(chroma_force_stats(force_mom, global_volume))
            records.append(record)
        if args.combine_ratios:
            record = {"label": "combined_ratios"}
            record.update(chroma_force_stats(combined_mom, global_volume))
            records.append(record)

    # Gauge force 用作诊断基准：若它与 Chroma 同口径一致，说明换算公式、
    # LatticeMom 打包和输出路径可信，剩余差异应集中在费米子 force。
    force_mom.data = 0
    current = hmc
    while current is not None:
        current.loadGauge(gauge)
        for monomial in current.gauge_monomials:
            monomial.force(1.0, force_mom)
        current = current.hmc_inner
    gauge_record = {"label": "gauge"}
    gauge_record.update(chroma_force_stats(force_mom, global_volume))
    print_force_probe("gauge", force_mom, li)
    records.append(gauge_record)

    result = {
        "run_id": args.run_id,
        "seed": args.seed,
        "latt_size": config["latt_size"],
        "gauge_path": str(config["gauge_path"]),
        "records": records,
    }
    if rank == 0:
        out = OP_DIR / "rng_results" / f"{args.run_id}.json"
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(out)
        print(json.dumps(result, indent=2))
    if handle is not None:
        handle.detach()


if __name__ == "__main__":
    main()
