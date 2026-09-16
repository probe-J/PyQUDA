#!/usr/bin/env python3
"""24x64 cancel-only 半步 momentum 对拍脚本。

用途：在初始组态和 Chroma RNG 采样后，只执行内层 STS leapfrog 的
第一个半步 momentum kick，并输出固定全局坐标的 10 分量动量。
"""

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy


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
    p.add_argument("--tol", type=float, default=1e-6)
    p.add_argument("--maxiter", type=int, default=5000)
    p.add_argument("--chroma-traj-length", type=float, default=0.1)
    p.add_argument("--run-id", required=True)
    return p.parse_args()


def make_config(args):
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
        traj_length=args.chroma_traj_length / numpy.sqrt(2.0),
        chroma_traj_length=None,
        nsteps_inner=1,
        nsteps_outer=1,
        integrator_inner="STSLeapfrog",
        integrator_outer="STSLeapfrog",
        placement="cancel_inner",
        hasenbusch_mu="0,0.0005,0.0015,0.0035,0.0075,0.0155",
        matpc_type="odd_odd",
        ratio_force_mode="pc_quda_norm",
        cancel_force_mode="pc_quda_norm",
        cancel_sample_scale=1.0,
        cancel_constdet="yes",
        monomial_subset="cancel_only",
        strange_matpc_type="odd_odd",
        strange_action_rational=True,
        use_chroma_accept=False,
        dump_modes="",
        run_id=args.run_id,
    )
    return rr.build_config(base_args)


def print_momentum_probe(mom, latt_info):
    coords = latt_info.coordinate()
    coords = coords.get() if hasattr(coords, "get") else coords
    data = mom.data.get()
    probes = [(1, 0, 0, 0), (3, 0, 0, 0), (5, 0, 0, 0), (7, 0, 0, 0), (9, 0, 0, 0)]
    print("HALFSTEP_MOM shape", data.shape)
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
                    "HALFSTEP_MOM", "site", site, "coord", *probe, "mu", mu,
                    "M01", *values[0:2], "M02", *values[2:4], "M12", *values[4:6],
                    "D0", values[6], "D1", values[7], "D2", values[8],
                )


def main():
    args = parse_args()
    config = make_config(args)
    outdir = OP_DIR / "rng_results" / ".cache" / args.run_id
    outdir.mkdir(parents=True, exist_ok=True)

    from pyquda_utils import core, io
    from chroma_rng_adapter import attach_chroma_rng
    from pyquda.field import LatticeMom

    core.init(None, config["latt_size"], resource_path=str(outdir), enable_force_monitor=False)
    latt_info = core.LatticeInfo(config["latt_size"], t_boundary=config["t_boundary"], anisotropy=1.0)
    logger = core.getLogger() if hasattr(core, "getLogger") else __import__("logging").getLogger(args.run_id)
    hmc = hr.make_hmc(latt_info, config, logger)
    gauge = io.readQIOGauge(str(config["gauge_path"]))
    hmc.initialize(args.seed, gauge)
    handle = attach_chroma_rng(hmc, seed=args.seed, precision="double", use_chroma_accept=False)
    hmc.gaussMom()
    hmc.samplePhi()

    inner = hmc.hmc_inner
    initial_mom = LatticeMom(latt_info)
    inner.saveMom(initial_mom)
    print("INITIAL_MOM")
    print_momentum_probe(initial_mom, latt_info)
    inner.updateMom(config["t"] / 2.0)
    mom = LatticeMom(latt_info)
    inner.saveMom(mom)
    print_momentum_probe(mom, latt_info)
    handle.detach()


if __name__ == "__main__":
    main()
