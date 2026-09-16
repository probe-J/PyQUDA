#!/usr/bin/env python3
"""12x32 Hasenbusch ratio monomial action 有限差分拆解。

用途：在同一次 samplePhi() 得到的固定 pseudofermion 下，分别计算每个
fermion monomial 的 [S(U+eps*dU)-S(U-eps*dU)]/(2 eps)，用于定位
ratio force 不闭合到底来自哪一个 Hasenbusch shift 区间。

输出：hasenbusch_operation/csv/monomial_fd_<run_id>.csv。
环境：运行前 source set_PyQUDA_env.sh；只写 hasenbusch_operation 内部目录。
"""

import argparse
import csv
import logging
import sys
from pathlib import Path
from types import SimpleNamespace


OP_DIR = Path(__file__).resolve().parents[1]
SCRIPT_DIR = OP_DIR / "scripts"
CSV_DIR = OP_DIR / "csv"
CACHE_ROOT = OP_DIR / ".cache" / "monomial_fd"
TESTS_DIR = OP_DIR.parent
PROJECT_DIR = TESTS_DIR.parent

sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(TESTS_DIR))

from pyquda.enum_quda import QudaVerbosity
from pyquda.field import LatticeGauge, LatticeMom
from pyquda_utils import core, io

import force_check_hasenbusch_12x32 as force_check
import run_hmc_hasenbusch_12x32 as hmc_runner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", choices=["1000", "10001"], default="1000")
    parser.add_argument("--t-boundary", type=int, default=-1, choices=[-1, 1])
    parser.add_argument("--seed", type=int, default=10086)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--hasenbusch-mu", default="0,0.0005,0.0015,0.0035,0.0075,0.0155")
    parser.add_argument(
        "--monomial-subset",
        choices=["full", "no_ratio", "no_cancel", "cancel_only", "ratio_only", "strange_only", "logdet_only"],
        default="ratio_only",
    )
    parser.add_argument("--ratio-force-mode", default="pc_quda_norm")
    parser.add_argument("--cancel-force-mode", default="pc_quda_norm")
    parser.add_argument("--matpc-type", default="odd_odd")
    parser.add_argument("--hasenbusch-force-sign", type=float, default=1.0)
    parser.add_argument("--hasenbusch-force-coeff-scale", type=float, default=1.0)
    parser.add_argument("--hasenbusch-force-output-scale", type=float, default=0.25)
    parser.add_argument("--hasenbusch-force-dagger", default="yes")
    parser.add_argument("--ratio-force-source", default="phi")
    parser.add_argument("--cancel-sample-scale", type=float, default=1.0)
    parser.add_argument("--sym-odd-clover-scale", type=float, default=1.0)
    parser.add_argument("--cancel-constdet", default="yes")
    parser.add_argument("--eps", default="3e-5")
    parser.add_argument("--tol-override", type=float, default=1e-12)
    parser.add_argument("--maxiter-override", type=int, default=5000)
    return parser.parse_args()


def make_config(args: argparse.Namespace) -> dict:
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
        hasenbusch_force_sign=args.hasenbusch_force_sign,
        hasenbusch_force_coeff_scale=args.hasenbusch_force_coeff_scale,
        hasenbusch_force_output_scale=args.hasenbusch_force_output_scale,
        hasenbusch_force_dagger=args.hasenbusch_force_dagger,
        monomial_subset=args.monomial_subset,
        ratio_force_source=args.ratio_force_source,
        ratio_force_mode=args.ratio_force_mode,
        cancel_force_mode=args.cancel_force_mode,
        cancel_sample_scale=args.cancel_sample_scale,
        sym_odd_clover_scale=args.sym_odd_clover_scale,
        cancel_constdet=args.cancel_constdet,
        matpc_type=args.matpc_type,
    )
    config = hmc_runner.base_config(runner_args)
    config["tol"] = args.tol_override
    config["maxiter"] = args.maxiter_override
    return config


def monomial_label(monomial, index: int) -> str:
    return getattr(monomial, "label", "") or f"monomial_{index}"


def monomial_action_at_shift(hmc, monomial, gauge0: LatticeGauge, direction: LatticeMom, eps: float) -> float:
    # 固定同一份 pseudofermion，只改变 gauge；只评估传入的单个 monomial action。
    hmc.loadGauge(gauge0)
    hmc.loadMom(direction)
    hmc.updateGauge(eps)
    hmc.loadGaugeMomSmeared()
    action = monomial.action()
    hmc.loadGaugeMom()
    return action


def main() -> int:
    args = parse_args()
    eps_values = [float(item) for item in args.eps.split(",") if item.strip()]
    config = make_config(args)
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    csv_file = CSV_DIR / f"monomial_fd_{args.run_id}.csv"

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger(args.run_id)

    core.init(None, config["latt_size"], resource_path=str(CACHE_ROOT / args.run_id), enable_force_monitor=True)
    latt_info = core.LatticeInfo(config["latt_size"], t_boundary=config["t_boundary"], anisotropy=1.0)
    hmc = hmc_runner.make_hmc(latt_info, config, logger)
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
    for idx, monomial in enumerate(hmc.fermion_monomials):
        label = monomial_label(monomial, idx)
        for eps in eps_values:
            s_plus = monomial_action_at_shift(hmc, monomial, gauge0, direction, eps)
            s_minus = monomial_action_at_shift(hmc, monomial, gauge0, direction, -eps)
            fd = (s_plus - s_minus) / (2.0 * eps)
            rows.append(
                {
                    "run_id": args.run_id,
                    "cfg": config["cfg"],
                    "seed": config["seed"],
                    "monomial_index": idx,
                    "monomial_label": label,
                    "eps": eps,
                    "S_plus": s_plus,
                    "S_minus": s_minus,
                    "finite_difference": fd,
                }
            )
    hmc.loadGauge(gauge0)

    with csv_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run_id",
                "cfg",
                "seed",
                "monomial_index",
                "monomial_label",
                "eps",
                "S_plus",
                "S_minus",
                "finite_difference",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(csv_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
