#!/usr/bin/env python3
"""输出 PyQUDA HMC 初始 Hamiltonian 的逐 monomial 分项。

用途：复现方案 B 的同 seed 采样，然后分别计算 gauge、logdet、
Hasenbusch ratio/cancel 和 strange action，用于与 Chroma XML 的
PEByMonomials 逐项对照。

输入：组态路径、格点尺寸、质量、边界、seed、求解参数。
输出：rng_results/<run_id>.json；不向桌面复制中间文件。
环境前提：运行前 source set_PyQUDA_env.sh。
"""

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from types import SimpleNamespace


SCRIPT_DIR = Path(__file__).resolve().parent
OP_DIR = SCRIPT_DIR.parent
PROJECT_DIR = OP_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_DIR))

import run_hmc_hasenbusch_12x32 as hr  # noqa: E402
import run_rng_compare as rr  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--latt", type=int, nargs=4, required=True)
    parser.add_argument("--gauge-path", required=True)
    parser.add_argument("--ml-bare", type=float, required=True)
    parser.add_argument("--ms-bare", type=float, required=True)
    parser.add_argument("--t-boundary", type=int, default=-1, choices=[1, -1])
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--maxiter", type=int, default=5000)
    parser.add_argument("--traj-length", type=float, default=0.1)
    parser.add_argument(
        "--chroma-traj-length",
        type=float,
        default=None,
        help="按 Chroma tau0 指定轨迹长度；内部自动除以 sqrt(2)。",
    )
    parser.add_argument("--nsteps-inner", type=int, default=2)
    parser.add_argument("--nsteps-outer", type=int, default=14)
    parser.add_argument(
        "--cancel-sample-scale",
        type=float,
        default=1.0,
        help="cancel refresh 的额外缩放；ChromaRNG.fill_eta 已包含 sqrt(0.5)。",
    )
    parser.add_argument("--integrate", action="store_true", help="采样后执行一条 trajectory 并输出末态分项")
    parser.add_argument(
        "--strange-action-rational",
        action="store_true",
        help="strange Metropolis action 使用 action rational 近似；force 仍用 force rational。",
    )
    parser.add_argument(
        "--strange-matpc-type",
        choices=["even_even_asymmetric", "odd_odd_asymmetric", "even_even", "odd_odd"],
        default="even_even_asymmetric",
        help="strange action 的 MatPC 类型；odd_odd 对齐 Chroma SEOPREC constdet。",
    )
    parser.add_argument("--run-id", required=True)
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> dict:
    # 复用 run_rng_compare 的配置构造，再覆盖 24x64 专用质量、体积和组态。
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
        chroma_traj_length=args.chroma_traj_length,
        nsteps_inner=args.nsteps_inner,
        nsteps_outer=args.nsteps_outer,
        integrator_inner="STSForceGrad",
        integrator_outer="STSForceGrad",
        placement="cancel_inner",
        hasenbusch_mu="0,0.0005,0.0015,0.0035,0.0075,0.0155",
        matpc_type="odd_odd",
        ratio_force_mode="pc_quda_norm",
        cancel_force_mode="pc_quda_norm",
        cancel_sample_scale=args.cancel_sample_scale,
        cancel_constdet="yes",
        monomial_subset="full",
        use_chroma_accept=False,
        dump_modes="",
        strange_matpc_type=args.strange_matpc_type,
        strange_action_rational=args.strange_action_rational,
        run_id=args.run_id,
    )
    config = rr.build_config(base_args)
    config["latt_size"] = list(args.latt)
    config["ml_bare"] = args.ml_bare
    config["ms_bare"] = args.ms_bare
    config["gauge_path"] = Path(args.gauge_path)
    config["maxiter"] = args.maxiter
    config["tol"] = args.tol
    config["strange_action_rational"] = args.strange_action_rational
    config["strange_matpc_type"] = args.strange_matpc_type
    return config


def monomial_label(monomial) -> str:
    """给 action 分项一个稳定、可读的标签。"""
    class_name = type(monomial).__name__
    if class_name == "CloverWilsonLogDetAction":
        return "light_logdet"
    if class_name == "CloverWilsonHasenbuschTwistAction":
        if getattr(monomial, "determinant_ratio", False):
            return f"has_ratio_mu{monomial.mu:g}_to_{monomial.mu_prec:g}"
        return f"has_cancel_mu{monomial.mu:g}"
    if class_name == "CloverWilsonAction":
        return "strange"
    return getattr(monomial, "label", class_name)


def collect_fermion_actions(hmc, records: list) -> None:
    """按 HMC.fermionAction 的递归顺序计算分项，避免 resident 状态不一致。"""
    hmc.loadGaugeMomSmeared()
    for monomial in hmc.fermion_monomials:
        records.append({"label": monomial_label(monomial), "action": float(monomial.action())})
    hmc.loadGaugeMom()
    if hmc.hmc_inner is not None:
        collect_fermion_actions(hmc.hmc_inner, records)


def collect_sampling_norms(hmc, records: list) -> None:
    """记录采样后 eta/phi 范数，用于定位 RNG 缩放与 rational 采样差异。"""
    current = hmc
    while current is not None:
        for monomial in current.fermion_monomials:
            item = {"label": monomial_label(monomial)}
            if hasattr(monomial, "eta"):
                item["eta_norm2"] = float(monomial.eta.norm2())
            if hasattr(monomial, "phi"):
                item["phi_norm2"] = float(monomial.phi.norm2())
            records.append(item)
        current = current.hmc_inner


def main() -> int:
    args = parse_args()
    rank = hr.mpi_rank_from_env()
    logger = logging.getLogger(args.run_id)
    logger.setLevel(logging.INFO)
    if rank == 0:
        logger.addHandler(logging.StreamHandler())

    config = build_config(args)
    outdir = OP_DIR / "rng_results"
    cache = outdir / ".cache" / args.run_id
    cache.mkdir(parents=True, exist_ok=True)

    from pyquda_utils import core, io
    from chroma_rng_adapter import attach_chroma_rng

    core.init(None, config["latt_size"], resource_path=str(cache), enable_force_monitor=False)
    latt_info = core.LatticeInfo(config["latt_size"], t_boundary=config["t_boundary"], anisotropy=1.0)
    hmc = hr.make_hmc(latt_info, config, logger)
    gauge = io.readQIOGauge(str(config["gauge_path"]))
    hmc.initialize(args.seed, gauge)
    handle = attach_chroma_rng(hmc, seed=args.seed, precision="double", use_chroma_accept=False)

    # 与方案 B 单轨迹完全相同的采样顺序；本脚本不执行积分和 Metropolis。
    hmc.gaussMom()
    hmc.samplePhi()
    totals = hr.get_actions(hmc)
    sampling_norms = []
    collect_sampling_norms(hmc, sampling_norms)
    fermion_terms = []
    collect_fermion_actions(hmc, fermion_terms)

    gauge_terms = []
    current = hmc
    while current is not None:
        for monomial in current.gauge_monomials:
            gauge_terms.append({"label": monomial_label(monomial), "action": float(monomial.action())})
        current = current.hmc_inner

    totals_new = None
    fermion_terms_new = None
    gauge_terms_new = None
    if args.integrate:
        # 与 run_rng_compare 的单轨迹一致；不执行 Metropolis，便于比较末态分项。
        hmc.integrate(config["t"], 2e-15)
        totals_new = hr.get_actions(hmc)
        fermion_terms_new = []
        collect_fermion_actions(hmc, fermion_terms_new)
        gauge_terms_new = []
        current = hmc
        while current is not None:
            for monomial in current.gauge_monomials:
                gauge_terms_new.append(
                    {"label": monomial_label(monomial), "action": float(monomial.action())}
                )
            current = current.hmc_inner

    result = {
        "run_id": args.run_id,
        "seed": args.seed,
        "latt_size": config["latt_size"],
        "gauge_path": str(config["gauge_path"]),
        "totals": totals,
        "gauge_terms": gauge_terms,
        "fermion_terms": fermion_terms,
        "fermion_terms_sum": sum(item["action"] for item in fermion_terms),
        "sampling_norms": sampling_norms,
        "integrated": bool(args.integrate),
        "totals_new": totals_new,
        "gauge_terms_new": gauge_terms_new,
        "fermion_terms_new": fermion_terms_new,
        "fermion_terms_new_sum": (
            sum(item["action"] for item in fermion_terms_new) if fermion_terms_new is not None else None
        ),
        "term_deltas": (
            [
                {"label": old["label"], "delta": new["action"] - old["action"]}
                for old, new in zip(fermion_terms, fermion_terms_new)
            ]
            if fermion_terms_new is not None
            else None
        ),
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in config.items()},
    }
    output = outdir / f"{args.run_id}.json"
    if rank == 0:
        with output.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        logger.info(f"hamiltonian term report written: {output}")
    handle.detach()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
