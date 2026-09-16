#!/usr/bin/env python3
"""RNG A/B 验证实验驱动（只新增，不改动现有 runner）。

方案 A：--plan A --arm native|chroma，各跑 --ntraj 条 trajectory，输出 JSON；
        之后用 --compare native.json chroma.json 做统计等价检验。
方案 B：--plan B --arm chroma 跑 1 条 trajectory（chroma 流），并输出
        uniforms/eta/mom 参考 dump 供 chroma_rng_dump.cc 侧逐数对拍。

用法示例（4×P100）：
    mpirun -np 4 python3 run_rng_compare.py --plan A --arm native \
        --latt 12 12 12 32 --cfg 1000 --seed 20260910 --ntraj 10 \
        --run-id rngA_native_12x32
"""

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent.parent))

import run_hmc_hasenbusch_12x32 as hr  # noqa: E402
from pyquda_comm import getLogger  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--plan", choices=["A", "B"], required=True)
    p.add_argument("--arm", choices=["native", "chroma"], default="native")
    p.add_argument("--latt", type=int, nargs=4, default=[12, 12, 12, 32])
    p.add_argument("--cfg", choices=["1000", "10001"], default="1000")
    p.add_argument("--gauge-path", default=None)
    p.add_argument("--ml-bare", type=float, default=None,
                   help="轻夸克裸质量；默认沿用 12x32 基线 -0.2770")
    p.add_argument("--ms-bare", type=float, default=None,
                   help="奇夸克裸质量；默认沿用 12x32 基线 -0.2400")
    p.add_argument("--t-boundary", type=int, default=-1, choices=[1, -1])
    p.add_argument("--seed", type=int, default=20260910)
    p.add_argument("--rng-precision", choices=["double", "single"], default="double")
    p.add_argument("--ntraj", type=int, default=None)
    p.add_argument("--maxiter", type=int, default=None,
                   help="线性求解最大迭代数；默认沿用 12x32 基线 1000")
    p.add_argument("--tol", type=float, default=1e-6,
                   help="线性求解相对残差；默认沿用 12x32 基线 1e-6")
    p.add_argument("--traj-length", type=float, default=1.0)
    p.add_argument("--chroma-traj-length", type=float, default=None,
                   help="按 Chroma tau0 指定轨迹长度；内部自动除以 sqrt(2)。")
    p.add_argument("--nsteps-inner", type=int, default=2)
    p.add_argument("--nsteps-outer", type=int, default=14)
    p.add_argument("--integrator-inner", default="STSForceGrad", choices=hr.INTEGRATORS.keys())
    p.add_argument("--integrator-outer", default="STSForceGrad", choices=hr.INTEGRATORS.keys())
    p.add_argument("--placement", default="cancel_inner",
                   choices=["cancel_inner", "cancel_gauge_inner", "all_fermion_outer", "all_outer"])
    p.add_argument("--hasenbusch-mu", default="0,0.0005,0.0015,0.0035,0.0075,0.0155")
    p.add_argument("--ratio-index", type=int, default=-1,
                   help="只保留指定 ratio action；-1 表示全部 ratio")
    p.add_argument("--matpc-type", default="odd_odd")
    p.add_argument("--ratio-force-mode", default="pc_quda_norm")
    p.add_argument("--cancel-force-mode", default="pc_quda_norm")
    p.add_argument("--cancel-constdet", default="yes", choices=["yes", "no"])
    p.add_argument(
        "--cancel-sample-scale",
        type=float,
        default=1.0,
        help="cancel refresh 的额外缩放；ChromaRNG.fill_eta 已包含 sqrt(0.5)，默认不再重复缩放。",
    )
    p.add_argument("--monomial-subset", default="full")
    p.add_argument("--strange-matpc-type", default="even_even_asymmetric",
                   choices=["even_even_asymmetric", "odd_odd_asymmetric", "even_even", "odd_odd"])
    p.add_argument("--strange-action-rational", action="store_true")
    p.add_argument("--use-chroma-accept", action="store_true")
    p.add_argument("--force-monitor", action="store_true")
    p.add_argument("--dump-modes", default="uniforms,eta,mom", help="plan B 参考_dump 模式")
    p.add_argument("--run-id", required=True)
    return p.parse_args()


def build_config(args: argparse.Namespace) -> dict:
    runner_args = SimpleNamespace(
        cfg=args.cfg,
        gauge_path=args.gauge_path,
        t_boundary=args.t_boundary,
        trajectory_length=args.traj_length,
        chroma_traj_length=args.chroma_traj_length,
        stop=args.ntraj or 1,
        start=0,
        warm=0,
        seed=args.seed,
        run_id=args.run_id,
        tol=args.tol,
        nsteps_inner=args.nsteps_inner,
        nsteps_outer=args.nsteps_outer,
        placement=args.placement,
        integrator_inner=args.integrator_inner,
        integrator_outer=args.integrator_outer,
        hasenbusch_mu=args.hasenbusch_mu,
        hasenbusch_force_sign=1.0,
        hasenbusch_force_coeff_scale=1.0,
        hasenbusch_force_output_scale=1.0,
        sym_odd_clover_scale=1.0,
        hasenbusch_force_dagger="yes",
        monomial_subset=args.monomial_subset,
        ratio_force_source="phi",
        ratio_force_mode=args.ratio_force_mode,
        cancel_force_mode=args.cancel_force_mode,
        cancel_sample_scale=args.cancel_sample_scale,
        cancel_constdet=args.cancel_constdet,
        matpc_type=args.matpc_type,
        stout_dim=4,
        stout_rho=0.125,
        stout_n_smear=1,
    )
    config = hr.base_config(runner_args)
    config["ratio_index"] = getattr(args, "ratio_index", -1)
    # base_config 来自 12x32 runner；24x64 及其他体积必须显式覆盖，避免读错质量。
    config["latt_size"] = list(args.latt)
    config["lattice"] = "x".join(str(x) for x in args.latt)
    if args.ml_bare is not None:
        config["ml_bare"] = args.ml_bare
    if args.ms_bare is not None:
        config["ms_bare"] = args.ms_bare
    # 重新按实际质量计算 Chroma SEOPREC 对齐的 Clover 系数。
    v0 = 0.951479
    config["clover_csw"] = (1 / (2 * (config["ml_bare"] + 4))) / v0**3
    config["clover_csw_strange"] = (1 / (2 * (config["ms_bare"] + 4))) / v0**3
    if args.maxiter is not None:
        config["maxiter"] = args.maxiter
    config["strange_matpc_type"] = args.strange_matpc_type
    config["strange_action_rational"] = args.strange_action_rational
    return config


def setup_logger(run_id: str) -> logging.Logger:
    logger = logging.getLogger(run_id)
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(f"{run_id} - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
    return logger


def main() -> int:
    args = parse_args()
    logger = setup_logger(args.run_id)
    outdir = Path(SCRIPT_DIR).parent / "rng_results"
    outdir.mkdir(parents=True, exist_ok=True)

    config = build_config(args)
    ntraj = args.ntraj if args.ntraj is not None else (1 if args.plan == "B" else 10)

    from pyquda_utils import core, io
    from pyquda.hmc import HMC
    from pyquda.chroma_rng import ChromaRNG

    cache = outdir / ".cache" / args.run_id
    cache.mkdir(parents=True, exist_ok=True)
    core.init(None, config["latt_size"], resource_path=str(cache), enable_force_monitor=args.force_monitor)
    latt_info = core.LatticeInfo(config["latt_size"], t_boundary=config["t_boundary"], anisotropy=1.0)
    hmc: HMC = hr.make_hmc(latt_info, config, logger)

    gauge = io.readQIOGauge(str(config["gauge_path"]))
    hmc.initialize(args.seed, gauge)

    handle = None
    if args.arm == "chroma":
        from chroma_rng_adapter import attach_chroma_rng

        handle = attach_chroma_rng(
            hmc,
            seed=args.seed,
            precision=args.rng_precision,
            use_chroma_accept=args.use_chroma_accept,
        )
        logger.info(f"chroma RNG attached: precision={args.rng_precision}, "
                    f"use_chroma_accept={args.use_chroma_accept}")

    # 方案 B：先输出参考 dump（独立 RNG 实例，不消耗 trajectory 流）
    if args.plan == "B":
        ref_rng = ChromaRNG(tuple(config["latt_size"]), precision=args.rng_precision)
        for mode in [m.strip() for m in args.dump_modes.split(",") if m.strip()]:
            path = outdir / f"{args.run_id}_pyquda_{mode}.txt"
            ref_rng.dump_reference(str(path), args.seed, mode)
            logger.info(f"reference dump written: {path}")

    records = []
    plaq0 = hmc.plaquette()
    logger.info(f"initial plaquette = {plaq0}")
    for i in range(ntraj):
        hmc.gaussMom()
        hmc.samplePhi()
        old = hr.get_actions(hmc)
        hmc.integrate(config["t"], 2e-15)
        new = hr.get_actions(hmc)
        delta_e = new["E"] - old["E"]
        accept = hmc.accept(delta_e)
        if accept:
            hmc.saveGauge(gauge)
        else:
            hmc.loadGauge(gauge)
        plaq = hmc.plaquette()
        rec = {
            "traj": i + 1,
            "K_old": old["K"], "P_gauge_old": old["P_gauge"], "P_fermion_old": old["P_fermion"],
            "K_new": new["K"], "P_gauge_new": new["P_gauge"], "P_fermion_new": new["P_fermion"],
            "delta_H": delta_e,
            "accept": bool(accept),
            "plaquette": plaq,
        }
        records.append(rec)
        logger.info(
            f"Traj {i + 1}: dH = {delta_e:.10g}, accept = {accept}, "
            f"plaquette_first = {plaq[0]:.10g}, plaquette = {plaq}"
        )

    result = {
        "run_id": args.run_id,
        "plan": args.plan,
        "arm": args.arm,
        "seed": args.seed,
        "rng_precision": args.rng_precision,
        "use_chroma_accept": args.use_chroma_accept,
        "latt_size": config["latt_size"],
        "ntraj": ntraj,
        "trajectory_length": config["t"],
        "nsteps_outer": config["nsteps_outer"],
        "nsteps_inner": config["nsteps_inner"],
        "integrator_outer": config["integrator_outer"],
        "integrator_inner": config["integrator_inner"],
        "matpc_type": config["matpc_type"],
        "hasenbusch_mu": config["hasenbusch_mu"],
        "initial_plaquette": plaq0,
        "records": records,
        # 记录完整配置，保证 11.1 的机器可读比较条件可从结果文件直接追溯。
        "run_config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in config.items()
        },
    }
    out = outdir / f"{args.run_id}.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"records written: {out}")
    if handle is not None:
        handle.detach()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
