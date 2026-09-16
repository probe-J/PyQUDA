#!/usr/bin/env python3
"""12x32 stout-smear Hasenbusch HMC 实验 driver。

主要输入：12x32 QIO/LIME 组态、Hasenbusch shift 链、trajectory length、积分器步数。
主要输出：hasenbusch_operation/logs 下的逐 trajectory 日志，用于解析 plaquette 第一分量、
Delta_E、接受/拒绝情况。脚本只在 hasenbusch_operation 内写日志和 cache。
"""

import argparse
import logging
import math
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from time import perf_counter


OP_DIR = Path(__file__).resolve().parents[1]
TESTS_DIR = OP_DIR.parent
PROJECT_DIR = TESTS_DIR.parent
GAUGE_DIR = Path("/home/jyg/stout_smear/test_stout_smesr/2ml_1ms_lime")
LOG_DIR = OP_DIR / "logs"
CACHE_ROOT = OP_DIR / ".cache" / "hasenbusch"

sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(TESTS_DIR))

from pyquda.action import CloverWilsonAction, GaugeAction
from pyquda.enum_quda import QudaVerbosity
from pyquda.hasenbusch import (
    CloverHasenbuschActions,
    CloverHasenbuschConfig,
    build_clover_hasenbusch_actions,
)
from pyquda.hmc import HMC, O2Nf1Ng0V, O2Nf2Ng0V, O4Nf5Ng0V, STSForceGrad, STSLeapfrog
from pyquda_utils import core, io
from pyquda_utils.hmc_param import fermionRationalParam as rationalParam
from pyquda_utils.hmc_param import symanzikOneLoopGaugeLoopParam as loopParam


def mpi_rank_from_env() -> int:
    for name in ("OMPI_COMM_WORLD_RANK", "PMIX_RANK", "PMI_RANK", "MV2_COMM_WORLD_RANK"):
        if name in os.environ:
            return int(os.environ[name])
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", choices=["1000", "10001"], required=True)
    parser.add_argument("--gauge-path", default=None)
    parser.add_argument("--t-boundary", type=int, default=-1, choices=[1, -1])
    parser.add_argument("--trajectory-length", type=float, default=1.0)
    parser.add_argument(
        "--chroma-traj-length",
        type=float,
        default=None,
        help="按 Chroma tau0 指定轨迹长度；内部自动除以 sqrt(2) 得到 PyQUDA t。",
    )
    parser.add_argument("--stop", type=int, default=5)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--warm", type=int, default=0)
    parser.add_argument("--seed", type=int, default=10086)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--nsteps-inner", type=int, default=2)
    parser.add_argument("--nsteps-outer", type=int, default=7)
    parser.add_argument("--stout-dim", type=int, choices=[3, 4], default=4)
    parser.add_argument("--stout-rho", type=float, default=0.125)
    parser.add_argument("--stout-n-smear", type=int, default=1)
    parser.add_argument(
        "--placement",
        choices=["cancel_gauge_inner", "cancel_inner", "all_fermion_outer", "all_outer"],
        default="cancel_gauge_inner",
        help=(
            "Hasenbusch monomial 布局：cancel_gauge_inner 对齐 Chroma 生产 XML；"
            "cancel_inner 为旧诊断布局；"
            "all_fermion_outer 只把 gauge 放 inner；all_outer 不使用嵌套 HMC。"
        ),
    )
    parser.add_argument(
        "--integrator-inner",
        choices=["STSForceGrad", "O2Nf1Ng0V", "O2Nf2Ng0V", "O4Nf5Ng0V"],
        default="STSForceGrad",
        help="inner HMC 使用的积分器；all_outer 布局不会使用该参数。",
    )
    parser.add_argument(
        "--integrator-outer",
        choices=["STSForceGrad", "O2Nf1Ng0V", "O2Nf2Ng0V", "O4Nf5Ng0V"],
        default="STSForceGrad",
        help="outer HMC 或非嵌套 HMC 使用的积分器。",
    )
    parser.add_argument(
        "--hasenbusch-mu",
        default="0,0.0005,0.0015,0.0035,0.0075,0.0155",
        help="Hasenbusch shift 链；首项必须为 0，才能望远镜分解回目标轻夸克 determinant。",
    )
    parser.add_argument(
        "--hasenbusch-force-sign",
        type=float,
        default=1.0,
        choices=[-1.0, 1.0],
        help="Hasenbusch force kernel sign probe. Default +1 keeps the QUDA force-test convention.",
    )
    parser.add_argument(
        "--hasenbusch-force-coeff-scale",
        type=float,
        default=1.0,
        help="Hasenbusch force coeff 的线性倍率探针；默认 1.0 保持 4*kappa^2 约定。",
    )
    parser.add_argument(
        "--hasenbusch-force-output-scale",
        type=float,
        default=1.0,
        help="Hasenbusch 单个 monomial general-link force 的整体倍率探针；默认 1.0。",
    )
    parser.add_argument(
        "--sym-odd-clover-scale",
        type=float,
        default=1.0,
        help="诊断用：只缩放 symmetric odd-odd Chroma 展开中的 active odd clover 导数项。",
    )
    parser.add_argument(
        "--hasenbusch-force-dagger",
        choices=["yes", "no"],
        default="yes",
        help="Hasenbusch force kernel 的 dagger 探针；默认 yes 保持当前实现。",
    )
    parser.add_argument(
        "--monomial-subset",
        choices=["full", "gauge_only", "no_ratio", "no_cancel", "cancel_only", "ratio_only", "strange_only", "logdet_only"],
        default="full",
        help="调试用 monomial 子集；默认 full，其他选项仅用于隔离 force/action 问题。",
    )
    parser.add_argument(
        "--ratio-force-source",
        choices=["phi"],
        default="phi",
        help="QUDA detratio 的 x0 固定为 ratio 源 pseudofermion phi。",
    )
    parser.add_argument(
        "--ratio-force-mode",
        choices=["quda_detratio", "pair", "pc_pair", "pc_quda_twist", "pc_quda_norm"],
        default="pc_quda_norm",
        help="ratio force 实现：默认使用已通过分支有限差分的 pc_quda_norm；其余模式仅用于诊断。",
    )
    parser.add_argument(
        "--cancel-force-mode",
        choices=["quda_tmclover", "pc_pair", "pc_constdet", "pc_quda_twist", "pc_quda_norm"],
        default="pc_quda_norm",
        help=(
            "cancel force 实现：默认 QUDA TMClover；pc_pair 完整 PC 导数；"
            "pc_constdet 只含 Chroma const-det D-link；pc_quda_twist 补 QUDA A^2 twist；"
            "pc_quda_norm 使用 QUDA MatPC 的 -kappa^2 与 ±mu/2 归一化。"
        ),
    )
    parser.add_argument(
        "--cancel-sample-scale",
        type=float,
        default=1.0,
        help="cancel refresh 后 phi 的整体倍率；Chroma multihasen cancel 使用 sqrt(0.5)。",
    )
    parser.add_argument(
        "--cancel-constdet",
        choices=["yes", "no"],
        default="yes",
        help="是否按 Chroma const-det cancel 去掉 clover trlog/sigma-trace action 贡献。",
    )
    parser.add_argument(
        "--matpc-type",
        choices=["even_even_asymmetric", "odd_odd_asymmetric", "even_even", "odd_odd"],
        default="odd_odd",
        help="Hasenbusch shifted-clover MatPC 类型；默认 odd_odd 对齐当前已验证的 Chroma SEOPREC 路径。",
    )
    return parser.parse_args()


def base_config(args: argparse.Namespace) -> dict:
    u_0, v_0 = 0.855453, 0.951479
    default_gauge = GAUGE_DIR / f"beta6.20_mu-0.2770_ms-0.2400_L12x32_cfg_{args.cfg}.lime"
    ml_bare, ms_bare = -0.2770, -0.2400
    kappa_light = 1 / (2 * (ml_bare + 4))
    kappa_strange = 1 / (2 * (ms_bare + 4))
    clover_csw_light = kappa_light / v_0**3
    clover_csw_strange = kappa_strange / v_0**3
    chroma_traj_length = getattr(args, "chroma_traj_length", None)
    trajectory_length = (
        chroma_traj_length / math.sqrt(2.0)
        if chroma_traj_length is not None
        else args.trajectory_length
    )
    return {
        "lattice": "12x32",
        "variant": "hasenbusch_twist_light_plus_strange",
        "cfg": args.cfg,
        "latt_size": [12, 12, 12, 32],
        "beta": 6.2,
        "u_0": u_0,
        "v_0": v_0,
        "ml_bare": ml_bare,
        "ms_bare": ms_bare,
        "clover_csw": clover_csw_light,
        "clover_csw_strange": clover_csw_strange,
        "tol": getattr(args, "tol", 1e-6),
        "maxiter": 1000,
        "start": args.start,
        "stop": args.stop,
        "warm": args.warm,
        "save": 5,
        "seed": args.seed,
        "t": trajectory_length,
        "t_boundary": args.t_boundary,
        "nsteps_inner": args.nsteps_inner,
        "nsteps_outer": args.nsteps_outer,
        "stout_n_smear": getattr(args, "stout_n_smear", 1),
        "stout_rho": getattr(args, "stout_rho", 0.125),
        "stout_dim": getattr(args, "stout_dim", 4),
        "stout_dir_ignore": 3 if getattr(args, "stout_dim", 4) == 3 else 4,
        "placement": args.placement,
        "integrator_inner": args.integrator_inner,
        "integrator_outer": args.integrator_outer,
        "gauge_path": Path(args.gauge_path or default_gauge),
        "run_id": args.run_id,
        "hasenbusch_mu": [float(item) for item in args.hasenbusch_mu.split(",") if item.strip()],
        "hasenbusch_force_sign": args.hasenbusch_force_sign,
        "hasenbusch_force_coeff_scale": args.hasenbusch_force_coeff_scale,
        "hasenbusch_force_output_scale": args.hasenbusch_force_output_scale,
        "sym_odd_clover_scale": args.sym_odd_clover_scale,
        "hasenbusch_force_dagger": args.hasenbusch_force_dagger,
        "monomial_subset": args.monomial_subset,
        "ratio_force_source": args.ratio_force_source,
        "ratio_force_mode": args.ratio_force_mode,
        "cancel_force_mode": args.cancel_force_mode,
        "cancel_sample_scale": args.cancel_sample_scale,
        "cancel_constdet": args.cancel_constdet == "yes",
        "matpc_type": args.matpc_type,
    }


def setup_logger(config: dict, rank: int) -> tuple[logging.Logger, Path, str]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"{config['run_id']}_{timestamp}.log"
    logger = logging.getLogger(config["run_id"])
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.INFO)
    if rank == 0:
        handler: logging.Handler = logging.FileHandler(log_file, encoding="utf-8")
    else:
        handler = logging.NullHandler()
    handler.setFormatter(logging.Formatter("%(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
    return logger, log_file, timestamp


def log_config(logger: logging.Logger, config: dict, log_file: Path, timestamp: str) -> None:
    fields = [
        ("timestamp", timestamp),
        ("log_file", str(log_file)),
        ("project_dir", str(PROJECT_DIR)),
        ("operation_dir", str(OP_DIR)),
        ("lattice", config["lattice"]),
        ("variant", config["variant"]),
        ("run_id", config["run_id"]),
        ("cfg", config["cfg"]),
        ("latt_size", config["latt_size"]),
        ("beta", config["beta"]),
        ("u_0", config["u_0"]),
        ("v_0", config["v_0"]),
        ("ml_bare", config["ml_bare"]),
        ("ms_bare", config["ms_bare"]),
        ("clover_csw", config["clover_csw"]),
        ("t_boundary", config["t_boundary"]),
        ("start", config["start"]),
        ("stop", config["stop"]),
        ("warm", config["warm"]),
        ("save", config["save"]),
        ("seed", config["seed"]),
        ("nsteps_outer", config["nsteps_outer"]),
        ("nsteps_inner", config["nsteps_inner"]),
        ("placement", config["placement"]),
        ("integrator_outer", config["integrator_outer"]),
        ("integrator_inner", config["integrator_inner"]),
        ("trajectory_length", config["t"]),
        ("stout_n_smear", config["stout_n_smear"]),
        ("stout_rho", config["stout_rho"]),
        ("stout_dim", config["stout_dim"]),
        ("stout_dir_ignore", config["stout_dir_ignore"]),
        ("gauge_path", config["gauge_path"]),
        ("hasenbusch_mu", config["hasenbusch_mu"]),
        ("hasenbusch_force_sign", config["hasenbusch_force_sign"]),
        ("hasenbusch_force_coeff_scale", config["hasenbusch_force_coeff_scale"]),
        ("hasenbusch_force_output_scale", config["hasenbusch_force_output_scale"]),
        ("sym_odd_clover_scale", config["sym_odd_clover_scale"]),
        ("hasenbusch_force_dagger", config["hasenbusch_force_dagger"]),
        ("monomial_subset", config["monomial_subset"]),
        ("ratio_force_source", config["ratio_force_source"]),
        ("ratio_force_mode", config["ratio_force_mode"]),
        ("cancel_force_mode", config["cancel_force_mode"]),
        ("cancel_sample_scale", config["cancel_sample_scale"]),
        ("cancel_constdet", config["cancel_constdet"]),
        ("matpc_type", config["matpc_type"]),
    ]
    logger.info("=" * 72)
    logger.info("HMC stout-smear Hasenbusch run configuration")
    logger.info("=" * 72)
    for key, value in fields:
        logger.info(f"{key} = {value}")
    logger.info("loop_param = symanzikOneLoopGaugeLoopParam(u_0, fermion_type=\"quenched\")")
    logger.info(
        "stout = "
        f"n_smear={config['stout_n_smear']} rho={config['stout_rho']} "
        f"dim={config['stout_dim']} dir_ignore={config['stout_dir_ignore']}"
    )
    logger.info(
        "integrator = "
        f"placement={config['placement']}, "
        f"outer={config['integrator_outer']}({config['nsteps_outer']}), "
        f"inner={config['integrator_inner']}({config['nsteps_inner']})"
    )
    logger.info("integrator_note = STSForceGrad implements Chroma LCM_STS_FORCE_GRAD shifted-force sequence")
    logger.info("light_logdet = Chroma-style -2*(trlogA_even + trlogA_odd)")
    logger.info("light_has_cancel = determinant action at largest Hasenbusch mu")
    logger.info("light_has_ratio = adjacent shift-chain ratio actions M(mu_i) / M(mu_{i+1})")
    logger.info("strange = CloverWilsonAction rationalParam(1, 12, 15, 7e-4, 32, 50)")
    logger.info("=" * 72)


def make_action_groups(latt_info, config: dict):
    # 12x32 runner 只保留诊断子集选择；完整 Hasenbusch action 构造复用 PyQUDA 工厂。
    gauge_action = GaugeAction(
        latt_info,
        loopParam(config["u_0"], fermion_type="quenched"),
        config["beta"],
    )
    hasenbusch = build_clover_hasenbusch_actions(
        latt_info,
        CloverHasenbuschConfig(
            mass=config["ml_bare"],
            shifts=config["hasenbusch_mu"],
            clover_csw=config["clover_csw"],
            tol=config["tol"],
            md_tol=config["tol"],
            action_tol=config["tol"],
            maxiter=config["maxiter"],
            matpc_type=config["matpc_type"],
            ratio_force_mode=config["ratio_force_mode"],
            cancel_force_mode=config["cancel_force_mode"],
            ratio_force_source=config["ratio_force_source"],
            cancel_constdet=config["cancel_constdet"],
            cancel_sample_scale=config["cancel_sample_scale"],
            force_sign=config["hasenbusch_force_sign"],
            force_coeff_scale=config["hasenbusch_force_coeff_scale"],
            force_output_scale=config["hasenbusch_force_output_scale"],
            force_dagger=config["hasenbusch_force_dagger"],
            sym_odd_clover_scale=config["sym_odd_clover_scale"],
            mu_scale=config.get("mu_scale"),
        ),
    )
    ratio_index = int(config.get("ratio_index", -1))
    if ratio_index >= 0:
        if ratio_index >= len(hasenbusch.ratios):
            raise ValueError(f"ratio_index {ratio_index} out of range")
        hasenbusch = CloverHasenbuschActions(
            logdet=hasenbusch.logdet,
            ratios=[hasenbusch.ratios[ratio_index]],
            cancel=hasenbusch.cancel,
        )
    strange = CloverWilsonAction(
        latt_info,
        rationalParam(1, 12, 15, 7e-4, 32, 50),
        config["ms_bare"],
        1,
        config["tol"],
        config["maxiter"],
        config["clover_csw_strange"],
        use_action_rational=config.get("strange_action_rational", False),
        matpc_type=config.get("strange_matpc_type", "even_even_asymmetric"),
    )
    return gauge_action, hasenbusch.logdet, hasenbusch.cancel, hasenbusch.ratios, strange


INTEGRATORS = {
    "STSForceGrad": STSForceGrad,
    "STSLeapfrog": STSLeapfrog,
    "O2Nf1Ng0V": O2Nf1Ng0V,
    "O2Nf2Ng0V": O2Nf2Ng0V,
    "O4Nf5Ng0V": O4Nf5Ng0V,
}


def make_integrator(name: str, n_steps: int):
    # 积分器名称写入日志，避免后续只看到步数而无法复现实验。
    return INTEGRATORS[name](n_steps)


def make_hmc(latt_info, config: dict, logger: logging.Logger) -> HMC:
    gauge_action, logdet, has_cancel, has_ratio, strange = make_action_groups(latt_info, config)
    placement = config["placement"]
    subset = config["monomial_subset"]
    include_logdet = subset in ("full", "no_ratio", "no_cancel", "logdet_only")
    include_cancel = subset in ("full", "no_ratio", "cancel_only")
    include_ratio = subset in ("full", "no_cancel", "ratio_only")
    include_strange = subset in ("full", "no_ratio", "no_cancel", "strange_only")
    if subset == "gauge_only":
        include_logdet = include_cancel = include_ratio = include_strange = False
    logdet_terms = [logdet] if include_logdet else []
    cancel_terms = [has_cancel] if include_cancel else []
    ratio_terms = [*has_ratio] if include_ratio else []
    strange_terms = [strange] if include_strange else []
    if placement == "cancel_gauge_inner":
        inner_monomials = [gauge_action, *cancel_terms]
        outer_monomials = [*logdet_terms, *ratio_terms, *strange_terms]
    elif placement == "cancel_inner":
        inner_monomials = [gauge_action, *logdet_terms, *cancel_terms]
        outer_monomials = [*ratio_terms, *strange_terms]
    elif placement == "all_fermion_outer":
        inner_monomials = [gauge_action]
        outer_monomials = [*logdet_terms, *cancel_terms, *ratio_terms, *strange_terms]
    elif placement == "all_outer":
        inner_monomials = []
        outer_monomials = [gauge_action, *logdet_terms, *cancel_terms, *ratio_terms, *strange_terms]
    else:
        raise ValueError(f"Unknown placement: {placement}")
    if include_logdet and include_cancel and not config["cancel_constdet"]:
        raise ValueError("independent logdet requires cancel_constdet=yes to avoid double counting")

    stout_kwargs = {
        "stout_n_smear": config["stout_n_smear"],
        "stout_rho": config["stout_rho"],
        "stout_dir_ignore": config["stout_dir_ignore"],
    }

    logger.info(f"inner_monomial_count = {len(inner_monomials)}")
    logger.info(f"outer_monomial_count = {len(outer_monomials)}")
    logger.info(f"monomial_count = {len(inner_monomials) + len(outer_monomials)}")
    logger.info(f"fermion_monomial_count = {len(inner_monomials) + len(outer_monomials) - 1}")
    logger.info(
        "monomial_layout = "
        f"inner={[type(item).__name__ for item in inner_monomials]}, "
        f"outer={[getattr(item, 'label', type(item).__name__) for item in outer_monomials]}"
    )

    outer_integrator = make_integrator(config["integrator_outer"], config["nsteps_outer"])
    # Chroma XML 的有效 RNG 消费顺序是 ratio -> cancel -> strange；
    # logdet 不消耗 RNG。adapter 会按这个顺序预生成 Gaussian 场。
    chroma_sampling_order = [*ratio_terms, *cancel_terms, *strange_terms]
    if placement == "all_outer":
        hmc = HMC(latt_info, outer_monomials, outer_integrator, **stout_kwargs)
        hmc.chroma_sampling_order = chroma_sampling_order
        return hmc

    inner_integrator = make_integrator(config["integrator_inner"], config["nsteps_inner"])
    hmc_inner = HMC(latt_info, inner_monomials, inner_integrator, **stout_kwargs)
    hmc = HMC(latt_info, outer_monomials, outer_integrator, hmc_inner, **stout_kwargs)
    hmc.chroma_sampling_order = chroma_sampling_order
    return hmc


def get_actions(hmc: HMC) -> dict:
    kinetic = hmc.momAction()
    gauge = hmc.gaugeAction()
    fermion = hmc.fermionAction()
    return {
        "K": kinetic,
        "P_gauge": gauge,
        "P_fermion": fermion,
        "P": gauge + fermion,
        "E": kinetic + gauge + fermion,
    }


def log_trajectory(logger, idx, plaq, old, new, delta_e, accept, warmup, accepted, rejected, elapsed):
    logger.info(
        f"Trajectory {idx}:\n"
        f"Plaquette = {plaq}\n"
        f"P_gauge_old = {old['P_gauge']}, P_fermion_old = {old['P_fermion']}, K_old = {old['K']}\n"
        f"P_gauge = {new['P_gauge']}, P_fermion = {new['P_fermion']}, K = {new['K']}\n"
        f"Delta_P_gauge = {new['P_gauge'] - old['P_gauge']}, "
        f"Delta_P_fermion = {new['P_fermion'] - old['P_fermion']}, "
        f"Delta_K = {new['K'] - old['K']}\n"
        f"P_old = {old['P']}, P = {new['P']}\n"
        f"Delta_P = {new['P'] - old['P']}, Delta_E = {delta_e}\n"
        f"acceptance rate = {math.exp(min(-delta_e, 0)) * 100:.2f}%\n"
        f"accept? {accept}\n"
        f"warmup? {warmup}\n"
        f"accepted_so_far = {accepted}, rejected_so_far = {rejected}\n"
        f"HMC time = {elapsed:.3f} secs\n"
    )


def main() -> int:
    args = parse_args()
    rank = mpi_rank_from_env()
    config = base_config(args)
    logger, log_file, timestamp = setup_logger(config, rank)
    log_config(logger, config, log_file, timestamp)
    run_cache = CACHE_ROOT / config["run_id"]
    run_cache.mkdir(parents=True, exist_ok=True)
    try:
        core.init(None, config["latt_size"], resource_path=str(run_cache), enable_force_monitor=True)
        latt_info = core.LatticeInfo(config["latt_size"], t_boundary=config["t_boundary"], anisotropy=1.0)
        hmc = make_hmc(latt_info, config, logger)
        hmc.setFermionVerbosity(QudaVerbosity.QUDA_SILENT)
        gauge = io.readQIOGauge(str(config["gauge_path"]))
        hmc.initialize(config["seed"], gauge)
        plaq = hmc.plaquette()
        logger.info(f"Trajectory {config['start']}:\nPlaquette = {plaq}\n")

        accepted = rejected = 0
        for i in range(config["start"], config["stop"]):
            start_time = perf_counter()
            hmc.gaussMom()
            hmc.samplePhi()
            old = get_actions(hmc)
            hmc.integrate(config["t"], 2e-15)
            new = get_actions(hmc)
            delta_e = new["E"] - old["E"]
            accept = hmc.accept(delta_e)
            warmup = i < config["warm"]
            if accept or warmup:
                hmc.saveGauge(gauge)
                accepted += int(accept)
            else:
                hmc.loadGauge(gauge)
                rejected += 1
            plaq = hmc.plaquette()
            log_trajectory(
                logger,
                i + 1,
                plaq,
                old,
                new,
                delta_e,
                accept,
                warmup,
                accepted,
                rejected,
                perf_counter() - start_time,
            )

        logger.info("Run status = ok")
        logger.info(
            "Run summary:\n"
            f"final Plaquette = {plaq}\n"
            f"accepted = {accepted}, rejected = {rejected}, total = {config['stop'] - config['start']}\n"
        )
        if rank == 0:
            print(log_file)
        return 0
    except Exception as exc:
        logger.info("Run status = failed")
        logger.info(f"Failure type = {type(exc).__name__}")
        logger.info(f"Failure reason = {exc}")
        logger.info(traceback.format_exc())
        if rank == 0:
            print(log_file)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
