#!/usr/bin/env python3
"""12x32 Hasenbusch force finite-difference 检查。

用途：固定 pseudofermion 后，沿随机 momentum 方向比较
  [S(U(+eps)) - S(U(-eps))] / (2 eps)
和 PyQUDA/QUDA force_v2 返回的 momentum 更新方向内积。

主要输入：cfg、Hasenbusch shift 链、monomial subset、eps 列表。
主要输出：hasenbusch_operation/csv/force_check_*.csv 与 logs/force_check_*.log。
环境前提：运行前 source set_PyQUDA_env.sh；脚本只写 hasenbusch_operation 内部目录。
"""

import argparse
import csv
import logging
import math
import sys
from pathlib import Path
from types import SimpleNamespace


OP_DIR = Path(__file__).resolve().parents[1]
SCRIPT_DIR = OP_DIR / "scripts"
LOG_DIR = OP_DIR / "logs"
CSV_DIR = OP_DIR / "csv"
CACHE_ROOT = OP_DIR / ".cache" / "force_check"
TESTS_DIR = OP_DIR.parent
PROJECT_DIR = TESTS_DIR.parent

sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(TESTS_DIR))

from pyquda.enum_quda import QudaVerbosity
from pyquda.field import LatticeGauge, LatticeMom
from pyquda_utils import core, io

import run_hmc_hasenbusch_12x32 as hmc_runner
import run_rng_compare as rr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", choices=["1000", "10001"], default="1000")
    parser.add_argument("--t-boundary", type=int, default=-1, choices=[-1, 1])
    parser.add_argument("--seed", type=int, default=10086)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--gauge-path", default=None)
    parser.add_argument("--latt", type=int, nargs=4, default=[24, 24, 24, 64])
    parser.add_argument("--ml-bare", type=float, default=-0.2790)
    parser.add_argument("--ms-bare", type=float, default=-0.2400)
    parser.add_argument("--ratio-index", type=int, default=-1, help="-1 表示全部 ratio；0 表示 ratio 1")
    parser.add_argument("--stout-dim", type=int, choices=[3, 4], default=4)
    parser.add_argument(
        "--hasenbusch-mu",
        default="0,0.0005,0.0015,0.0035,0.0075,0.0155",
        help="逗号分隔的 Hasenbusch shift 链。",
    )
    parser.add_argument(
        "--monomial-subset",
        choices=["full", "no_ratio", "no_cancel", "cancel_only", "ratio_only", "strange_only", "logdet_only"],
        default="full",
    )
    parser.add_argument(
        "--hasenbusch-force-sign",
        type=float,
        choices=[-1.0, 1.0],
        default=1.0,
        help="Hasenbusch force kernel 符号探针；普通 strange_only 不受影响。",
    )
    parser.add_argument(
        "--hasenbusch-force-coeff-scale",
        type=float,
        default=1.0,
        help="Hasenbusch force coeff 的线性倍率探针；默认保持 4*kappa^2 约定。",
    )
    parser.add_argument(
        "--hasenbusch-force-output-scale",
        type=float,
        default=1.0,
        help="Hasenbusch 单个 monomial general-link force 的整体倍率探针。",
    )
    parser.add_argument(
        "--hasenbusch-force-dagger",
        choices=["yes", "no"],
        default="yes",
        help="Hasenbusch force kernel dagger 探针。",
    )
    parser.add_argument(
        "--ratio-force-source",
        choices=["phi", "rhs"],
        default="phi",
        help="determinant-ratio force 的 h_x0 探针：phi 或 M(mu_prec)^dag phi。",
    )
    parser.add_argument(
        "--ratio-force-mode",
        choices=["quda_detratio", "pair", "pc_pair", "pc_quda_twist", "pc_quda_norm"],
        default="pc_quda_norm",
        help="ratio force 实现：默认 QUDA detratio；pc_pair/pc_quda_twist 为显式 PC 诊断；pc_quda_norm 使用 QUDA MatPC 对称归一化四项公式。",
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
        "--sym-odd-clover-scale",
        type=float,
        default=1.0,
        help="诊断用：只缩放 symmetric odd-odd Chroma 展开中的 active odd clover 导数项。",
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
        help="Hasenbusch shifted-clover MatPC 类型；Chroma SEOPREC 对齐优先测试 odd_odd。",
    )
    parser.add_argument(
        "--eps",
        default="1e-6,3e-6,1e-5,3e-5",
        help="逗号分隔的正负扰动步长列表。",
    )
    parser.add_argument(
        "--skip-force",
        action="store_true",
        help="只计算固定 pseudofermion 下的 action finite-difference，不调用 fermionForce。",
    )
    parser.add_argument(
        "--diagnose-force-terms",
        action="store_true",
        help="额外逐项计算 Hasenbusch bilinear force 对随机方向的内积。",
    )
    parser.add_argument(
        "--diagnose-monomial-fd",
        action="store_true",
        help="额外逐 monomial 计算 action finite-difference 和该 monomial force 投影。",
    )
    parser.add_argument(
        "--tol-override",
        type=float,
        default=None,
        help="诊断用覆盖 Dirac 求解容差；默认沿用 runner 配置。",
    )
    parser.add_argument(
        "--maxiter-override",
        type=int,
        default=None,
        help="诊断用覆盖最大迭代次数；默认沿用 runner 配置。",
    )
    parser.add_argument(
        "--clover-csw",
        type=float,
        default=None,
        help="诊断用覆盖 Clover 系数；用于验证 Chroma SEOPREC 归一化。",
    )
    parser.add_argument(
        "--strange-matpc-type",
        choices=["even_even_asymmetric", "odd_odd_asymmetric", "even_even", "odd_odd"],
        default="even_even_asymmetric",
    )
    parser.add_argument("--strange-action-rational", action="store_true")
    return parser.parse_args()


def make_config(args: argparse.Namespace) -> dict:
    # 复用 HMC runner 的参数口径，避免 force check 与实际 HMC probe 构造不同。
    runner_args = SimpleNamespace(
        plan="B",
        arm="chroma",
        latt=args.latt,
        cfg=args.cfg,
        gauge_path=args.gauge_path,
        ml_bare=args.ml_bare,
        ms_bare=args.ms_bare,
        t_boundary=args.t_boundary,
        rng_precision="double",
        ntraj=1,
        maxiter=5000,
        tol=1e-10,
        traj_length=1.0,
        chroma_traj_length=None,
        nsteps_inner=1,
        nsteps_outer=56,
        integrator_inner="O2Nf1Ng0V",
        integrator_outer="O2Nf1Ng0V",
        placement="all_outer",
        seed=args.seed,
        run_id=args.run_id,
        hasenbusch_mu=args.hasenbusch_mu,
        hasenbusch_force_sign=args.hasenbusch_force_sign,
        hasenbusch_force_coeff_scale=args.hasenbusch_force_coeff_scale,
        hasenbusch_force_output_scale=args.hasenbusch_force_output_scale,
        hasenbusch_force_dagger=args.hasenbusch_force_dagger,
        monomial_subset=args.monomial_subset,
        stout_dim=args.stout_dim,
        stout_rho=0.125,
        stout_n_smear=1,
        ratio_force_source=args.ratio_force_source,
        ratio_force_mode=args.ratio_force_mode,
        cancel_force_mode=args.cancel_force_mode,
        cancel_sample_scale=args.cancel_sample_scale,
        sym_odd_clover_scale=args.sym_odd_clover_scale,
        cancel_constdet=args.cancel_constdet,
        matpc_type=args.matpc_type,
        strange_matpc_type=args.strange_matpc_type,
        strange_action_rational=args.strange_action_rational,
        use_chroma_accept=False,
        dump_modes="",
    )
    config = rr.build_config(runner_args)
    if args.clover_csw is not None:
        config["clover_csw"] = args.clover_csw
    config["strange_matpc_type"] = args.strange_matpc_type
    config["strange_action_rational"] = args.strange_action_rational
    if args.ratio_index >= 0:
        config["monomial_subset"] = "ratio_only"
    if args.tol_override is not None:
        # 只用于有限差分诊断，避免求解残差被 eps 放大后误判 force 闭合。
        config["tol"] = args.tol_override
    if args.maxiter_override is not None:
        # 与 tol_override 配合使用；默认值保持原实验脚本不变。
        config["maxiter"] = args.maxiter_override
    return config


def setup_logger(run_id: str) -> tuple[logging.Logger, Path]:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"force_check_{run_id}.log"
    logger = logging.getLogger(run_id)
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(log_file, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    return logger, log_file


def action_at_shift(hmc, gauge0: LatticeGauge, direction: LatticeMom, eps: float) -> float:
    # 用 QUDA resident momentum 沿同一个方向扰动 gauge，再直接计算固定 phi 下的 fermion action。
    hmc.loadGauge(gauge0)
    hmc.loadMom(direction)
    hmc.updateGauge(eps)
    return hmc.fermionAction()


def monomial_action_at_shift(hmc, monomial, gauge0: LatticeGauge, direction: LatticeMom, eps: float) -> float:
    # 在同一组 pseudofermion 下，只计算单个 monomial 的 action 变化。
    hmc.loadGauge(gauge0)
    hmc.loadMom(direction)
    hmc.updateGauge(eps)
    hmc.loadGaugeMomSmeared()
    action = monomial.action()
    hmc.loadGaugeMom()
    return action


def monomial_force_projection(hmc, latt_info, monomial, gauge0: LatticeGauge, direction: LatticeMom) -> tuple[float, float]:
    # 单独执行一个 monomial 的 force_v2，经过 stout force 回传后与同一扰动方向做内积。
    zero_mom = LatticeMom(latt_info)
    force_mom = LatticeMom(latt_info)
    hmc.loadGauge(gauge0)
    hmc.loadMom(zero_mom)
    hmc.loadGaugeMomSmeared()
    monomial.force_v2(1.0, hmc.force_v2)
    hmc.loadGaugeMom()
    hmc.saveMom(force_mom)
    return kinetic_inner_product(hmc, latt_info, force_mom, direction), kinetic_norm(hmc, force_mom)


def copy_mom_like(latt_info, data) -> LatticeMom:
    mom = LatticeMom(latt_info)
    # 复制一份独立 momentum 数据，避免 plus/minus 临时对象和输入对象共享底层数组。
    mom.data = data.copy()
    return mom


def kinetic_inner_product(hmc, latt_info, a: LatticeMom, b: LatticeMom) -> float:
    # 使用 QUDA momAction 定义 momentum 度量，避免手写 LatticeMom 9 分量的权重/符号。
    # 对 K(P)=1/2<P,P>，K(a+b)-K(a-b)=2<a,b>。
    plus = copy_mom_like(latt_info, a.data + b.data)
    minus = copy_mom_like(latt_info, a.data - b.data)
    hmc.loadMom(plus)
    k_plus = hmc.momAction()
    hmc.loadMom(minus)
    k_minus = hmc.momAction()
    return (k_plus - k_minus) / 2.0


def kinetic_norm(hmc, mom: LatticeMom) -> float:
    # QUDA momAction 对 ASQTAD_MOM_LINKS 的 9 分量表示带有常数/符号偏移，
    # 不能直接当作正定范数；这里只用 host 数组范数判断方向是否非零。
    return math.sqrt(float((mom.data * mom.data).sum()))


def host_inner_product(a: LatticeMom, b: LatticeMom) -> float:
    # 仅作为诊断输出，实际 force 对比仍使用 kinetic_inner_product 的 momAction 差分度量。
    return float((a.data * b.data).sum())


def host_weighted_inner_product(a: LatticeMom, b: LatticeMom) -> float:
    # LatticeMom 的 9 分量表示为 3 个复非对角元加 3 个纯虚对角元；
    # 反厄米矩阵内积中非对角元同时出现在上下三角，因此权重为 2。
    offdiag = 2.0 * (a.data[..., :6] * b.data[..., :6]).sum()
    diag = (a.data[..., 6:] * b.data[..., 6:]).sum()
    return float(offdiag + diag)


def iter_fermion_monomials(hmc):
    # 同时遍历 outer 和 inner HMC，避免嵌套布局时漏掉内层 monomial。
    for monomial in getattr(hmc, "fermion_monomials", []):
        yield monomial
    inner = getattr(hmc, "hmc_inner", None)
    if inner is not None:
        yield from iter_fermion_monomials(inner)


def set_force_term_filter(hmc, term_label):
    # 仅 Hasenbusch action 暴露 set_force_term_filter；普通 strange action 保持原样。
    for monomial in iter_fermion_monomials(hmc):
        setter = getattr(monomial, "set_force_term_filter", None)
        if setter is not None:
            setter(term_label)


def unique_force_terms(hmc):
    # 从最近一次完整 force_v2 中收集 bilinear 项标签，保持首次出现顺序。
    terms = []
    seen = set()
    for monomial in iter_fermion_monomials(hmc):
        for item in getattr(monomial, "force_term_history", []):
            label = item["label"]
            if label in seen:
                continue
            seen.add(label)
            terms.append(dict(item))
    return terms


def write_force_term_csv(
    term_csv_file: Path,
    args: argparse.Namespace,
    config: dict,
    hmc,
    latt_info,
    gauge0: LatticeGauge,
    direction: LatticeMom,
    terms: list,
) -> None:
    # 对每个 bilinear 项单独走完整 fermionForce 管线，保证包含 stout force 回传和 momentum 投影。
    zero_mom = LatticeMom(latt_info)
    rows = []
    for idx, term in enumerate(terms):
        set_force_term_filter(hmc, term["label"])
        hmc.loadGauge(gauge0)
        hmc.loadMom(zero_mom)
        hmc.fermionForce(1.0)
        term_mom = LatticeMom(latt_info)
        hmc.saveMom(term_mom)
        rows.append(
            {
                "run_id": args.run_id,
                "cfg": config["cfg"],
                "subset": config["monomial_subset"],
                "term_index": idx,
                "term_label": term["label"],
                "raw_label": term["raw_label"],
                "sign": term["sign"],
                "include_clover": term["include_clover"],
                "include_hopping": term["include_hopping"],
                "force_output_scale": term["force_output_scale"],
                "coeff": term["coeff"],
                "epsilon_even": term["epsilon_even"],
                "epsilon_odd": term["epsilon_odd"],
                "force_dot_direction": kinetic_inner_product(hmc, latt_info, term_mom, direction),
                "host_force_dot_direction": host_inner_product(term_mom, direction),
                "host_weighted_force_dot_direction": host_weighted_inner_product(term_mom, direction),
                "force_norm": kinetic_norm(hmc, term_mom),
            }
        )
    set_force_term_filter(hmc, None)
    with term_csv_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run_id",
                "cfg",
                "subset",
                "term_index",
                "term_label",
                "raw_label",
                "sign",
                "include_clover",
                "include_hopping",
                "force_output_scale",
                "coeff",
                "epsilon_even",
                "epsilon_odd",
                "force_dot_direction",
                "host_force_dot_direction",
                "host_weighted_force_dot_direction",
                "force_norm",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_monomial_fd_csv(
    monomial_csv_file: Path,
    args: argparse.Namespace,
    config: dict,
    hmc,
    latt_info,
    gauge0: LatticeGauge,
    direction: LatticeMom,
    eps_values: list[float],
) -> None:
    # 用 full-chain sample 后的固定 pseudofermion，逐 monomial 对齐 action FD 与 force。
    rows = []
    for mono_index, monomial in enumerate(iter_fermion_monomials(hmc)):
        label = getattr(monomial, "label", f"monomial_{mono_index}")
        force_dot, force_norm = monomial_force_projection(hmc, latt_info, monomial, gauge0, direction)
        for eps in eps_values:
            s_plus = monomial_action_at_shift(hmc, monomial, gauge0, direction, eps)
            s_minus = monomial_action_at_shift(hmc, monomial, gauge0, direction, -eps)
            fd = (s_plus - s_minus) / (2 * eps)
            denom_opp = max(abs(fd), abs(-force_dot), 1.0)
            rows.append(
                {
                    "run_id": args.run_id,
                    "cfg": config["cfg"],
                    "subset": config["monomial_subset"],
                    "monomial_index": mono_index,
                    "monomial_label": label,
                    "eps": eps,
                    "S_plus": s_plus,
                    "S_minus": s_minus,
                    "finite_difference": fd,
                    "force_dot_direction": force_dot,
                    "fd_plus_force_dot": fd + force_dot,
                    "relative_error_opposite_sign": abs(fd + force_dot) / denom_opp,
                    "force_norm": force_norm,
                }
            )
    with monomial_csv_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run_id",
                "cfg",
                "subset",
                "monomial_index",
                "monomial_label",
                "eps",
                "S_plus",
                "S_minus",
                "finite_difference",
                "force_dot_direction",
                "fd_plus_force_dot",
                "relative_error_opposite_sign",
                "force_norm",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    config = make_config(args)
    eps_values = [float(item) for item in args.eps.split(",") if item.strip()]
    logger, log_file = setup_logger(args.run_id)
    csv_file = CSV_DIR / f"force_check_{args.run_id}.csv"
    term_csv_file = CSV_DIR / f"force_terms_{args.run_id}.csv"
    monomial_csv_file = CSV_DIR / f"monomial_fd_{args.run_id}.csv"
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)

    logger.info(f"run_id = {args.run_id}")
    logger.info(f"cfg = {config['cfg']}")
    logger.info(f"subset = {config['monomial_subset']}")
    logger.info(f"shift_chain = {config['hasenbusch_mu']}")
    logger.info(f"eps_values = {eps_values}")
    logger.info(f"tol = {config['tol']}")
    logger.info(f"maxiter = {config['maxiter']}")

    core.init(None, config["latt_size"], resource_path=str(CACHE_ROOT / args.run_id), enable_force_monitor=True)
    latt_info = core.LatticeInfo(config["latt_size"], t_boundary=config["t_boundary"], anisotropy=1.0)
    hmc = hmc_runner.make_hmc(latt_info, config, logger)
    hmc.setFermionVerbosity(QudaVerbosity.QUDA_SILENT)
    if args.ratio_index >= 0:
        ratio_counter = 0
        filtered = []
        for monomial in hmc.fermion_monomials:
            if getattr(monomial, "determinant_ratio", False):
                if ratio_counter == args.ratio_index:
                    filtered.append(monomial)
                ratio_counter += 1
        hmc.fermion_monomials = filtered

    gauge = io.readQIOGauge(str(config["gauge_path"]))
    hmc.initialize(config["seed"], gauge)
    hmc.samplePhi()

    gauge0 = LatticeGauge(latt_info)
    hmc.saveGauge(gauge0)

    hmc.gaussMom()
    direction = LatticeMom(latt_info)
    hmc.saveMom(direction)

    direction_norm = kinetic_norm(hmc, direction)
    if direction_norm <= 0:
        raise RuntimeError(
            "force check direction_norm <= 0; resident momentum flags or save/load mom are inconsistent"
        )
    if args.skip_force:
        # symmetric MatPC 先做 action-only probe；force 分支实现后再关闭 skip-force 做闭合检查。
        force_dot_direction = float("nan")
        host_force_dot_direction = float("nan")
        host_weighted_force_dot_direction = float("nan")
        force_norm = float("nan")
        logger.info("skip_force = true")
        logger.info(f"direction_norm = {direction_norm:.17g}")
    else:
        zero_mom = LatticeMom(latt_info)
        force_mom = LatticeMom(latt_info)
        set_force_term_filter(hmc, None)
        hmc.loadGauge(gauge0)
        hmc.loadMom(zero_mom)
        hmc.fermionForce(1.0)
        hmc.saveMom(force_mom)

        force_dot_direction = kinetic_inner_product(hmc, latt_info, force_mom, direction)
        host_force_dot_direction = host_inner_product(force_mom, direction)
        host_weighted_force_dot_direction = host_weighted_inner_product(force_mom, direction)
        force_norm = kinetic_norm(hmc, force_mom)
        logger.info(f"force_dot_direction = {force_dot_direction:.17g}")
        logger.info(f"host_force_dot_direction = {host_force_dot_direction:.17g}")
        logger.info(f"host_weighted_force_dot_direction = {host_weighted_force_dot_direction:.17g}")
        logger.info(f"direction_norm = {direction_norm:.17g}")
        logger.info(f"force_norm = {force_norm:.17g}")
        if args.diagnose_force_terms:
            terms = unique_force_terms(hmc)
            logger.info(f"force_term_count = {len(terms)}")
            write_force_term_csv(term_csv_file, args, config, hmc, latt_info, gauge0, direction, terms)
            logger.info(f"force_term_csv = {term_csv_file}")
        if args.diagnose_monomial_fd:
            write_monomial_fd_csv(monomial_csv_file, args, config, hmc, latt_info, gauge0, direction, eps_values)
            logger.info(f"monomial_fd_csv = {monomial_csv_file}")

    with csv_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run_id",
                "cfg",
                "subset",
                "shift_chain",
                "eps",
                "S_plus",
                "S_minus",
                "finite_difference",
                "force_dot_direction",
                "host_force_dot_direction",
                "host_weighted_force_dot_direction",
                "fd_minus_force_dot",
                "fd_plus_force_dot",
                "relative_error_same_sign",
                "relative_error_opposite_sign",
                "direction_norm",
                "force_norm",
            ],
        )
        writer.writeheader()
        for eps in eps_values:
            s_plus = action_at_shift(hmc, gauge0, direction, eps)
            s_minus = action_at_shift(hmc, gauge0, direction, -eps)
            finite_difference = (s_plus - s_minus) / (2 * eps)
            denom_same = max(abs(finite_difference), abs(force_dot_direction), 1.0)
            denom_opp = max(abs(finite_difference), abs(-force_dot_direction), 1.0)
            row = {
                "run_id": args.run_id,
                "cfg": config["cfg"],
                "subset": config["monomial_subset"],
                "shift_chain": ",".join(str(x) for x in config["hasenbusch_mu"]),
                "eps": eps,
                "S_plus": s_plus,
                "S_minus": s_minus,
                "finite_difference": finite_difference,
                "force_dot_direction": force_dot_direction,
                "host_force_dot_direction": host_force_dot_direction,
                "host_weighted_force_dot_direction": host_weighted_force_dot_direction,
                "fd_minus_force_dot": finite_difference - force_dot_direction,
                "fd_plus_force_dot": finite_difference + force_dot_direction,
                "relative_error_same_sign": abs(finite_difference - force_dot_direction) / denom_same,
                "relative_error_opposite_sign": abs(finite_difference + force_dot_direction) / denom_opp,
                "direction_norm": direction_norm,
                "force_norm": force_norm,
            }
            writer.writerow(row)
            logger.info(
                "eps={eps:.3e} S_plus={sp:.17g} S_minus={sm:.17g} "
                "fd={fd:.17g} fd-force={df:.17g} fd+force={pf:.17g}".format(
                    eps=eps,
                    sp=s_plus,
                    sm=s_minus,
                    fd=finite_difference,
                    df=finite_difference - force_dot_direction,
                    pf=finite_difference + force_dot_direction,
                )
            )

    print(csv_file)
    if args.diagnose_force_terms and not args.skip_force:
        print(term_csv_file)
    if args.diagnose_monomial_fd and not args.skip_force:
        print(monomial_csv_file)
    print(log_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
