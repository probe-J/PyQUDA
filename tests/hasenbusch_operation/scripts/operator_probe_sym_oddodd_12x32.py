#!/usr/bin/env python3
"""12x32 symmetric odd-odd Hasenbusch MatPC 算子一致性探针。

用途：比较 QUDA MatQuda 的 `QUDA_CLOVER_HASENBUSCH_TWIST_DSLASH`
odd_odd symmetric MatPC 输出，和手写分解
`src + c * base + d * i_gamma5_A_src` 的关系。

主要输入：cfg、t_boundary、seed、Hasenbusch shift 链。
主要输出：`hasenbusch_operation/csv/operator_probe_<run_id>.json`。
运行前提：在 p100 上先 source
`/home/jyg/stout_smear/PyQUDA_stout_smeared_force/set_PyQUDA_env.sh`。
"""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace


OP_DIR = Path(__file__).resolve().parents[1]
SCRIPT_DIR = OP_DIR / "scripts"
CSV_DIR = OP_DIR / "csv"
CACHE_ROOT = OP_DIR / ".cache" / "operator_probe"
TESTS_DIR = OP_DIR.parent
PROJECT_DIR = TESTS_DIR.parent

sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(TESTS_DIR))

import cupy as cp

from pyquda.enum_quda import QudaDagType, QudaParity, QudaVerbosity
from pyquda.field import LatticeFermion, LatticeGauge
from pyquda_utils import core, io

import run_hmc_hasenbusch_12x32 as hmc_runner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", choices=["1000", "10001"], default="1000")
    parser.add_argument("--t-boundary", type=int, choices=[-1, 1], default=-1)
    parser.add_argument("--seed", type=int, default=10086)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--hasenbusch-mu",
        default="0,0.0005,0.0015,0.0035,0.0075,0.0155",
        help="逗号分隔的 Hasenbusch shift 链；最后一个 shift 用于 cancel monomial。",
    )
    return parser.parse_args()


def make_config(args: argparse.Namespace) -> dict:
    # 复用 HMC runner 的参数，保证 probe 和 force_check 使用同一组物理输入。
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
        hasenbusch_force_sign=-1.0,
        hasenbusch_force_coeff_scale=1.0,
        hasenbusch_force_output_scale=1.0,
        hasenbusch_force_dagger="yes",
        monomial_subset="cancel_only",
        ratio_force_source="phi",
        ratio_force_mode="pc_pair",
        cancel_force_mode="pc_pair",
        cancel_sample_scale=1.0,
        cancel_constdet=False,
        matpc_type="odd_odd",
    )
    return hmc_runner.base_config(runner_args)


def get_hasenbusch_cancel(hmc):
    # 当前 probe 只关注 cancel monomial；若布局变化，递归遍历 inner HMC。
    for monomial in getattr(hmc, "fermion_monomials", []):
        if getattr(monomial, "label", "") == "has_cancel":
            return monomial
    inner = getattr(hmc, "hmc_inner", None)
    if inner is not None:
        return get_hasenbusch_cancel(inner)
    raise RuntimeError("has_cancel monomial not found")


def half_from_odd(latt_info, odd_field: LatticeFermion) -> LatticeFermion:
    # QUDA MatPC half-vector 通过 even_ptr 传递；这里把物理 odd 数据打包回 even 半格。
    out = LatticeFermion(latt_info)
    out.data = 0
    out.even = odd_field.odd
    return out


def build_basis(action, src: LatticeFermion, dagger: QudaDagType):
    latt_info = action.latt_info
    psi_odd = action._active_pc_field(src)

    if dagger == QudaDagType.QUDA_DAG_NO:
        action._dslash_block(action.pc_tmp1, psi_odd, QudaParity.QUDA_EVEN_PARITY, dagger)
        action._clover_inv_parity(action.pc_tmp2, action.pc_tmp1, QudaParity.QUDA_EVEN_PARITY, dagger)
        action._dslash_block(action.pc_tmp3, action.pc_tmp2, QudaParity.QUDA_ODD_PARITY, dagger)
        action._clover_inv_parity(action.pc_tmp4, action.pc_tmp3, QudaParity.QUDA_ODD_PARITY, dagger)
    else:
        action._clover_inv_parity(action.pc_tmp1, psi_odd, QudaParity.QUDA_ODD_PARITY, dagger)
        action._dslash_block(action.pc_tmp2, action.pc_tmp1, QudaParity.QUDA_EVEN_PARITY, dagger)
        action._clover_inv_parity(action.pc_tmp3, action.pc_tmp2, QudaParity.QUDA_EVEN_PARITY, dagger)
        action._dslash_block(action.pc_tmp4, action.pc_tmp3, QudaParity.QUDA_ODD_PARITY, dagger)

    base = half_from_odd(latt_info, action.pc_tmp4)

    action._clover_parity(action.pc_tmp5, psi_odd, QudaParity.QUDA_ODD_PARITY, dagger)
    action._i_gamma5_parity(action.pc_tmp1, action.pc_tmp5, QudaParity.QUDA_ODD_PARITY, 1.0)
    shift = half_from_odd(latt_info, action.pc_tmp1)
    return base, shift


def complex_fit(target, base, shift):
    # 最小二乘拟合 target ~= c * base + d * shift；返回复系数和相对残差。
    b0 = base.even.data.reshape(-1)
    b1 = shift.even.data.reshape(-1)
    y = target.even.data.reshape(-1)
    mat = cp.stack([b0, b1], axis=1)
    coeff, *_ = cp.linalg.lstsq(mat, y, rcond=None)
    residual = y - mat @ coeff
    rel = cp.linalg.norm(residual) / max(float(cp.linalg.norm(y).get()), 1.0)
    return {
        "base_coeff": {"real": float(coeff[0].real.get()), "imag": float(coeff[0].imag.get())},
        "shift_coeff": {"real": float(coeff[1].real.get()), "imag": float(coeff[1].imag.get())},
        "relative_residual": float(rel.get()),
        "target_norm": float(cp.linalg.norm(y).get()),
        "base_norm": float(cp.linalg.norm(b0).get()),
        "shift_norm": float(cp.linalg.norm(b1).get()),
    }


def main() -> int:
    args = parse_args()
    config = make_config(args)
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)

    core.init(None, config["latt_size"], resource_path=str(CACHE_ROOT / args.run_id), enable_force_monitor=True)
    latt_info = core.LatticeInfo(config["latt_size"], t_boundary=config["t_boundary"], anisotropy=1.0)
    hmc = hmc_runner.make_hmc(latt_info, config, core.getLogger() if hasattr(core, "getLogger") else __import__("logging").getLogger(args.run_id))
    hmc.setFermionVerbosity(QudaVerbosity.QUDA_SILENT)
    gauge = io.readQIOGauge(str(config["gauge_path"]))
    hmc.initialize(config["seed"], gauge)
    hmc.samplePhi()

    action = get_hasenbusch_cancel(hmc)
    src = action.phi
    results = {
        "run_id": args.run_id,
        "cfg": config["cfg"],
        "t_boundary": config["t_boundary"],
        "seed": config["seed"],
        "mu": action.mu,
        "kappa": float(action.invert_param.kappa),
        "expected_quda_base_coeff": -float(action.invert_param.kappa) ** 2,
        "expected_chroma_base_coeff": -0.25,
        "expected_shift_coeff_mu": action.mu,
        "expected_shift_coeff_mu_over_2": action.mu / 2.0,
        "dagger_cases": {},
    }

    for name, dagger in (("PLUS", QudaDagType.QUDA_DAG_NO), ("MINUS", QudaDagType.QUDA_DAG_YES)):
        out_quda = LatticeFermion(latt_info)
        action._mat(out_quda, src, dagger, action.mu)
        unit = LatticeFermion(latt_info)
        unit.data = 0
        unit.even = src.even
        target = LatticeFermion(latt_info)
        target.data = out_quda.data - unit.data
        base, shift = build_basis(action, src, dagger)
        results["dagger_cases"][name] = complex_fit(target, base, shift)

    out_file = CSV_DIR / f"operator_probe_{args.run_id}.json"
    out_file.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    print(out_file)
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
