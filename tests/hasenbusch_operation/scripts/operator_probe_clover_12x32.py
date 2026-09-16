#!/usr/bin/env python3
"""标准 Clover odd-odd MatPC 的 QUDA/Chroma 结构探针。

用途：在同一 smeared gauge 上比较 QUDA CloverWilsonAction 的 MatPC 输出
与手工构造的 Chroma symmetric odd-odd 基础算子 I-0.25*A^-1*D*A^-1*D。
输出基础项拟合系数，用于判断两侧算子是否存在整体缩放或结构差异。
"""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace


SCRIPT_DIR = Path(__file__).resolve().parent
OP_DIR = SCRIPT_DIR.parent
PROJECT_DIR = OP_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_DIR))

import cupy as cp

import run_hmc_hasenbusch_12x32 as hr
from pyquda.enum_quda import QudaDagType, QudaParity
from pyquda.field import LatticeFermion
from pyquda.quda import MatQuda
from pyquda_utils import core, io


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", choices=["1000", "10001"], default="1000")
    parser.add_argument("--t-boundary", type=int, default=-1, choices=[1, -1])
    parser.add_argument("--seed", type=int, default=10086)
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
        tol=1e-10,
        nsteps_inner=1,
        nsteps_outer=1,
        placement="all_outer",
        integrator_inner="O2Nf1Ng0V",
        integrator_outer="O2Nf1Ng0V",
        hasenbusch_mu="0,0.0005",
        hasenbusch_force_sign=1.0,
        hasenbusch_force_coeff_scale=1.0,
        hasenbusch_force_output_scale=1.0,
        sym_odd_clover_scale=1.0,
        hasenbusch_force_dagger="yes",
        monomial_subset="strange_only",
        ratio_force_source="phi",
        ratio_force_mode="pc_quda_norm",
        cancel_force_mode="pc_quda_norm",
        cancel_sample_scale=1.0,
        cancel_constdet="yes",
        matpc_type="odd_odd",
        stout_dim=4,
        stout_rho=0.125,
        stout_n_smear=1,
    )
    return hr.base_config(runner_args)


def half_from_odd(latt_info, odd_field):
    """把物理 odd 半场打包回 QUDA MatPC 的 even_ptr 半向量槽位。"""
    out = LatticeFermion(latt_info)
    out.data = 0
    out.even = odd_field.odd
    return out


def get_strange(hmc):
    for monomial in hmc.fermion_monomials:
        if type(monomial).__name__ == "CloverWilsonAction":
            return monomial
    raise RuntimeError("strange CloverWilsonAction not found")


def build_chroma_base(helper, src):
    """构造 Chroma symmetric odd-odd 基础项 A^-1 D A^-1 D src。"""
    psi_odd = helper._active_pc_field(src)
    helper._dslash_block(helper.pc_tmp1, psi_odd, QudaParity.QUDA_EVEN_PARITY, QudaDagType.QUDA_DAG_NO)
    helper._clover_inv_parity(helper.pc_tmp2, helper.pc_tmp1, QudaParity.QUDA_EVEN_PARITY, QudaDagType.QUDA_DAG_NO)
    helper._dslash_block(helper.pc_tmp3, helper.pc_tmp2, QudaParity.QUDA_ODD_PARITY, QudaDagType.QUDA_DAG_NO)
    helper._clover_inv_parity(helper.pc_tmp4, helper.pc_tmp3, QudaParity.QUDA_ODD_PARITY, QudaDagType.QUDA_DAG_NO)
    return half_from_odd(helper.latt_info, helper.pc_tmp4)


def apply_quda_matpc(action, out, src):
    old_dagger = action.invert_param.dagger
    action.invert_param.dagger = QudaDagType.QUDA_DAG_NO
    out.data = 0
    MatQuda(out.even_ptr, src.even_ptr, action.invert_param)
    action.invert_param.dagger = old_dagger


def apply_a_oo_inverse(helper, out):
    """把 Chroma asymmetric MatPC 输出左乘 A_oo^-1，得到 symmetric SEOPREC 算子。"""
    out_odd = helper._active_pc_field(out)
    helper._clover_inv_parity(helper.pc_tmp5, out_odd, QudaParity.QUDA_ODD_PARITY, QudaDagType.QUDA_DAG_NO)
    result = half_from_odd(helper.latt_info, helper.pc_tmp5)
    return result


def fit_two_basis(target, src, base):
    """拟合 target ~= a0*src + a1*base，并给出归一化后的基础项系数。"""
    y = target.even.data.reshape(-1)
    x0 = src.even.data.reshape(-1)
    x1 = base.even.data.reshape(-1)
    mat = cp.stack([x0, x1], axis=1)
    coeff = cp.linalg.lstsq(mat, y, rcond=None)[0]
    residual = y - mat @ coeff
    denom = max(abs(float(coeff[0].real.get())), 1e-30)
    return {
        "identity_coeff_real": float(coeff[0].real.get()),
        "identity_coeff_imag": float(coeff[0].imag.get()),
        "base_coeff_real": float(coeff[1].real.get()),
        "base_coeff_imag": float(coeff[1].imag.get()),
        "normalized_base_coeff": float(coeff[1].real.get()) / denom,
        "relative_residual": float((cp.linalg.norm(residual) / max(float(cp.linalg.norm(y).get()), 1.0)).get()),
        "target_norm": float(cp.linalg.norm(y).get()),
        "src_norm": float(cp.linalg.norm(x0).get()),
        "base_norm": float(cp.linalg.norm(x1).get()),
    }


def main():
    args = parse_args()
    rank = hr.mpi_rank_from_env()
    config = make_config(args)
    cache = OP_DIR / ".cache" / "operator_probe_clover"
    cache.mkdir(parents=True, exist_ok=True)
    core.init(None, config["latt_size"], resource_path=str(cache / args.run_id), enable_force_monitor=False)
    latt_info = core.LatticeInfo(config["latt_size"], t_boundary=config["t_boundary"], anisotropy=1.0)

    logger = core.getLogger() if hasattr(core, "getLogger") else __import__("logging").getLogger(args.run_id)
    hmc = hr.make_hmc(latt_info, config, logger)
    gauge = io.readQIOGauge(str(config["gauge_path"]))
    hmc.initialize(args.seed, gauge)
    hmc.loadGaugeMomSmeared()

    strange = get_strange(hmc)
    strange.sampleEta()
    src = strange.eta.copy()

    # 用 Hasenbusch helper 显式构造 Chroma 基础项；mu=0 时不加入 twist。
    from pyquda.action.clover_hasenbusch import CloverWilsonHasenbuschTwistAction
    helper = CloverWilsonHasenbuschTwistAction(
        latt_info,
        config["ms_bare"],
        0.0,
        config["tol"],
        config["maxiter"],
        config["clover_csw"],
        determinant_ratio=False,
        constdet=True,
        matpc_type="odd_odd",
        label="clover_probe_helper",
    )
    helper.updateClover()
    out = LatticeFermion(latt_info)
    strange.updateClover()
    apply_quda_matpc(strange, out, src)
    out_sym = apply_a_oo_inverse(helper, out)

    base = build_chroma_base(helper, src)
    fit = fit_two_basis(out_sym, src, base)

    result = {
        "run_id": args.run_id,
        "cfg": args.cfg,
        "seed": args.seed,
        "mass": config["ms_bare"],
        "matpc_type": "odd_odd",
        "expected_chroma_base_coeff": -0.25,
        "fit": fit,
    }
    out_file = OP_DIR / "csv" / f"operator_probe_clover_{args.run_id}.json"
    if rank == 0:
        out_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(out_file)
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
