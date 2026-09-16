#!/usr/bin/env python3
"""比较 Chroma SEOPREC Clover 算子 dump 与 PyQUDA MatPC 输出。

输入：rng_results/chroma_clover_operator_{src,out}.txt。
输出：rng_results/chroma_clover_operator_compare.json。
环境前提：source set_PyQUDA_env.sh。
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import cupy as cp
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
OP_DIR = SCRIPT_DIR.parent
PROJECT_DIR = OP_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_DIR))

import run_hmc_hasenbusch_12x32 as hr
from pyquda.enum_quda import QudaDagType, QudaMatPCType, QudaParity, QudaVerbosity
from pyquda.field import LatticeFermion
from pyquda.quda import MatQuda
from pyquda.chroma_rng import ChromaRNG
from pyquda.action.clover_hasenbusch import CloverWilsonHasenbuschTwistAction
from pyquda_utils import core, io


SRC_FILE = OP_DIR / "rng_results/chroma_clover_operator_src.txt"
OUT_FILE = OP_DIR / "rng_results/chroma_clover_operator_out.txt"
RESULT_FILE = OP_DIR / "rng_results/chroma_clover_operator_compare.json"


def load_chroma_dump(path: Path):
    """读取 Chroma odd-subset dump，返回坐标到 24 个复数的字典。"""
    data = {}
    with path.open() as f:
        next(f)
        for line in f:
            values = line.split()
            coord = tuple(int(v) for v in values[:4])
            numbers = np.asarray(values[4:], dtype=np.float64).reshape(12, 2)
            data[coord] = numbers[:, 0] + 1j * numbers[:, 1]
    return data


def dump_to_field(latt_info, data):
    """把 Chroma odd-subset dump 写入 PyQUDA 物理 odd checkerboard。"""
    field = LatticeFermion(latt_info)
    field.data = 0
    coord_map = coordinate_map(latt_info)
    for (x, y, z, t), values in data.items():
        slot, index = coord_map[(x, y, z, t)]
        assert slot == 1
        field.data[slot].reshape(-1, 4, 3)[index] = cp.asarray(values.reshape(4, 3))
    return field


def field_to_array(latt_info, field):
    """把 PyQUDA 物理 odd checkerboard 转回坐标字典，便于和 dump 比较。"""
    out = {}
    host = field.data.get()
    for coord, (slot, index) in coordinate_map(latt_info).items():
        if slot != 1:
            continue
        out[coord] = host[slot].reshape(-1, 4, 3)[index].reshape(-1)
    return out


def coordinate_map(latt_info):
    """建立全局坐标 -> (checkerboard slot, half-lattice index) 的映射。"""
    coords = latt_info.coordinate()
    coords = coords.get() if hasattr(coords, "get") else coords
    lx, ly, lz, lt = latt_info.size
    lxh = lx // 2
    out = {}
    for slot in range(2):
        for t in range(lt):
            for z in range(lz):
                for y in range(ly):
                    for xh in range(lxh):
                        coord = tuple(int(v) for v in coords[:, slot, t, z, y, xh])
                        index = ((t * lz + z) * ly + y) * lxh + xh
                        out[coord] = (slot, index)
    return out


def compare_dicts(a, b):
    keys = sorted(a.keys())
    assert keys == sorted(b.keys())
    va = np.stack([a[k] for k in keys])
    vb = np.stack([b[k] for k in keys])
    diff = va - vb
    return {
        "count": len(keys),
        "max_abs_diff": float(np.max(np.abs(diff))),
        "l2_diff": float(np.linalg.norm(diff)),
        "a_norm": float(np.linalg.norm(va)),
        "b_norm": float(np.linalg.norm(vb)),
        "relative_l2": float(np.linalg.norm(diff) / max(np.linalg.norm(va), 1e-30)),
        "best_real_scale": float(np.real(np.vdot(vb, va)) / max(np.real(np.vdot(va, va)), 1e-30)),
    }


def fit_operator_relation(quda_out, chroma_out, src):
    """拟合 out_Q ≈ a*out_C + b*src，检验 hopping 基是否只差整体比例。"""
    keys = sorted(chroma_out.keys())
    y = np.stack([quda_out[k] for k in keys]).reshape(-1)
    x0 = np.stack([chroma_out[k] for k in keys]).reshape(-1)
    x1 = np.stack([src[k] for k in keys]).reshape(-1)
    mat = np.stack([x0, x1], axis=1)
    coeff = np.linalg.lstsq(mat, y, rcond=None)[0]
    residual = y - mat @ coeff
    return {
        "chroma_out_coeff": float(coeff[0].real),
        "src_coeff": float(coeff[1].real),
        "residual_l2": float(np.linalg.norm(residual)),
        "target_l2": float(np.linalg.norm(y)),
        "relative_residual": float(np.linalg.norm(residual) / max(np.linalg.norm(y), 1e-30)),
    }


def fit_linear_basis(target, basis: dict):
    """拟合 target ≈ sum_i coeff_i*basis_i，返回系数和相对残差。"""
    keys = sorted(target.keys())
    y = np.stack([target[k] for k in keys]).reshape(-1)
    mat = np.stack([np.stack([basis[name][k] for k in keys]).reshape(-1) for name in basis], axis=1)
    coeff = np.linalg.lstsq(mat, y, rcond=None)[0]
    residual = y - mat @ coeff
    return {
        "coefficients": {name: float(value.real) for name, value in zip(basis, coeff)},
        "residual_l2": float(np.linalg.norm(residual)),
        "target_l2": float(np.linalg.norm(y)),
        "relative_residual": float(np.linalg.norm(residual) / max(np.linalg.norm(y), 1e-30)),
    }


def apply_clover(helper, field, inverse: bool):
    """在物理 odd checkerboard 上应用 A_oo 或 A_oo^-1。"""
    out = LatticeFermion(field.latt_info)
    out.data = 0
    if inverse:
        helper._clover_inv_parity(out, field, QudaParity.QUDA_ODD_PARITY, QudaDagType.QUDA_DAG_NO)
    else:
        helper._clover_parity(out, field, QudaParity.QUDA_ODD_PARITY, QudaDagType.QUDA_DAG_NO)
    return out


def make_config(seed: int):
    args = SimpleNamespace(
        cfg="1000",
        gauge_path=None,
        t_boundary=-1,
        trajectory_length=1.0,
        stop=1,
        start=0,
        warm=0,
        seed=seed,
        run_id="chroma_clover_dump_compare",
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
        strange_matpc_type="odd_odd",
        strange_action_rational=True,
        stout_dim=4,
        stout_rho=0.125,
        stout_n_smear=1,
    )
    return hr.base_config(args)


def active_half_from_odd(field):
    """QUDA odd-odd MatPC 的 active half 存放在 even_ptr 槽位。"""
    out = LatticeFermion(field.latt_info)
    out.data = 0
    out.data[0] = field.data[1]
    return out


def odd_full_from_active_half(field):
    out = LatticeFermion(field.latt_info)
    out.data = 0
    out.data[1] = field.data[0]
    return out


def apply_matpc(action, src_odd, matpc_type):
    old_matpc = action.invert_param.matpc_type
    old_dagger = action.invert_param.dagger
    action.invert_param.matpc_type = matpc_type
    action.invert_param.dagger = QudaDagType.QUDA_DAG_NO
    src_half = active_half_from_odd(src_odd)
    out_half = LatticeFermion(action.latt_info)
    out_half.data = 0
    MatQuda(out_half.even_ptr, src_half.even_ptr, action.invert_param)
    action.invert_param.matpc_type = old_matpc
    action.invert_param.dagger = old_dagger
    return odd_full_from_active_half(out_half)


def get_strange(hmc):
    for m in hmc.fermion_monomials:
        if type(m).__name__ == "CloverWilsonAction":
            return m
    raise RuntimeError("strange action not found")


def main():
    chroma_src = load_chroma_dump(SRC_FILE)
    chroma_out = load_chroma_dump(OUT_FILE)
    config = make_config(10086)
    cache = OP_DIR / ".cache" / "chroma_clover_compare"
    cache.mkdir(parents=True, exist_ok=True)
    core.init(None, config["latt_size"], resource_path=str(cache), enable_force_monitor=False)
    latt_info = core.LatticeInfo(config["latt_size"], t_boundary=config["t_boundary"], anisotropy=1.0)
    logger = core.getLogger() if hasattr(core, "getLogger") else __import__("logging").getLogger("compare")
    hmc = hr.make_hmc(latt_info, config, logger)
    hmc.setFermionVerbosity(QudaVerbosity.QUDA_SILENT)
    gauge = io.readQIOGauge(str(config["gauge_path"]))
    hmc.initialize(config["seed"], gauge)
    hmc.loadGaugeMomSmeared()
    strange = get_strange(hmc)
    strange.updateClover()
    helper = CloverWilsonHasenbuschTwistAction(
        latt_info,
        config["ms_bare"],
        0.0,
        config["tol"],
        config["maxiter"],
        config["clover_csw_strange"],
        determinant_ratio=False,
        constdet=True,
        matpc_type="odd_odd",
        label="clover_dump_compare_helper",
    )
    helper.updateClover()

    # 先验证 Chroma RNG 源场和 dump 的布局/数值一致性。
    rng_src = LatticeFermion(latt_info)
    rng = ChromaRNG(tuple(config["latt_size"]), precision="double")
    rng.set_seed(config["seed"])
    rng.fill_eta(rng_src, 1)
    rng_compare = compare_dicts(field_to_array(latt_info, rng_src), chroma_src)

    src_odd = dump_to_field(latt_info, chroma_src)
    out_sym = apply_matpc(strange, src_odd, QudaMatPCType.QUDA_MATPC_ODD_ODD)
    out_asym = apply_matpc(strange, src_odd, QudaMatPCType.QUDA_MATPC_ODD_ODD_ASYMMETRIC)
    chroma_out_field = dump_to_field(latt_info, chroma_out)
    clover_basis_fields = {
        "chroma_out": chroma_out_field,
        "A_chroma_out": apply_clover(helper, chroma_out_field, inverse=False),
        "Ainv_chroma_out": apply_clover(helper, chroma_out_field, inverse=True),
        "src": src_odd,
        "A_src": apply_clover(helper, src_odd, inverse=False),
        "Ainv_src": apply_clover(helper, src_odd, inverse=True),
    }
    clover_basis_arrays = {
        name: field_to_array(latt_info, field) for name, field in clover_basis_fields.items()
    }

    result = {
        "seed": config["seed"],
        "latt_size": config["latt_size"],
        "chroma_src_norm2": sum(float(np.vdot(v, v)) for v in chroma_src.values()),
        "chroma_out_norm2": sum(float(np.vdot(v, v)) for v in chroma_out.values()),
        "rng_source_compare": rng_compare,
        "quda_symmetric_compare": compare_dicts(field_to_array(latt_info, out_sym), chroma_out),
        "quda_asymmetric_compare": compare_dicts(field_to_array(latt_info, out_asym), chroma_out),
    }
    result["quda_symmetric_operator_fit"] = fit_operator_relation(
        field_to_array(latt_info, out_sym), chroma_out, chroma_src
    )
    result["quda_asymmetric_operator_fit"] = fit_operator_relation(
        field_to_array(latt_info, out_asym), chroma_out, chroma_src
    )
    result["quda_symmetric_clover_basis_fit"] = fit_linear_basis(
        field_to_array(latt_info, out_sym), clover_basis_arrays
    )
    result["quda_asymmetric_clover_basis_fit"] = fit_linear_basis(
        field_to_array(latt_info, out_asym), clover_basis_arrays
    )
    RESULT_FILE.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(RESULT_FILE)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
