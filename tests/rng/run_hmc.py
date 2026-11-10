#!/usr/bin/env python3
"""最小 GPU A/B driver：JSON 配置和 QIO 组态输入，独立目录输出日志及组态检查点。

配置校验和 --help 只需标准库；实际运行需已安装本工作树的 PyQUDA、CuPy/MPI。
QUDA/CuPy 缓存都放在 --output-dir；检查点是组态快照，不是完整 RNG 重启文件。
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys

DEFAULT_CONFIG = Path(__file__).with_name("config_12x32.json")
REPO = Path(__file__).resolve().parents[2]


def validate_config(c):
    """在加载 GPU 库之前拒绝缺字段、非有限数及不一致的归一化。"""
    schema = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    if not isinstance(c, dict) or c.keys() != schema.keys():
        raise ValueError("config must contain exactly the documented config_12x32.json keys")
    def check(value, model, name):
        if isinstance(model, dict):
            if not isinstance(value, dict) or value.keys() != model.keys():
                raise ValueError(f"invalid keys: {name}")
            for key in model:
                check(value[key], model[key], f"{name}.{key}")
        elif isinstance(model, list):
            if not isinstance(value, list) or len(value) != len(model):
                raise ValueError(f"invalid length: {name}")
            for item, ref in zip(value, model):
                check(item, ref, name)
        elif isinstance(model, bool) or isinstance(model, str) or model is None:
            if type(value) is not type(model):
                raise ValueError(f"invalid type: {name}")
        elif type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"expected finite number: {name}")
        elif isinstance(model, int) and type(value) is not int:
            raise ValueError(f"expected integer: {name}")
    check(c, schema, "config")
    for key in ("integrator", "placement", "matpc_type", "ratio_force_mode", "cancel_force_mode",
                "cancel_constdet", "cancel_sample_scale", "strange_matpc_type", "strange_action_rational",
                "rng_precision", "stout_n_smear", "stout_dir_ignore"):
        if c[key] != schema[key]:
            raise ValueError(f"unsupported {key}: {c[key]}")
    if any(n <= 0 or n % 2 for n in c["latt_size"]) or c["t_boundary"] not in (-1, 1):
        raise ValueError("need four positive even lattice extents and t_boundary +/-1")
    if not 0 <= c["seed"] < 2**47:
        raise ValueError("seed must be in [0, 2**47)")
    for key in ("tol", "maxiter", "nsteps_inner", "nsteps_outer", "t", "chroma_tau0", "beta", "u_0", "v_0"):
        if c[key] <= 0:
            raise ValueError(f"{key} must be positive")
    if c["tol"] >= 1 or not 0 < c["u_0"] <= 1 or c["stout_rho"] < 0:
        raise ValueError("invalid tolerance, tadpole or stout coefficient")
    shifts = c["hasenbusch_mu"]
    if shifts[0] != 0 or any(b <= a for a, b in zip(shifts, shifts[1:])):
        raise ValueError("Hasenbusch shifts must start at zero and increase")
    def close(actual, expected, name):
        if not math.isclose(actual, expected, rel_tol=1e-13, abs_tol=1e-15):
            raise ValueError(f"inconsistent {name}: {actual} != {expected}")
    for mass, kappa, clover in (("ml_bare", "kappa_light", "clover_csw"),
                                ("ms_bare", "kappa_strange", "clover_csw_strange")):
        if c[mass] <= -4:
            raise ValueError("mass must exceed -4")
        close(c[kappa], 1 / (2 * (c[mass] + 4)), kappa)
        close(c[clover], c[kappa] / c["v_0"]**3, clover)
    close(c["mu_scale"], 1 / c["kappa_light"], "mu_scale")
    for mu, shift in zip(c["quda_mu"], shifts):
        close(mu, shift * c["mu_scale"], "quda_mu")
    close(c["quda_momentum_over_chroma"], math.sqrt(2), "momentum normalization")
    close(c["t"] * c["quda_momentum_over_chroma"], c["chroma_tau0"], "trajectory normalization")
    a = -4 * math.log(c["u_0"]) / 3.06839
    expected = [1, -(1 + 0.4805 * a) / (20 * c["u_0"]**2), -0.03325 * a / c["u_0"]**2]
    for actual, ref in zip(c["gauge_loop_coefficients"], expected):
        close(actual, ref, "quenched gauge loop coefficient")
    for mode in ("force", "sample", "action"):
        if any(x <= 0 for x in c["rational"]["offset_" + mode]):
            raise ValueError("rational offsets must be positive")
    return c


def load_config(path):
    return validate_config(json.loads(Path(path).read_text(encoding="utf-8")))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--gauge", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--arm", choices=("native", "chroma"), required=True)
    p.add_argument("--ntraj", type=int, default=2)
    p.add_argument("--use-chroma-accept", action="store_true")
    p.add_argument("--mom-action-only", action="store_true", help="check momentum without fermion solves or MD")
    args = p.parse_args(argv)
    if args.ntraj < 1 or (args.use_chroma_accept and args.arm != "chroma"):
        p.error("ntraj must be positive; --use-chroma-accept requires --arm chroma")
    return args


def make_hmc(li, c):
    """只调用库工厂；显式记录 Chroma 的 ratio/cancel/strange 流顺序。"""
    from pyquda.action import CloverWilsonAction, GaugeAction
    from pyquda.action.abstract import RationalParam
    from pyquda.hasenbusch import CloverHasenbuschConfig, StoutHMCConfig, build_clover_hmc
    from pyquda.hmc import STSForceGrad
    from pyquda_utils.hmc_param import symanzikOneLoopGaugeLoopParam
    gauge = GaugeAction(li, symanzikOneLoopGaugeLoopParam(c["u_0"], "quenched"), c["beta"])
    strange = CloverWilsonAction(li, RationalParam(**c["rational"]), c["ms_bare"], 1,
                                 c["tol"], c["maxiter"], c["clover_csw_strange"],
                                 use_action_rational=c["strange_action_rational"], matpc_type=c["strange_matpc_type"])
    has = CloverHasenbuschConfig(
        mass=c["ml_bare"], shifts=c["hasenbusch_mu"], clover_csw=c["clover_csw"],
        tol=c["tol"], md_tol=c["tol"], action_tol=c["tol"], maxiter=c["maxiter"],
        mu_scale=c["mu_scale"], matpc_type=c["matpc_type"], ratio_force_mode=c["ratio_force_mode"],
        cancel_force_mode=c["cancel_force_mode"], cancel_constdet=c["cancel_constdet"],
        cancel_sample_scale=c["cancel_sample_scale"])
    built = build_clover_hmc(li, gauge, [], [strange], STSForceGrad(c["nsteps_outer"]),
                            STSForceGrad(c["nsteps_inner"]), StoutHMCConfig(c["stout_n_smear"], c["stout_rho"], c["stout_dir_ignore"]),
                            use_hasenbusch=True, hasenbusch=has, placement=c["placement"])
    built.hmc.chroma_sampling_order = [*built.hasenbusch_actions.ratios, built.hasenbusch_actions.cancel, strange]
    from dataclasses import asdict
    built.hmc.effective_hasenbusch = asdict(has)
    return built.hmc


def fingerprint(field):
    data = field.getHost()
    return hashlib.sha256(data.tobytes(order="C")).hexdigest()


def rng_state(hmc, handle):
    state = hmc.random.getstate()
    probe = random.Random()
    probe.setstate(state)  # 仅复制状态预览下一 seed，不消耗原生流。
    return {"native_state_sha256": hashlib.sha256(repr(state).encode()).hexdigest(),
            "native_next_seed": probe.randrange(2**32),
            "chroma_seed": None if handle is None else int(handle.rng.get_seed())}


def momentum_check(hmc, mom, comm, volume):
    """10 实数布局还原 -Tr(P^2)/2；QUDA 每 link 减 4，总共减 16V。"""
    import numpy as np
    d = mom.getHost()
    raw = comm.allreduce(float(np.sum(d[..., :6]**2) + 0.5 * np.sum(d[..., 6:9]**2)))
    measured = float(hmc.momAction())
    expected = raw - 16 * volume
    # gauss_su3(sigma=1) 的每 link 动能均值/方差均为 4；8 sigma 仅作尺度冒烟检查。
    scale_z = (raw - 16 * volume) / math.sqrt(16 * volume)
    result = dict(host_minus_trace_p2_over_2=raw, quda_constant=-16 * volume,
                  expected=expected, measured=measured, native_scale_z=scale_z,
                  passed=math.isclose(measured, expected, rel_tol=1e-10, abs_tol=1e-7) and abs(scale_z) < 8)
    hmc.loadMom(mom)
    result["reloaded"] = float(hmc.momAction())
    result["passed"] &= math.isclose(result["reloaded"], measured, rel_tol=1e-10, abs_tol=1e-7)
    return result


def actions(hmc):
    values = dict(K=float(hmc.momAction()), gauge=float(hmc.gaugeAction()), fermion=float(hmc.fermionAction()))
    values["H"] = sum(values.values())
    if not all(math.isfinite(v) for v in values.values()):
        raise RuntimeError("non-finite Hamiltonian")
    return values


def prepare_output(out, comm):
    """只有根 rank 检查/创建目录；广播失败，避免晚到 rank 误判已生成日志。"""
    error = None
    if comm.rank == 0:
        try:
            if out == REPO or REPO in out.parents:
                raise ValueError("output-dir must be outside the source tree")
            if out.exists() and any(out.iterdir()):
                raise ValueError("output-dir must be empty; choose a fresh directory")
            out.mkdir(parents=True, exist_ok=True)
            (out / "quda-cache").mkdir()
        except (OSError, ValueError) as exc:
            error = str(exc)
    error = comm.bcast(error, root=0)
    if error is not None:
        raise ValueError(error)
    comm.Barrier()


def main(argv=None):
    args = parse_args(argv)
    c = load_config(args.config)
    gauge_path, out = args.gauge.expanduser().resolve(), args.output_dir.expanduser().resolve()
    if not gauge_path.is_file():
        raise ValueError(f"missing gauge: {gauge_path}")
    from mpi4py import MPI
    prepare_output(out, MPI.COMM_WORLD)
    cache = out / "quda-cache"
    os.environ["QUDA_RESOURCE_PATH"] = str(cache)
    os.environ["QUDA_PROFILE_OUTPUT_BASE"] = str(out / "quda-profile")
    os.environ["CUPY_CACHE_DIR"] = str(out / "cupy-cache")
    sys.dont_write_bytecode = True
    from pyquda_utils import core, io
    from pyquda.hmc import HMC, STSForceGrad
    from pyquda.field import LatticeMom
    from pyquda_comm import getMPIComm
    core.init(None, c["latt_size"], resource_path=str(cache), profile_output_base=str(out / "quda-profile"))
    comm = getMPIComm()
    li = core.LatticeInfo(c["latt_size"], t_boundary=c["t_boundary"], anisotropy=1.0)
    hmc = HMC(li, [], STSForceGrad(1)) if args.mom_action_only else make_hmc(li, c)
    gauge = io.readQIOGauge(str(gauge_path))
    hmc.initialize(c["seed"], gauge)
    handle = None
    try:
        if args.arm == "chroma":
            from chroma_rng_adapter import attach_chroma_rng
            handle = attach_chroma_rng(hmc, seed=c["seed"], precision=c["rng_precision"], use_chroma_accept=args.use_chroma_accept)
        result = dict(config=c, arm=args.arm, use_chroma_accept=args.use_chroma_accept,
                      mom_action_only=args.mom_action_only, ntraj=args.ntraj, gauge=str(gauge_path),
                      mpi_size=comm.size, effective_hasenbusch=getattr(hmc, "effective_hasenbusch", None), records=[])
        with (out / f"rank{comm.rank}.jsonl").open("x", encoding="utf-8") as log:
            def emit(record):
                log.write(json.dumps(record, allow_nan=False) + "\n")
                log.flush()
            emit(result)
            for index in range(args.ntraj):
                record = dict(traj=index + 1, seed_before=rng_state(hmc, handle))
                mom = LatticeMom(li)
                hmc.saveMom(mom)
                record["momentum_before_sha256"] = fingerprint(mom)
                if not args.mom_action_only:
                    record["eta_before_sha256"] = [fingerprint(m.eta) for m in hmc.chroma_sampling_order]
                hmc.gaussMom()
                hmc.saveMom(mom)
                record.update(momentum_sha256=fingerprint(mom), seed_after_momentum=rng_state(hmc, handle))
                if args.mom_action_only:
                    record["momentum_check"] = momentum_check(hmc, mom, comm, math.prod(c["latt_size"]))
                else:
                    hmc.samplePhi()
                    record["eta_sha256"] = [fingerprint(m.eta) for m in hmc.chroma_sampling_order]
                record["seed_after_refresh"] = rng_state(hmc, handle)
                if not args.mom_action_only:
                    record["old"] = actions(hmc)
                    hmc.integrate(c["t"], 2e-15)
                    record["new"] = actions(hmc)
                    record["delta_H"] = record["new"]["H"] - record["old"]["H"]
                    record["accept"] = bool(hmc.accept(record["delta_H"]))
                    hmc.saveGauge(gauge) if record["accept"] else hmc.loadGauge(gauge)
                    record["plaquette"] = [float(v) for v in hmc.plaquette()]
                    gauge.save(str(out / f"gauge-{index + 1:04d}.npy"))
                record["seed_after"] = rng_state(hmc, handle)
                emit(record)
                result["records"].append(comm.gather(record, root=0))
                if args.mom_action_only and not record["momentum_check"]["passed"]:
                    raise RuntimeError("momentum action/scale check failed; see rank JSONL")
            if comm.rank == 0:
                (out / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    finally:
        if handle is not None:
            handle.detach()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
