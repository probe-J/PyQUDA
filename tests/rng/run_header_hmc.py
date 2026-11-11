#!/usr/bin/env python3
"""按 Chroma 生产头运行最小 RNG/HMC 对拍，CG/MG 共用同一构造入口。

输入：extract_chroma_header.py 生成的 JSON、原始 LIME、可选 MG 配置。
输出：仅在源树外的 --output-dir 写逐 rank JSONL、缓存及求解器审计。
需先加载 PyQUDA/MPI 环境；--help/配置检查无需 CUDA。默认原生 RNG，
--arm chroma 才显式注入 Chroma 流和接受判定；不修改输入及生产工作区。
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

from chroma_hmc_support import (
    energies, force_gate, force_gradient_probe, normalize_config,
    read_chroma_gauge, verify_mg_evidence,
)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--gauge', type=Path, help='覆盖配置中的输入路径，不改变物理参数')
    ap.add_argument('--solver', choices=('cg', 'mg'), default='cg')
    ap.add_argument('--solver-config', type=Path, help='显式 MG 参数及已加载 QUDA 库的种子证明')
    ap.add_argument('--grid', type=int, nargs=4, help='MPI 进程网格；每维必须整除格点尺寸')
    ap.add_argument('--arm', choices=('native', 'chroma'), default='native')
    ap.add_argument('--mode', choices=('gate', 'trajectory'), default='trajectory')
    ap.add_argument('--ntraj', type=int, default=2)
    args = ap.parse_args(argv)
    if args.ntraj < 1 or (args.solver == 'mg' and args.solver_config is None):
        ap.error('ntraj must be positive; MG requires --solver-config')
    return args


def solver_config(args):
    if args.solver == 'cg':
        return dict(plain_precision='double', predictor=dict(enabled=False), multigrid=dict(enabled=False))
    c = json.loads(args.solver_config.read_text(encoding='utf-8'))
    if c.get('plain_precision') != 'double' or c.get('predictor', {}).get('enabled', False):
        raise ValueError('aligned sampling/action must use double CG without predictor')
    if not c.get('multigrid', {}).get('enabled') or not c['multigrid'].get('force_gradient'):
        raise ValueError('MG mode requires MD and force-gradient MG explicitly enabled')
    return c


def build(li, c, accel):
    """质量、action 放置和采样顺序均传库接口，不在 runner 重写 Hasenbusch。"""
    from pyquda.hasenbusch import StoutHMCConfig, build_clover_hmc_from_monomials
    from pyquda.hmc import STSForceGrad
    from pyquda_utils.hmc_param import symanzikTreeGaugeLoopParam
    i = c['integrator']
    hmc, actions = build_clover_hmc_from_monomials(
        li, c['monomials'], symanzikTreeGaugeLoopParam(c['u0']), c['beta'],
        i['inner_monomials'], i['outer_monomials'], c['sampling_order'],
        STSForceGrad(i['outer']), STSForceGrad(i['inner']),
        StoutHMCConfig(**c['stout']), solver_accel=accel)
    for alias, target in c['aliases'].items():
        actions[alias] = actions[target]
    # 固定场力检查也须获得与实际 MD 相同的 stout/context 绑定。
    list(hmc._solverAccelerators())
    return hmc, actions


def fingerprint(field):
    return hashlib.sha256(field.getHost().tobytes(order='C')).hexdigest()


def main(argv=None):
    args = parse_args(argv)
    c = normalize_config(json.loads(args.config.read_text(encoding='utf-8')))
    accel = solver_config(args)
    gauge_path = (args.gauge or Path(c['gauge_path'])).expanduser().resolve(strict=True)
    out = args.output_dir.expanduser().resolve()
    from mpi4py import MPI
    from run_hmc import prepare_output
    prepare_output(out, MPI.COMM_WORLD)
    grid = args.grid or c.get('grid_size')
    if (grid is not None and (len(grid) != 4 or any(type(n) is not int or n <= 0 or length % n for n, length in zip(grid, c['latt_size']))
                             or math.prod(grid) != MPI.COMM_WORLD.size)):
        raise ValueError('grid must partition the lattice and match MPI ranks')
    os.environ['QUDA_RESOURCE_PATH'] = str(out / 'quda-cache')
    os.environ['CUPY_CACHE_DIR'] = str(out / 'cupy-cache')
    from pyquda_utils import core
    from pyquda.field import LatticeMom
    from pyquda_comm import getMPIComm
    from pyquda.solver_accel import (
        configure_solver_accel_telemetry, set_solver_accel_validation,
        get_solver_accel_telemetry, write_solver_accel_telemetry,
    )
    core.init(grid, c['latt_size'], resource_path=str(out / 'quda-cache'),
              profile_output_base=str(out / 'quda-profile'))
    comm = getMPIComm()
    configure_solver_accel_telemetry(str(out / 'solver_accel.json'), enabled=True)
    set_solver_accel_validation(True)
    li = core.LatticeInfo(c['latt_size'], t_boundary=c['t_boundary'], anisotropy=1.)
    hmc, actions = build(li, c, accel)
    gauge, input_info = read_chroma_gauge(str(gauge_path), li)
    hmc.initialize(c['seed_packed'], gauge)
    handle = None
    if args.arm == 'chroma':
        from chroma_rng_adapter import attach_chroma_rng
        handle = attach_chroma_rng(hmc, c['seed_packed'], precision='double', use_chroma_accept=True)
    def seed_state():
        if handle is not None:
            return int(handle.rng.get_seed())
        return hashlib.sha256(repr(hmc.random.getstate()).encode()).hexdigest()
    def events():
        data = get_solver_accel_telemetry()
        return data['events'] if isinstance(data, dict) else data
    expected_md_tols = {m['id'] + (':numer' if m['role'] == 'ratio' else ''): m['md_tol']
                        for m in c['monomials'] if m['role'] in ('ratio', 'cancel')}
    # QUDA 动量为 sqrt(2)*P_Chroma；两种 RNG 臂使用相同 QUDA 时间，保持 A 对照。
    tau_quda = c['integrator']['tau0'] / math.sqrt(2.)
    with (out / f'rank{comm.rank}.jsonl').open('x', encoding='utf-8') as log:
        def emit(row):
            line = json.dumps(row, ensure_ascii=False, allow_nan=False)
            log.write(line + '\n')
            log.flush()
            if comm.rank == 0:
                print(line, flush=True)
        emit(dict(type='configuration', config=c, solver=args.solver, solver_config=accel,
                  arm=args.arm, mpi_size=comm.size, gauge=str(gauge_path), input_reader=input_info,
                  tau_quda=tau_quda, momentum_normalization='P_QUDA=sqrt(2)*P_Chroma'))
        try:
            # 每个新运行都做有限差分检查；随后重置物理流，检查不消耗 trajectory RNG。
            hmc.gaussMom()
            hmc.samplePhi()
            force_gate(hmc, actions, gauge, emit)
            force_gradient_probe(hmc, gauge)
            if args.solver == 'mg':
                verify_mg_evidence(events(), expected_md_tols, temporary=True,
                                   expected_md_tols=expected_md_tols)
            emit(dict(type='gate_complete', passed=True))
            if args.mode == 'gate':
                return
            if handle is not None:
                handle.detach()
            # 释放所有层及质量比 helper 的基底/pending；沿用同一配置，正式 MD 冷启动。
            hmc.closeSolverAccel()
            hmc.configureSolverAccel(accel)
            hmc.loadGauge(gauge)
            hmc.initializeRNG(c['seed_packed'])
            if args.arm == 'chroma':
                handle = attach_chroma_rng(hmc, c['seed_packed'], precision='double', use_chroma_accept=True)
            for index in range(args.ntraj):
                hmc.beginTrajectory(index + 1)
                start, evidence_start, before = time.monotonic(), len(events()), seed_state()
                hmc.gaussMom()
                mom = LatticeMom(li)
                hmc.saveMom(mom)
                hmc.samplePhi()
                hashes = dict(momentum=fingerprint(mom), eta={m.label: fingerprint(m.eta)
                              for m in hmc.chroma_sampling_order})
                refresh = seed_state()
                old = energies(hmc)
                hmc.integrate(tau_quda, 2e-15)
                new = energies(hmc)
                if args.solver == 'mg':
                    verify_mg_evidence(events()[evidence_start:], expected_md_tols, temporary=True,
                                       expected_md_tols=expected_md_tols)
                dh = new['H'] - old['H']
                proposal = [float(x) for x in hmc.plaquette()]
                accepted = bool(hmc.accept(dh))
                if accepted:
                    hmc.saveGauge(gauge)
                else:
                    hmc.loadGauge(gauge)
                emit(dict(type='trajectory', trajectory=index + 1, old=old, new=new, delta_H=dh,
                          accept=accepted, plaquette=[float(x) for x in hmc.plaquette()],
                          plaquette_proposal=proposal, seed_before=before, seed_after_refresh=refresh,
                          seed_after=seed_state(), fields=hashes,
                          metropolis=getattr(hmc, 'last_metropolis', None),
                          elapsed_seconds=time.monotonic() - start))
        except Exception as exc:
            emit(dict(type='failure', error=repr(exc)))
            raise
        finally:
            try:
                write_solver_accel_telemetry()
            finally:
                if handle is not None:
                    handle.detach()
                hmc.closeSolverAccel()


if __name__ == '__main__':
    main()
