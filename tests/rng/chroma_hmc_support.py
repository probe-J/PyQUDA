"""对齐 Chroma 头文件的配置、QIO 输入和固定赝费米子力检查。

由 run_header_hmc.py 调用；标准库可做配置审计，实际场操作需 PyQUDA/MPI。
输入数据和检查日志都由调用方提供，本模块不初始化 GPU、不创建文件。
"""
import json
import math
from contextlib import contextmanager


def chroma_reunit_columns(gauge):
    """对应 Chroma reunit.cc 的 a[c][0/1]：按列 Gram-Schmidt，再补第三列。"""
    import numpy as np
    out = gauge.copy()
    u, v = out[..., :, 0], out[..., :, 1]
    norm = np.sqrt(abs(u[..., 0])**2 + abs(u[..., 1])**2 + abs(u[..., 2])**2)
    u *= (1. / norm)[..., None]
    dot = u[..., 0].conj()*v[..., 0] + u[..., 1].conj()*v[..., 1] + u[..., 2].conj()*v[..., 2]
    v -= dot[..., None] * u
    norm = np.sqrt(abs(v[..., 0])**2 + abs(v[..., 1])**2 + abs(v[..., 2])**2)
    v *= (1. / norm)[..., None]
    out[..., :, 2] = np.cross(u.conj(), v.conj())
    if not np.isfinite(out).all():
        raise RuntimeError('non-finite input reunitarization')
    return out


def read_chroma_gauge(filename, li):
    """读取原始 QIO 并校验 checksum；绕过 reader 的单精度按行自动投影。"""
    from pyquda_io.lime import Lime
    from pyquda_io.mpi_utils import getSublatticeSize, readMPIFile
    from pyquda_io.io_utils import checksumSciDAC
    from pyquda.field import LatticeGauge
    from pyquda_comm import getMPIComm
    from mpi4py import MPI
    import numpy as np
    lime = Lime(filename)
    file_xml = lime.loadXML('scidac-private-file-xml')
    record_xml = lime.loadXML('scidac-private-record-xml')
    checksum = lime.loadXML('scidac-checksum')
    size = [int(x) for x in file_xml.find('dims').text.split()]
    if size != list(li.global_size):
        raise ValueError('QIO dimensions differ from header')
    precision = {'D': 8, 'F': 4}[record_xml.find('precision').text]
    if int(record_xml.find('colors').text) != 3 or int(record_xml.find('datacount').text) != 4 or int(record_xml.find('typesize').text) != 18 * precision:
        raise ValueError('unsupported QIO gauge record')
    lx, ly, lz, lt = getSublatticeSize(size)
    raw = readMPIFile(filename, f'>c{2 * precision}', lime.record('ildg-binary-data').offset,
                      (lt, lz, ly, lx, 4, 3, 3), (3, 2, 1, 0))
    expected = tuple(int(checksum.find(k).text, 16) for k in ('suma', 'sumb'))
    if tuple(checksumSciDAC(size, raw.reshape(lt * lz * ly * lx, 36))) != expected:
        raise ValueError('QIO checksum mismatch')
    raw = raw.transpose(4, 0, 1, 2, 3, 5, 6).astype('<c16')
    # Gauss IO audit：读入初始化与后续 reunit=true 各执行一遍按列投影。
    projected = chroma_reunit_columns(chroma_reunit_columns(raw))
    change = getMPIComm().allreduce(float(np.max(np.abs(projected - raw))), op=MPI.MAX)
    return LatticeGauge(li, li.evenodd(projected, True)), dict(
        input_precision=precision * 8, reunit=True, reunit_passes=2,
        algorithm='Chroma column Gram-Schmidt',
        max_abs_input_change=change, bitwise_match_verified=False)


def normalize_config(config):
    """按显式 copyList 合并物理副本；拒绝未支持的参数而不是猜测。"""
    c = json.loads(json.dumps(config))
    def finite(value):
        if isinstance(value, dict):
            return all(finite(v) for v in value.values())
        if isinstance(value, list):
            return all(finite(v) for v in value)
        return not isinstance(value, float) or math.isfinite(value)
    if not finite(c):
        raise ValueError('configuration contains non-finite numbers')
    if (len(c['latt_size']) != 4 or any(type(n) is not int or n <= 0 or n % 2 for n in c['latt_size'])
            or c['t_boundary'] not in (-1, 1)):
        raise ValueError('need four positive even lattice extents and temporal boundary +/-1')
    seed = c['seed_words']
    if (len(seed) != 4 or any(type(n) is not int or not 0 <= n < 4096 for n in seed)
            or seed[3] >= 2048 or c['seed_packed'] != sum(n << (12 * i) for i, n in enumerate(seed))):
        raise ValueError('seed words and packed 47-bit state disagree')
    i = c['integrator']
    if any(i.get(k) != 'LCM_STS_FORCE_GRAD' for k in ('outer_type', 'inner_type')):
        raise ValueError('only two-level LCM_STS_FORCE_GRAD is supported')
    if (type(i['outer']) is not int or type(i['inner']) is not int
            or min(i['outer'], i['inner'], i['tau0']) <= 0):
        raise ValueError('integrator steps and trajectory length must be positive')
    if c['beta'] <= 0 or not 0 < c['u0'] <= 1:
        raise ValueError('invalid gauge coupling/tadpole coefficient')
    s = c['stout']
    if (type(s['n_smear']) is not int or s['n_smear'] < 1 or s['rho'] < 0
            or s['dir_ignore'] not in (-1, 3, 4)):
        raise ValueError('only isotropic 3D/4D stout is supported')
    roles = {'hasenbusch_ratio': 'ratio', 'hasenbusch_cancel': 'cancel'}
    groups = {}
    aliases = {}
    by_id = {r['id']: r for r in c['monomials']}
    if len(by_id) != len(c['monomials']):
        raise ValueError('duplicate monomial id')
    canonical = {mid: mid for mid in by_id}
    for pair in c.get('copy_list', []):
        source, dest = pair['copyFrom'], pair['copyTo']
        if (source not in by_id or dest not in by_id or source == dest
                or canonical[dest] != dest or canonical[source] != source):
            raise ValueError('invalid/ambiguous copyList')
        canonical[dest] = source
    for row in c['monomials']:
        row['role'] = roles.get(row['role'], row['role'])
        role = row['role']
        if role not in ('gauge', 'ratio', 'cancel', 'logdet', 'strange'):
            raise ValueError('unsupported monomial role: ' + role)
        default_csw = c['clover_csw_strange'] if role == 'strange' or (
            role == 'logdet' and row.get('n_flavor') == 1) else c['clover_csw_light']
        row['clover_csw'] = default_csw if row.get('clover_csw') is None else row['clover_csw']
        row.setdefault('tol', 1e-12)
        row.setdefault('maxiter', 10000)
        if (type(row['tol']) not in (int, float) or not 0 < row['tol'] < 1
                or type(row['maxiter']) is not int or row['maxiter'] < 1):
            raise ValueError('invalid solver tolerance/iteration cap')
        if role != 'gauge' and (type(row.get('mass')) not in (int, float) or row['mass'] <= -4):
            raise ValueError('invalid bare mass')
        if role == 'ratio' and (type(row.get('mass_precond')) not in (int, float)
                                or row['mass_precond'] <= row['mass']):
            raise ValueError('scalar mass chain must strictly increase')
        # 只有 copyList 明确相连的对象才能共享 phi；物理签名仍须逐项一致。
        key = canonical[row['id']]
        physical = {k: row.get(k) for k in (
            'role', 'mass', 'mass_precond', 'clover_csw', 'n_flavor', 'rational')}
        if key not in groups:
            groups[key] = dict(row, id=key, _physical=physical, _copies=[])
        group = groups[key]
        if group['_physical'] != physical:
            raise ValueError(f'inconsistent HMC/MD copies: {key}')
        group['_copies'].append(row)
        aliases[row['id']] = group['id']
    for group in groups.values():
        copies = group.pop('_copies')
        group.pop('_physical')
        if len({r['maxiter'] for r in copies}) != 1:
            raise ValueError(f'different copy iteration caps unsupported: {group["id"]}')
        ham = next(r for r in copies if r['id'] == group['id'])
        md = next((r for r in copies if r['id'] != group['id']), ham)
        group['action_tol'] = ham.get('action_tol', ham['tol'])
        group['md_tol'] = md.get('md_tol', md['tol'])
        if any(type(group[k]) not in (int, float) or not 0 < group[k] < 1 for k in ('action_tol', 'md_tol')):
            raise ValueError('per-pole tolerances require explicit multishift adaptation')
    c['monomials'] = list(groups.values())
    c['aliases'] = aliases
    sampling = c['sampling_order']
    momentum = [mid for mid in sampling if mid.startswith('momentum[')]
    if momentum and sampling[:4] != [f'momentum[{d}]' for d in range(4)]:
        raise ValueError('momentum refresh must precede pseudofermions in direction order')
    if momentum and len(momentum) != 4:
        raise ValueError('momentum directions must occur exactly once')
    c['sampling_order'] = [aliases[mid] for mid in sampling if not mid.startswith('momentum[')]
    if len(c['sampling_order']) != len(set(c['sampling_order'])):
        raise ValueError('sampling_order duplicates a physical pseudofermion')
    if c['stout']['dir_ignore'] == -1:
        c['stout']['dir_ignore'] = 4
    # 一个对象不能同时放进两层，否则 H/force/sample 都会重复计算。
    for key in ('inner_monomials', 'outer_monomials'):
        i[key] = [aliases[mid] for mid in i[key]]
    placed = i['inner_monomials'] + i['outer_monomials']
    if len(placed) != len(set(placed)) or set(placed) != {m['id'] for m in c['monomials']}:
        raise ValueError('each physical monomial must occur in exactly one integrator level')
    expected = {m['id'] for m in c['monomials'] if m['role'] in ('ratio', 'cancel', 'strange')}
    if set(c['sampling_order']) != expected:
        raise ValueError('sampling_order must refresh every physical pseudofermion once')
    return c


def energies(hmc):
    row = dict(kinetic=float(hmc.momAction()), gauge=float(hmc.gaugeAction()),
               fermion=float(hmc.fermionAction()))
    row['H'] = sum(row.values())
    if not all(math.isfinite(x) for x in row.values()):
        raise RuntimeError(f'non-finite Hamiltonian: {row}')
    return row


@contextmanager
def fd_tolerances(action):
    """gate 单独用双精度严格残差；退出后恢复 header 的 MD/action 残差。"""
    saved, seen = [], set()

    def visit(obj):
        if obj is None or id(obj) in seen:
            return
        seen.add(id(obj))
        for key in ('md_tol', 'action_tol'):
            if hasattr(obj, key):
                saved.append((obj, key, getattr(obj, key)))
                setattr(obj, key, 1e-12)
        param = getattr(obj, 'invert_param', None)
        if param is not None and id(param) not in seen:
            seen.add(id(param))
            saved.append((param, 'tol', param.tol))
            param.tol = 1e-12
        for key in ('numer', 'denom', '_sym_force_helper'):
            visit(getattr(obj, key, None))

    try:
        visit(action)
        yield
    finally:
        for obj, key, value in reversed(saved):
            setattr(obj, key, value)


def momentum_dot(a, b):
    """MILC 10-real 存储：非对角权重 2、三个虚对角权重 1；忽略 padding。"""
    import numpy as np
    from pyquda_comm import getMPIComm
    x, y = a.getHost(), b.getHost()
    local = 2 * np.sum(x[..., :6] * y[..., :6]) + np.sum(x[..., 6:9] * y[..., 6:9])
    return float(getMPIComm().allreduce(float(local)))


def force_gate(hmc, actions, gauge, emit):
    """固定 phi 的 thin-link 群流 FD；包含规范力，不允许近零斜率放行。"""
    from pyquda.field import LatticeMom
    li = hmc.latt_info
    original = LatticeMom(li)
    hmc.saveMom(original)
    direction = original.copy()
    direction.data *= .1
    zero = LatticeMom(li)
    zero.data = 0
    passed = True
    unique = {id(a): a for a in actions.values()}
    try:
        for action in unique.values():
            phi = getattr(action, 'phi', None)
            phi_saved = None if phi is None else phi.data.copy()
            smeared = hasattr(action, 'force_v2')
            rows = []
            try:
                with fd_tolerances(action):
                    hmc.loadGauge(gauge)
                    hmc.loadMom(zero)
                    if smeared:
                        hmc.loadGaugeMomSmeared()
                        action.force_v2(1., hmc.force_v2)
                        hmc.loadGaugeMom()
                    else:
                        action.force(1.)
                    force = LatticeMom(li)
                    hmc.saveMom(force)
                    slope = -momentum_dot(force, direction)
                    # 两点动能公式只作诊断，避免大范围 K 的相减污染小 ratio 力。
                    p, m = force.copy(), force.copy()
                    p.data += direction.data
                    m.data -= direction.data
                    hmc.loadMom(p)
                    kp = float(hmc.momAction())
                    hmc.loadMom(m)
                    km = float(hmc.momAction())
                    kinetic_slope = -(kp - km) / 2
                    metric_ok = (math.isfinite(slope) and math.isfinite(kinetic_slope)
                                 and abs(slope - kinetic_slope) <= 1e-7 + 1e-8 * abs(slope))
                    # 小质量间隔 ratio 的 S 约 3e5、斜率仅约 0.025；
                    # 加入较大步长寻找截断/舍入误差之间的收敛区间，不放宽门槛。
                    for eps in (1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5):
                        values = []
                        for displacement in (eps, -eps):
                            if phi is not None:
                                phi.data = phi_saved.copy()
                            hmc.loadGauge(gauge)
                            hmc.loadMom(direction)
                            # eps 是 QUDA 内部群流时间；FD 不再额外除 sqrt(2)。
                            hmc.updateGauge(displacement)
                            if smeared:
                                hmc.loadGaugeMomSmeared()
                            values.append(float(action.action()))
                            if smeared:
                                hmc.loadGaugeMom()
                        if not all(math.isfinite(v) for v in values + [slope]):
                            raise RuntimeError(f'non-finite FD values: {action.label}')
                        fd = (values[0] - values[1]) / (2 * eps)
                        scale = max(abs(fd), abs(slope))
                        error = abs(fd - slope)
                        limit = 1e-8 + 1e-5 * scale
                        # 中心差分低于浮点消减分辨率时标为失败，不以增大 atol 掩盖。
                        roundoff = 8 * max(math.ulp(v) for v in values) / (2 * eps)
                        ok = metric_ok and scale >= 1e-7 and error <= limit and roundoff <= limit
                        rows.append(dict(epsilon=eps, plus=values[0], minus=values[1],
                                         difference=fd, force_slope=slope, absolute_error=error,
                                         relative_error=error / max(scale, 1e-300),
                                         roundoff_bound=roundoff, passed=ok))
                ok = any(a['passed'] and b['passed'] and
                         abs(a['difference'] - b['difference']) <=
                         2e-8 + 2e-5 * max(abs(a['difference']), abs(b['difference']))
                         for a, b in zip(rows, rows[1:]))
                action.fd_gate_passed = ok
                emit(dict(type='force_gate', label=action.label, passed=ok,
                          kinetic_slope=kinetic_slope, metric_passed=metric_ok, rows=rows))
                passed &= ok
            finally:
                if phi is not None:
                    phi.data = phi_saved
    finally:
        hmc.loadGauge(gauge)
        hmc.loadMom(original)
    if not passed or not unique:
        raise RuntimeError('force finite-difference gate failed; trajectories prohibited')


def verify_mg_evidence(data, labels, *, temporary, expected_md_tols=None):
    """顺序验收每次 MG 调用；普通 MD 可显式重试，FG 不得失败或重建。"""
    labels = tuple(labels)
    expected_md_tols = (dict.fromkeys(labels, 1e-12) if expected_md_tols is None
                        else expected_md_tols)
    for label in labels:
        tol = expected_md_tols.get(label)
        if type(tol) not in (int, float) or not math.isfinite(tol) or not 0 < tol < 1:
            raise ValueError(f'invalid expected MD tolerance: {label} {tol}')
    events = data['events'] if isinstance(data, dict) else data
    kind = lambda e: e.get('event', e.get('type'))
    for e in events:
        if (e.get('scope') == 'temporary'
                and kind(e) in ('setup', 'refresh', 'rebuild', 'solve_attempt_failure')):
            raise RuntimeError(f'FG unexpectedly changed MG basis: {e}')
        if (e.get('scope') == 'temporary' and kind(e) == 'update'
                and (e.get('refresh_iterations') != 0 or e.get('basis_frozen') is not True)):
            raise RuntimeError(f'FG update did not freeze MG basis: {e}')
        if e.get('fallback'):
            raise RuntimeError(f'unexpected solver fallback: {e}')
        if kind(e) == 'solve':
            if e.get('scope') in ('sample', 'action') and e.get('multigrid_used'):
                raise RuntimeError(f'sample/action must remain CG: {e}')
            if (e.get('action') in labels and e.get('scope') in ('md', 'temporary')
                    and e.get('multigrid_used') is not True):
                raise RuntimeError(f'missing real MG solve: {e}')
    for label in labels:
        for scope in (('md', 'temporary') if temporary else ('md',)):
            phase, recovery, retried, completed = 'idle', None, False, 0

            def invalid(event):
                raise RuntimeError(f'missing real MG/original-residual evidence: {label} {scope}: {event}')

            # 同步串行调用无需 solve_id；保留原事件次序，不能用另一调用的成功掩盖失败。
            for e in events:
                if e.get('action') != label or e.get('scope') != scope:
                    continue
                event = kind(e)
                if event == 'solve_attempt_failure':
                    if (scope == 'temporary' or type(e.get('retry')) is not int
                            or e['retry'] != 0 or retried or recovery is not None):
                        invalid(e)
                    phase, recovery = 'idle', 'rebuild'
                elif event in ('rebuild', 'setup'):
                    if phase != 'idle':
                        invalid(e)
                    if recovery is not None:
                        if event != recovery:
                            invalid(e)
                        recovery = 'setup' if event == 'rebuild' else 'ready'
                elif event == 'solve':
                    if recovery not in (None, 'ready'):
                        invalid(e)
                    if e.get('direction') == 'Mdag' and phase == 'idle':
                        phase = 'Mdag'
                        if recovery == 'ready':
                            recovery, retried = None, True
                    elif e.get('direction') == 'M' and phase == 'Mdag':
                        phase = 'M'
                    else:
                        invalid(e)
                elif event == 'normal_residual':
                    # 零 RHS 尚无显式事件，孤立残差不能代替真实两方向求解证据。
                    if phase != 'M' or recovery is not None:
                        invalid(e)
                    target, residual = e.get('target'), e.get('true_residual')
                    if (e.get('equation') != 'original_normal'
                            or type(e.get('converged')) is not bool
                            or type(target) not in (int, float) or not math.isfinite(target)
                            or not 0 < target <= expected_md_tols[label]):
                        invalid(e)
                    good = (type(residual) in (int, float) and math.isfinite(residual)
                            and 0 <= residual <= target and e['converged'] is True)
                    if good:
                        completed += 1
                        phase, retried = 'idle', False
                    else:
                        if scope == 'temporary':
                            invalid(e)
                        # 普通 MD 的坏残差须先由失败事件封口，再重建并成功；不可直接续算。
                        phase = 'failed_residual'
            if not completed or phase != 'idle' or recovery is not None:
                invalid('incomplete solve/retry sequence')


def force_gradient_probe(hmc, gauge):
    """仅 gate：普通力预热后检查两层 FG，最终恢复规范场、原动量。"""
    from pyquda.field import LatticeMom
    saved = LatticeMom(hmc.latt_info)
    hmc.saveMom(saved)
    try:
        hmc.loadGauge(gauge)
        hmc.updateMom(0.)
        hmc.hmc_inner.updateMom(0.)
        for level in (hmc, hmc.hmc_inner):
            level.forceGradient(1e-6, 1e-4)
    finally:
        hmc.loadGauge(gauge)
        hmc.loadMom(saved)
