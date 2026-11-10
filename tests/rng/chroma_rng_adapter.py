"""测试用 Chroma RNG 注入器：输入 HMC 和 seed，运行时替换随机场生成方法。

显式调用 attach_chroma_rng 才启用；不修改原生 HMC 类或 solver 行为。
所有嵌套层共享一条流，须先调用顶层 gaussMom，再采样 eta 或加载内层动量。
真实场生成需要 PyQUDA/CuPy；导入本模块和假对象单元测试不需要 GPU。
detach 只恢复方法绑定，不回滚已经生成的场、诊断记录或 RNG 状态。
"""

import math
from types import MethodType


_MISSING = object()


def _iter_hmc_levels(hmc):
    """按外层到内层顺序遍历 HMC。"""
    while hmc is not None:
        yield hmc
        hmc = getattr(hmc, "hmc_inner", None)


def _active_parity_of(monomial):
    """优先使用 action 的预条件奇偶性，未声明时采用 even。"""
    if hasattr(monomial, "_active_parity"):
        return int(monomial._active_parity())
    parity = getattr(monomial, "active_parity", None)
    return 0 if parity is None else int(parity)


def _has_eta_sampler(monomial):
    """排除继承 sampleEta 却没有伪费米子场的 LogDet 等 action。"""
    return callable(getattr(monomial, "sampleEta", None)) and getattr(monomial, "eta", None) is not None


class ChromaRNGHandle:
    """保存共享随机流、eta 缓存和可撤销的实例方法替换。"""

    def __init__(self, hmc, rng, use_chroma_accept):
        self.hmc = hmc
        self.rng = rng
        self.use_chroma_accept = use_chroma_accept
        self.eta_fields = {}
        self.uniform_log = []
        self._replaced = []

    def _replace(self, obj, name, method):
        # 区分实例覆盖与类继承，避免 detach 后遗留遮蔽类方法的属性。
        self._replaced.append((obj, name, vars(obj).get(name, _MISSING)))
        setattr(obj, name, MethodType(method, obj))

    def detach(self):
        """恢复原有方法绑定；重复调用无副作用。"""
        for obj, name, original in reversed(self._replaced):
            if original is _MISSING:
                delattr(obj, name)
            else:
                setattr(obj, name, original)
        self._replaced.clear()

    def dump_stream(self, path, seed, mode):
        """离线输出参考 dump；该操作会重置并消耗共享流。"""
        self.rng.dump_reference(path, seed, mode)


def attach_chroma_rng(hmc, seed, precision="double", use_chroma_accept=False):
    """显式注入共享流；默认保留原生 accept，不提供额外布局选项。

    chroma_sampling_order（若存在）须包含全部真实 eta action 各一次；
    无 eta 的项被忽略，未声明顺序时按 HMC 层级顺序。
    每次顶层 gaussMom 先刷新动量，再刷新全部 eta；sampleEta 仅复制缓存。
    可选 accept 遇到非有限 delta_s 时抛出 ValueError，不推进随机流。
    """
    from pyquda.chroma_rng import ChromaRNG

    if use_chroma_accept:
        from pyquda_comm import getMPIComm

    levels = list(_iter_hmc_levels(hmc))
    candidates = [monomial for level in levels for monomial in getattr(level, "fermion_monomials", [])
                  if _has_eta_sampler(monomial)]
    sampling_order = getattr(hmc, "chroma_sampling_order", None)
    if sampling_order is None:
        sampling_order = candidates
    else:
        sampling_order = [monomial for monomial in sampling_order if _has_eta_sampler(monomial)]
        # 在实例方法替换前按对象身份检查，避免重复取数或遗漏后混用原生流。
        ordered_ids = [id(monomial) for monomial in sampling_order]
        if len(ordered_ids) != len(set(ordered_ids)):
            raise ValueError("chroma_sampling_order contains duplicate eta actions")
        if set(ordered_ids) != {id(monomial) for monomial in candidates}:
            raise ValueError("chroma_sampling_order must contain exactly all HMC eta actions")

    global_size = tuple(int(value) for value in hmc.latt_info.global_size)
    rng = ChromaRNG(global_size, precision=precision)
    rng.set_seed(seed)
    handle = ChromaRNGHandle(hmc, rng, use_chroma_accept)
    momentum = None

    def refresh_fields():
        nonlocal momentum
        new_momentum = rng.make_momentum(hmc.latt_info)
        new_eta_fields = {}
        for monomial in sampling_order:
            parity = _active_parity_of(monomial)
            field = monomial.eta.copy()
            rng.fill_eta(field, parity)
            if parity == 1:
                # RNG 按物理 odd 子格取数，QUDA 预条件求解源却固定从 slot0 读取。
                field.data[0] = field.data[1]
                field.data[1] = 0
            new_eta_fields[monomial] = field
        momentum = new_momentum
        handle.eta_fields.clear()
        handle.eta_fields.update(new_eta_fields)

    def sample_eta(self):
        if self not in handle.eta_fields:
            raise RuntimeError("call top-level gaussMom() before sampleEta()")
        self.eta.data = handle.eta_fields[self].data.copy()

    # 缓存初始为空也必须立即替换 sampleEta，不能依赖缓存成员判断。
    for monomial in sampling_order:
        handle._replace(monomial, "sampleEta", sample_eta)

    def load_momentum(self):
        if momentum is None:
            raise RuntimeError("call top-level gaussMom() before inner gaussMom()")
        self.loadMom(momentum)
        self.gauge_param.use_resident_mom = 1
        self.gauge_param.make_resident_mom = 1
        self.gauge_param.return_result_mom = 0

    def refresh_momentum(self):
        refresh_fields()
        load_momentum(self)

    for index, level in enumerate(levels):
        handle._replace(level, "gaussMom", refresh_momentum if index == 0 else load_momentum)

    if use_chroma_accept:

        def accept(self, delta_s):
            if not math.isfinite(delta_s):
                raise ValueError("delta_s must be finite")
            if delta_s <= 0:
                self.last_metropolis = {"acceptance_probability": 1.0, "metropolis_uniform": None}
                handle.uniform_log.append(None)
                return True
            # 各 rank 同步推进一次标量流，再以根 rank 的判定与 uniform 为准。
            uniform = rng.random_scalar()
            probability = math.exp(-delta_s)
            accepted, uniform = getMPIComm().bcast((uniform <= probability, uniform))
            self.last_metropolis = {"acceptance_probability": probability, "metropolis_uniform": uniform}
            handle.uniform_log.append(uniform)
            return accepted

        for level in levels:
            handle._replace(level, "accept", accept)

    return handle
