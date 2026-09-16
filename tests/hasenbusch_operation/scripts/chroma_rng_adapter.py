"""ChromaRNG 注入器：把 Chroma 随机数流挂到 PyQUDA HMC 上（仅测试，运行时注入）。

原则：
- 不修改任何现有 PyQUDA/QUDA 文件；所有替换在运行时以 bound method 完成，
  detach() 可完全还原。
- 单一 ChromaRNG 实例贯穿外层/内层 HMC（对应 chroma 全局单流）。
- 替换点：
  1) FermionAction.sampleEta —— 用 chroma gaussian(eta, active parity) 填场
  2) HMC.gaussMom           —— 用 chroma refreshP 动量（taproj 后取反厄米部分）
  3) HMC.accept（可选）     —— 用 chroma 标量流均匀数

用法：
    handle = attach_chroma_rng(hmc, seed=20260910)
    ... 跑 trajectory ...
    handle.detach()
"""

import math
from types import MethodType
from typing import Optional

from pyquda_comm import getMPIComm

from pyquda.chroma_rng import ChromaRNG


def _iter_hmc_levels(hmc):
    current = hmc
    while current is not None:
        yield current
        current = getattr(current, "hmc_inner", None)


def _iter_monomials(hmc):
    for level in _iter_hmc_levels(hmc):
        for m in getattr(level, "fermion_monomials", []):
            yield m


def _active_parity_of(monomial):
    # PyQUDA 的 QUDA solve 接口统一从 ``even_ptr`` 读取奇偶预条件后的源。
    # 优先使用 action 的 _active_parity()，Hasenbusch ratio/cancel 依赖该方法。
    if hasattr(monomial, "_active_parity"):
        return int(monomial._active_parity())
    active_parity = getattr(monomial, "active_parity", None)
    return 0 if active_parity is None else int(active_parity)


class ChromaRNGHandle:
    def __init__(self, hmc, rng: ChromaRNG, replaced: list, use_chroma_accept: bool):
        self.hmc = hmc
        self.rng = rng
        self._replaced = replaced
        self.use_chroma_accept = use_chroma_accept
        self.uniform_log: list = []

    def detach(self) -> None:
        for obj, name, original in self._replaced:
            setattr(obj, name, original)
        self._replaced.clear()

    def dump_stream(self, path: str, seed: int, mode: str) -> None:
        """以当前 precision 输出参考 dump（会消耗并重置流，仅离线对拍用）。"""
        self.rng.dump_reference(path, seed, mode)


def attach_chroma_rng(
    hmc,
    seed: int,
    precision: str = "double",
    use_chroma_accept: bool = False,
) -> ChromaRNGHandle:
    global_size = tuple(int(v) for v in hmc.latt_info.global_size)
    rng = ChromaRNG(global_size, precision=precision)
    rng.set_seed(seed)

    replaced: list = []
    sampling_order = getattr(hmc, "chroma_sampling_order", None)
    if sampling_order is None:
        sampling_order = list(_iter_monomials(hmc))

    # Chroma HMC 的 RNG 消费顺序是 momentum -> ratio eta -> cancel eta -> strange eta。
    # 每条 trajectory 的顶层 gaussMom() 都重新生成整套场，保证多轨迹 A 方案
    # 也能推进 Chroma RNG 状态，而不是复用第一轨迹的固定随机场。
    momentum = None
    eta_fields = {}

    def refresh_fields():
        nonlocal momentum
        momentum = rng.make_momentum(hmc.latt_info)
        eta_fields.clear()
        for monomial in sampling_order:
            if not hasattr(monomial, "sampleEta"):
                continue
            parity = _active_parity_of(monomial)
            field = monomial.eta.copy()
            rng.fill_eta(field, parity)
            if hasattr(monomial, "_active_parity"):
                active_parity = int(monomial._active_parity())
            else:
                active_parity = getattr(monomial, "active_parity", None)
            if active_parity is not None and int(active_parity) == 1:
                field.data[0] = field.data[1]
                field.data[1] = 0
            eta_fields[monomial] = field

    # 1) sampleEta 替换
    def _make_eta_method(monomial):
        def sampleEta(self):
            field = eta_fields[monomial]
            self.eta.data = field.data.copy()

        return sampleEta

    for monomial in sampling_order:
        if monomial in eta_fields:
            replaced.append((monomial, "sampleEta", monomial.sampleEta))
            monomial.sampleEta = MethodType(_make_eta_method(monomial), monomial)

    # 2) gaussMom 替换。只有顶层 HMC 负责刷新随机场；
    #    内层 HMC 仅复用同一 momentum，避免重复消耗 RNG。
    def _chroma_gauss_mom_refresh(self):
        from pyquda.field import LatticeMom

        refresh_fields()

        self.loadMom(momentum)
        self.gauge_param.use_resident_mom = 1
        self.gauge_param.make_resident_mom = 1
        self.gauge_param.return_result_mom = 0

    def _chroma_gauss_mom_reload(self):
        from pyquda.field import LatticeMom

        self.loadMom(momentum)
        self.gauge_param.use_resident_mom = 1
        self.gauge_param.make_resident_mom = 1
        self.gauge_param.return_result_mom = 0

    for level_index, level in enumerate(_iter_hmc_levels(hmc)):
        replaced.append((level, "gaussMom", level.gaussMom))
        method = _chroma_gauss_mom_refresh if level_index == 0 else _chroma_gauss_mom_reload
        level.gaussMom = MethodType(method, level)

    # 3) accept 替换（可选）
    if use_chroma_accept:

        def _chroma_accept(self, delta_s: float):
            u = rng.random_scalar()
            return getMPIComm().bcast(u <= math.exp(min(-delta_s, 0)))

        for level in _iter_hmc_levels(hmc):
            replaced.append((level, "accept", level.accept))
            level.accept = MethodType(_chroma_accept, level)

    return ChromaRNGHandle(hmc, rng, replaced, use_chroma_accept)


def _chroma_sample_eta(self, rng: ChromaRNG, parity: int):
    rng.fill_eta(self.eta, parity)
