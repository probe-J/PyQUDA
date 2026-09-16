from abc import ABC, abstractmethod
from math import exp
from random import Random
from typing import List, Optional, Union

import numpy

from pyquda_comm import getLogger, getMPIComm, getArrayBackend
from pyquda_comm.array import arrayRandomGetState, arrayRandomSetState, arrayRandomSeed
from .field import LatticeInfo, LatticeGauge, LatticeMom, LatticeReal
from .quda import (
    QudaGaugeObservableParam,
    gaugeObservablesQuda,
    gaussMomQuda,
    loadGaugeQuda,
    saveGaugeQuda,
    momResidentQuda,
    momActionQuda,
    updateGaugeFieldQuda,
)
from .enum_quda import QudaBoolean, QudaTboundary, QudaVerbosity
from .dirac import WilsonDirac, StaggeredDirac
from .action.abstract import Action, FermionAction, StaggeredFermionAction

nullptr = numpy.empty((0, 0), "<c16")


class Integrator(ABC):
    def __init__(self, n_steps: int) -> None:
        self.n_steps = n_steps

    @abstractmethod
    def integrate(self, updateGauge, updateMom, t: float):
        raise NotImplementedError("The integrator is not implemented yet")


class INT_3G1F(Integrator):
    alpha_ = 0.1
    beta_ = 0.1

    def integrate(self, updateGauge, updateMom, t: float):
        assert self.n_steps % 2 == 0
        dt = t / self.n_steps
        for _ in range(0, self.n_steps, 2):
            updateGauge(dt * (1 / 6 - self.alpha_ / 3))
            updateMom(dt / 3, fermion=False)
            updateGauge(dt * (1 / 3 + self.alpha_ / 3 - self.beta_))
            updateMom(dt, gauge=False)
            updateGauge(dt * (self.alpha_ / 3 + self.beta_))
            updateMom(dt / 3, fermion=False)
            updateGauge(dt * (1 / 3 - self.alpha_ * 2 / 3))
            updateMom(dt / 3, fermion=False)
            updateGauge(dt * (1 / 3 + self.alpha_ * 2 / 3))
            updateMom(dt / 3, fermion=False)
            updateGauge(dt * (1 / 3 - self.alpha_ * 2 / 3))
            updateMom(dt / 3, fermion=False)
            updateGauge(dt * (self.alpha_ / 3 + self.beta_))
            updateMom(dt, gauge=False)
            updateGauge(dt * (1 / 3 + self.alpha_ / 3 - self.beta_))
            updateMom(dt / 3, fermion=False)
            updateGauge(dt * (1 / 6 - self.alpha_ / 3))


class O2Nf1Ng0V(Integrator):
    R"""https://doi.org/10.1016/S0010-4655(02)00754-3
    BAB: Eq.(23), \xi=0"""

    def integrate(self, updateGauge, updateMom, t: float):
        dt = t / self.n_steps
        updateMom(dt / 2)
        for _ in range(self.n_steps - 1):
            updateGauge(dt)
            updateMom(dt)
        updateGauge(dt)
        updateMom(dt / 2)


class O2Nf1Ng0P(Integrator):
    R"""https://doi.org/10.1016/S0010-4655(02)00754-3
    ABA: Eq.(24), \xi=0"""

    def integrate(self, updateGauge, updateMom, t: float):
        dt = t / self.n_steps
        updateGauge(dt / 2)
        for _ in range(self.n_steps - 1):
            updateMom(dt)
            updateGauge(dt)
        updateMom(dt)
        updateGauge(dt / 2)


class O2Nf2Ng0V(Integrator):
    R"""https://doi.org/10.1016/S0010-4655(02)00754-3
    BABAB: Eq.(25), Eq.(31)"""

    lambda_ = 0.1931833275037836

    def integrate(self, updateGauge, updateMom, t: float):
        dt = t / self.n_steps
        updateMom(self.lambda_ * dt)
        for _ in range(self.n_steps - 1):
            updateGauge(dt / 2)
            updateMom((1 - 2 * self.lambda_) * dt)
            updateGauge(dt / 2)
            updateMom(2 * self.lambda_ * dt)
        updateGauge(dt / 2)
        updateMom((1 - 2 * self.lambda_) * dt)
        updateGauge(dt / 2)
        updateMom(self.lambda_ * dt)


class O2Nf2Ng0P(Integrator):
    R"""https://doi.org/10.1016/S0010-4655(02)00754-3
    ABABA: Eq.(32), Eq.(31)"""

    lambda_ = 0.1931833275037836

    def integrate(self, updateGauge, updateMom, t: float):
        dt = t / self.n_steps
        updateGauge(self.lambda_ * dt)
        for _ in range(self.n_steps - 1):
            updateMom(dt / 2)
            updateGauge((1 - 2 * self.lambda_) * dt)
            updateMom(dt / 2)
            updateGauge(2 * self.lambda_ * dt)
        updateMom(dt / 2)
        updateGauge((1 - 2 * self.lambda_) * dt)
        updateMom(dt / 2)
        updateGauge(self.lambda_ * dt)


class STSLeapfrog(Integrator):
    R"""Chroma LCM_STS_LEAPFROG 的递归 STS leapfrog 积分器。

    当前层负责 updateMom，子层通过 updateGauge 演化。该类用于
    与 Chroma 的无 force-gradient 积分器做逐步隔离对拍。
    """

    def integrate(self, updateGauge, updateMom, t: float):
        dt = t / self.n_steps
        half_dt = dt / 2.0
        updateMom(half_dt)
        for _ in range(self.n_steps - 1):
            updateGauge(dt)
            updateMom(dt)
        updateGauge(dt)
        updateMom(half_dt)


class STSForceGrad(Integrator):
    R"""Chroma ``LCM_STS_FORCE_GRAD`` 的递归 STS force-gradient 积分器。

    使用 arXiv:1111.5059 的 shifted-force 实现，不显式构造 Hessian。当前层负责
    ``updateMom``，子层负责半步演化；临时 gauge/momentum 的保存与恢复由 HMC 提供。
    """

    lambda_ = 1.0 / 6.0
    xi_ = 1.0 / 72.0

    def integrate(self, updateGauge, updateMom, t: float, forceGradient=None):
        if forceGradient is None:
            raise ValueError("STSForceGrad requires an HMC forceGradient callback")
        dt = t / self.n_steps
        lambda_dt = self.lambda_ * dt
        half_dt = dt / 2.0
        middle_dt = (1.0 - 2.0 * self.lambda_) * dt
        double_lambda_dt = 2.0 * lambda_dt
        shifted_force_dt = 2.0 * dt**3 * self.xi_ / middle_dt

        updateMom(lambda_dt)
        for _ in range(self.n_steps - 1):
            updateGauge(half_dt)
            forceGradient(shifted_force_dt, middle_dt)
            updateGauge(half_dt)
            updateMom(double_lambda_dt)
        updateGauge(half_dt)
        forceGradient(shifted_force_dt, middle_dt)
        updateGauge(half_dt)
        updateMom(lambda_dt)


class O4Nf5Ng0V(Integrator):
    R"""https://doi.org/10.1016/S0010-4655(02)00754-3
    BABABABABAB: Eq.(63), Eq.(71)"""

    rho_ = 0.2539785108410595
    theta_ = -0.03230286765269967
    vartheta_ = 0.08398315262876693
    lambda_ = 0.6822365335719091

    def integrate(self, updateGauge, updateMom, t: float):
        dt = t / self.n_steps
        updateMom(self.vartheta_ * dt)
        for _ in range(self.n_steps - 1):
            updateGauge(self.rho_ * dt)
            updateMom(self.lambda_ * dt)
            updateGauge(self.theta_ * dt)
            updateMom((1 - 2 * (self.lambda_ + self.vartheta_)) * dt / 2)
            updateGauge((1 - 2 * (self.theta_ + self.rho_)) * dt)
            updateMom((1 - 2 * (self.lambda_ + self.vartheta_)) * dt / 2)
            updateGauge(self.theta_ * dt)
            updateMom(self.lambda_ * dt)
            updateGauge(self.rho_ * dt)
            updateMom(2 * self.vartheta_ * dt)
        updateGauge(self.rho_ * dt)
        updateMom(self.lambda_ * dt)
        updateGauge(self.theta_ * dt)
        updateMom((1 - 2 * (self.lambda_ + self.vartheta_)) * dt / 2)
        updateGauge((1 - 2 * (self.theta_ + self.rho_)) * dt)
        updateMom((1 - 2 * (self.lambda_ + self.vartheta_)) * dt / 2)
        updateGauge(self.theta_ * dt)
        updateMom(self.lambda_ * dt)
        updateGauge(self.rho_ * dt)
        updateMom(self.vartheta_ * dt)


class O4Nf5Ng0P(Integrator):
    R"""https://doi.org/10.1016/S0010-4655(02)00754-3
    ABABABABABA: Eq.(72), Eq.(80)"""

    rho_ = 0.2750081212332419
    theta_ = -0.1347950099106792
    vartheta_ = -0.08442961950707149
    lambda_ = 0.3549000571574260

    def integrate(self, updateGauge, updateMom, t: float):
        dt = t / self.n_steps
        updateGauge(self.rho_ * dt)
        for _ in range(self.n_steps - 1):
            updateMom(self.vartheta_ * dt)
            updateGauge(self.theta_ * dt)
            updateMom(self.lambda_ * dt)
            updateGauge((1 - 2 * (self.theta_ + self.rho_)) * dt / 2)
            updateMom((1 - 2 * (self.lambda_ + self.vartheta_)) * dt)
            updateGauge((1 - 2 * (self.theta_ + self.rho_)) * dt / 2)
            updateMom(self.lambda_ * dt)
            updateGauge(self.theta_ * dt)
            updateMom(self.vartheta_ * dt)
            updateGauge(2 * self.rho_ * dt)
        updateMom(self.vartheta_ * dt)
        updateGauge(self.theta_ * dt)
        updateMom(self.lambda_ * dt)
        updateGauge((1 - 2 * (self.theta_ + self.rho_)) * dt / 2)
        updateMom((1 - 2 * (self.lambda_ + self.vartheta_)) * dt)
        updateGauge((1 - 2 * (self.theta_ + self.rho_)) * dt / 2)
        updateMom(self.lambda_ * dt)
        updateGauge(self.theta_ * dt)
        updateMom(self.vartheta_ * dt)
        updateGauge(self.rho_ * dt)


class HMC:
    def __init__(
        self,
        latt_info: LatticeInfo,
        monomials: List[Union[Action, FermionAction, StaggeredFermionAction]],
        integrator: Integrator,
        hmc_inner: Optional["HMC"] = None,
        *,
        stout_n_smear: int = 1,
        stout_rho: float = 0.125,
        stout_dir_ignore: int = 4,
        stout_smear_anisotropy: float = 1.0,
    ) -> None:
        self.latt_info = latt_info
        self.gauge_monomials: List[Action] = [
            monomial for monomial in monomials if not isinstance(monomial, FermionAction)
        ]
        self.fermion_monomials: List[FermionAction] = [
            monomial for monomial in monomials if isinstance(monomial, FermionAction)
        ]
        self.integrator = integrator
        self.hmc_inner = hmc_inner
        # HMC stout reverse 当前一次只回传一层；禁止把多层 forward 误配给单层 reverse。
        if stout_n_smear != 1:
            raise NotImplementedError("HMC stout force reverse currently supports exactly one smear level")
        if stout_dir_ignore not in (3, 4):
            raise ValueError("HMC stout_dir_ignore must be 3 (S3) or 4 (S4)")
        if not numpy.isfinite(stout_rho) or not numpy.isfinite(stout_smear_anisotropy):
            raise ValueError("stout_rho and stout_smear_anisotropy must be finite")
        if stout_smear_anisotropy != 1.0:
            raise NotImplementedError("HMC stout reverse currently supports isotropic rho only")
        self.stout_n_smear = int(stout_n_smear)
        self.stout_rho = float(stout_rho)
        self.stout_dir_ignore = int(stout_dir_ignore)
        self.stout_smear_anisotropy = float(stout_smear_anisotropy)
        self.stout_config = (
            self.stout_n_smear,
            self.stout_rho,
            self.stout_dir_ignore,
            self.stout_smear_anisotropy,
        )
        if hmc_inner is not None and hmc_inner.stout_config != self.stout_config:
            raise ValueError("outer and inner HMC must use the same stout forward/reverse configuration")
        if len(self.fermion_monomials) == 0:
            self.is_staggered = False
        else:
            self.is_staggered = isinstance(self.fermion_monomials[0], StaggeredFermionAction)
            for monomial in self.fermion_monomials[1:]:
                if self.is_staggered != isinstance(monomial, StaggeredFermionAction):
                    getLogger().critical(
                        "FermionAction and StaggeredFermionAction cannot be used at the same time", ValueError
                    )
        if not self.is_staggered:
            self._dirac = WilsonDirac(latt_info, 0, 0, 0)
        else:
            self._dirac = StaggeredDirac(latt_info, 0, 0, 0)
        self.gauge_param = self._dirac.gauge_param
        self.obs_param = QudaGaugeObservableParam()
        self.obs_param.remove_staggered_phase = QudaBoolean(self.is_staggered)
        self.fuseFermionAction()
        self.gauge = None
        self.smeared = None
        self.mom = None
        self.force = None
        self.force_v2 = None

    def fuseFermionAction(self):
        if self.is_staggered:
            from .action.hisq import HISQAction, MultiHISQAction

            hisq_monomials = [monomial for monomial in self.fermion_monomials if isinstance(monomial, HISQAction)]
            if hisq_monomials != []:
                hisq_monomials = [MultiHISQAction(self.latt_info, hisq_monomials)]
            self.fermion_monomials = hisq_monomials + [
                monomial for monomial in self.fermion_monomials if not isinstance(monomial, HISQAction)
            ]

    def initializeRNG(self, seed: int):
        self.random = Random(seed * self.latt_info.volume + self.latt_info.mpi_rank)
        self.random.setstate(
            (
                self.random.VERSION,
                tuple(self.random.randrange(2**32) if i < 624 else self.random.randrange(625) for i in range(625)),
                None,
            )
        )
        current = self
        while current.hmc_inner is not None:
            current.hmc_inner.random = current.random
            current = current.hmc_inner

    def initializeFields(self):
        self.gauge = LatticeGauge(self.latt_info)
        self.smeared = LatticeGauge(self.latt_info)
        self.mom = LatticeMom(self.latt_info)
        self.force = LatticeMom(self.latt_info)
        self.force_v2 = LatticeGauge(self.latt_info)
        if self.hmc_inner is not None:
            self.hmc_inner.initializeFields()

    def initialize(self, seed: int, gauge: LatticeGauge, mom: Optional[LatticeMom] = None):
        self.initializeRNG(seed)
        self.initializeFields()
        self.loadGauge(gauge)
        if mom is None:
            self.loadMom(LatticeMom(gauge.latt_info))
        else:
            self.loadMom(mom)

    def setFermionVerbosity(self, verbosity: QudaVerbosity):
        for fermion_monomial in self.fermion_monomials:
            fermion_monomial.setVerbosity(verbosity)
        if self.hmc_inner is not None:
            self.hmc_inner.setFermionVerbosity(verbosity)

    def _stoutSmear(self):
        # import os
        # import cupy as cp

        # rho = 0.125
        # dim = 4

        # if not hasattr(self, "_stout_smear"):
        #     with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "stout_smear.cu")) as f:
        #         self._stout_smear = cp.RawModule(
        #             code=f.read(), options=("--std=c++11",), name_expressions=(f"stout_smear<double, {dim}>",)
        #         ).get_function(f"stout_smear<double, {dim}>")

        # Lx, Ly, Lz, Lt = self.smeared.latt_info.size
        # gauge = self.smeared.data
        # smeared = self.smeared.data.copy()
        # self._stout_smear((Lx * Ly * Lz // 2, dim, 2), (Lt, 1, 1), (smeared, gauge, rho, Lx, Ly, Lz, Lt))
        # self.smeared.data = smeared

        self.smeared.stoutSmear(
            self.stout_n_smear,
            self.stout_rho,
            self.stout_dir_ignore,
        )

    def _stoutSmearReverse(self):
        # import os
        # import cupy as cp

        # rho = 0.125
        # dim = 4

        # if not hasattr(self, "_compute_lambda_kernel"):
        #     with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "stout_smear.cu")) as f:
        #         self._compute_lambda_kernel = cp.RawModule(
        #             code=f.read(), options=("--std=c++11",), name_expressions=(f"compute_lambda_kernel<double, {dim}>",)
        #         ).get_function(f"compute_lambda_kernel<double, {dim}>")
        # if not hasattr(self, "_stout_smear_reverse"):
        #     with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "stout_smear.cu")) as f:
        #         self._stout_smear_reverse = cp.RawModule(
        #             code=f.read(), options=("--std=c++11",), name_expressions=(f"stout_smear_reverse<double, {dim}>",)
        #         ).get_function(f"stout_smear_reverse<double, {dim}>")

        # Lx, Ly, Lz, Lt = self.smeared.latt_info.size
        # gauge = self.smeared.data
        # force = self.force_v2.data.copy()
        # force_ori = self.force_v2.data.copy()
        # lambda_ = self.force_v2.data.copy()
        # self._compute_lambda_kernel(
        #     (Lx * Ly * Lz // 2, dim, 2), (Lt, 1, 1), (force, lambda_, gauge, rho, Lx, Ly, Lz, Lt)
        # )
        # self._stout_smear_reverse((Lx * Ly * Lz // 2, dim, 2), (Lt, 1, 1), (force, lambda_, gauge, rho, Lx, Ly, Lz, Lt))
        # self.force_v2.data = force

        from pyquda import quda

        smear_param = quda.QudaGaugeSmearParam()
        smear_param.smear_type = 1
        smear_param.n_steps = self.stout_n_smear
        smear_param.rho = self.stout_rho
        smear_param.dir_ignore = self.stout_dir_ignore
        smear_param.smear_anisotropy = self.stout_smear_anisotropy
        self.gauge_param.use_resident_force = 0
        self.gauge_param.return_result_force = 1
        self.gauge_param.overwrite_force = 0
        quda.computeStoutForceQuda(self.force_v2.data_ptrs, self.gauge_param, smear_param)

    def loadGaugeMomSmeared(self):
        if self.gauge is not None and self.smeared is not None and self.force is not None and self.mom is not None:
            saveGaugeQuda(self.gauge.data_ptrs, self.gauge_param)
            self.gauge_param.make_resident_mom = 0
            self.gauge_param.return_result_mom = 1
            momResidentQuda(self.mom.data_ptrs, self.gauge_param)
            self.gauge_param.make_resident_mom = 1
            self.gauge_param.return_result_mom = 0

            self.smeared.data = self.gauge.data
            if self.gauge_param.t_boundary == QudaTboundary.QUDA_ANTI_PERIODIC_T:
                self.smeared.setAntiPeriodicT()
            if self.is_staggered:
                self.smeared.staggeredPhase(True)
            self._stoutSmear()
            if self.is_staggered:
                self.smeared.staggeredPhase(False)
            if self.gauge_param.t_boundary == QudaTboundary.QUDA_ANTI_PERIODIC_T:
                self.smeared.setAntiPeriodicT()
            self.gauge_param.use_resident_gauge = 0
            loadGaugeQuda(self.smeared.data_ptrs, self.gauge_param)
            self.gauge_param.use_resident_gauge = 1
            self.force.data = 0
            self.force_v2.data = 0

    def loadGaugeMom(self):
        if self.gauge is not None and self.smeared is not None and self.force is not None and self.mom is not None:
            self.gauge_param.use_resident_gauge = 0
            loadGaugeQuda(self.gauge.data_ptrs, self.gauge_param)
            self.gauge_param.use_resident_gauge = 1

            self.smeared.data = self.gauge.data
            if self.gauge_param.t_boundary == QudaTboundary.QUDA_ANTI_PERIODIC_T:
                self.smeared.setAntiPeriodicT()
            if self.is_staggered:
                self.smeared.staggeredPhase(True)
            self._stoutSmearReverse()

            import cupy as cp

            uf = cp.einsum("...ab,...bc->...ac", self.gauge.data, self.force_v2.data)
            uf_ah = (uf - uf.conj().swapaxes(axis1=-2, axis2=-1)) / 2
            uf_tlah = uf_ah - (uf_ah.trace(axis1=-2, axis2=-1) / 3)[..., None, None] * cp.eye(3)
            self.force.data[..., 0] = uf_tlah[..., 0, 1].real
            self.force.data[..., 1] = uf_tlah[..., 0, 1].imag
            self.force.data[..., 2] = uf_tlah[..., 0, 2].real
            self.force.data[..., 3] = uf_tlah[..., 0, 2].imag
            self.force.data[..., 4] = uf_tlah[..., 1, 2].real
            self.force.data[..., 5] = uf_tlah[..., 1, 2].imag
            self.force.data[..., 6] = uf_tlah[..., 0, 0].imag
            self.force.data[..., 7] = uf_tlah[..., 1, 1].imag
            self.force.data[..., 8] = uf_tlah[..., 2, 2].imag

            self.mom += self.force

            # stout force 回传后，host momentum 已经累加 thin-link force；
            # 这里显式按 host->resident 方向刷新 QUDA momentum，和 loadMom() 语义一致。
            self.gauge_param.use_resident_mom = 0
            self.gauge_param.make_resident_mom = 1
            self.gauge_param.return_result_mom = 0
            momResidentQuda(self.mom.data_ptrs, self.gauge_param)
            self.gauge_param.use_resident_mom = 1

    def seed(self, state):
        seed = self.random.randrange(2**32)
        backend = getArrayBackend()
        if state is None:
            state = arrayRandomGetState(backend)
            arrayRandomSeed(seed, backend)
        else:
            arrayRandomSetState(state, backend)
        return state

    def samplePhi(self):
        R"""
        \phi\sim e^{-S_f}=e^{-\phi^\dagger\mathcal{M}^{-1}\phi}

        \mathcal{M}^{-\frac{1}{2}}\phi=\eta\sim e^{-\eta^\dagger\eta}
        """
        state = self.seed(None)

        self.loadGaugeMomSmeared()
        for monomial in self.fermion_monomials:
            monomial.sample()
        self.loadGaugeMom()
        if self.hmc_inner is not None:
            self.hmc_inner.samplePhi()

        self.seed(state)

    def momAction(self) -> float:
        # momActionQuda(nullptr, ...) 依赖 resident momentum；这里显式固定 flag，
        # 避免前一次 save/load mom 后 use_resident_mom 状态遗留导致读到空 host 指针。
        self.gauge_param.use_resident_mom = 1
        self.gauge_param.make_resident_mom = 1
        self.gauge_param.return_result_mom = 0
        return momActionQuda(nullptr, self.gauge_param)

    def gaugeAction(self) -> float:
        action = 0
        for monomial in self.gauge_monomials:
            action += monomial.action()
        if self.hmc_inner is not None:
            action += self.hmc_inner.gaugeAction()
        return action

    def fermionAction(self, use_action_param: bool = False) -> float:
        """
        use_action_param: use rational parameters for fermion action istead of molecular dynamics.
        """
        action = 0
        self.loadGaugeMomSmeared()
        for monomial in self.fermion_monomials:
            action += monomial.action()  # if not use_action_param else monomial.actionFA()
        self.loadGaugeMom()
        if self.hmc_inner is not None:
            action += self.hmc_inner.fermionAction()
        return action

    def gaugeForce(self, dt: float):
        for monomial in self.gauge_monomials:
            monomial.force(dt, None)

    def fermionForce(self, dt: float):
        self.loadGaugeMomSmeared()
        for monomial in self.fermion_monomials:
            # 各 action 的 V2 接口对 dt 的历史处理不一致：普通 Clover 会在
            # QUDA 内部缩放，Hasenbusch/logdet 则返回未缩放导数。统一要求这里
            # 先累计 dS/dV，再在 stout reverse 前只乘一次真实积分步长。
            monomial.force_v2(1.0, self.force_v2)
        self.force_v2 *= dt

        self.loadGaugeMom()

    def updateGauge(self, dt: float):
        updateGaugeFieldQuda(nullptr, nullptr, dt, False, True, self.gauge_param)

    def updateMom(self, dt: float, gauge: bool = True, fermion: bool = True):
        if gauge:
            self.gaugeForce(dt)
        if fermion:
            self.fermionForce(dt)

    def forceGradient(self, shifted_force_dt: float, momentum_dt: float):
        """在临时 shifted gauge 上重算当前时间尺度的 force，并恢复原始规范场。"""
        gauge_saved = LatticeGauge(self.latt_info)
        mom_saved = LatticeMom(self.latt_info)
        self.saveGauge(gauge_saved)
        self.saveMom(mom_saved)

        # P=0 后先生成 shifted gauge U'=exp(shifted_force_dt*F[U])U。
        self.loadMom(LatticeMom(self.latt_info))
        self.updateMom(shifted_force_dt)
        self.updateGauge(1.0)

        # 在 U' 上把当前层 force 累加到原动量，再恢复 U；pseudofermion 保持不变。
        self.loadMom(mom_saved)
        self.updateMom(momentum_dt)
        self.loadGauge(gauge_saved)

    def loadGauge(self, gauge: LatticeGauge):
        gauge_in = gauge.copy()
        if self.is_staggered:
            gauge_in.staggeredPhase(False)
        if self.gauge_param.t_boundary == QudaTboundary.QUDA_ANTI_PERIODIC_T:
            gauge_in.setAntiPeriodicT()
        self.gauge_param.use_resident_gauge = 0
        loadGaugeQuda(gauge_in.data_ptrs, self.gauge_param)
        self.gauge_param.use_resident_gauge = 1

    def saveGauge(self, gauge: LatticeGauge):
        saveGaugeQuda(gauge.data_ptrs, self.gauge_param)
        if self.gauge_param.t_boundary == QudaTboundary.QUDA_ANTI_PERIODIC_T:
            gauge.setAntiPeriodicT()
        if self.is_staggered:
            gauge.staggeredPhase(True)

    def loadMom(self, mom: LatticeMom):
        # 从 host momentum 写入 QUDA resident momentum，并让后续 updateGauge/momAction 使用 resident。
        self.gauge_param.use_resident_mom = 0
        self.gauge_param.make_resident_mom = 1
        self.gauge_param.return_result_mom = 0
        momResidentQuda(mom.data_ptrs, self.gauge_param)
        self.gauge_param.use_resident_mom = 1

    def saveMom(self, mom: LatticeMom):
        # momResidentQuda 在 return_result_mom 分支会清空 resident；读回 host 后立刻 reload，
        # 保证 saveMom 只是“保存副本”，不会改变 HMC 后续仍使用 resident momentum 的状态。
        self.gauge_param.use_resident_mom = 1
        self.gauge_param.make_resident_mom = 0
        self.gauge_param.return_result_mom = 1
        momResidentQuda(mom.data_ptrs, self.gauge_param)
        self.gauge_param.make_resident_mom = 1
        self.gauge_param.return_result_mom = 0
        momResidentQuda(mom.data_ptrs, self.gauge_param)  # keep momResident
        self.gauge_param.use_resident_mom = 1

    def gaussMom(self):
        gaussMomQuda(self.random.randrange(2**32), 1.0)
        # gaussMomQuda 生成的是 resident momentum，后续动作都应从 resident 读取。
        self.gauge_param.use_resident_mom = 1
        self.gauge_param.make_resident_mom = 1
        self.gauge_param.return_result_mom = 0

    def projectSU3(self, tol: float):
        gauge = LatticeGauge(self.latt_info)
        self.saveGauge(gauge)
        gauge.projectSU3(tol)
        self.loadGauge(gauge)

    def integrate(self, t: float, project_tol: float = 0.0):
        update_gauge = self.updateGauge if self.hmc_inner is None else self.hmc_inner.integrate
        if isinstance(self.integrator, STSForceGrad):
            self.integrator.integrate(update_gauge, self.updateMom, t, self.forceGradient)
        else:
            self.integrator.integrate(update_gauge, self.updateMom, t)
        if project_tol > 0.0:
            self.projectSU3(project_tol)

    def accept(self, delta_s: float):
        return getMPIComm().bcast(self.random.random() < exp(min(-delta_s, 0)))

    def plaquette(self):
        self.obs_param.compute_plaquette = QudaBoolean.QUDA_BOOLEAN_TRUE
        gaugeObservablesQuda(self.obs_param)
        self.obs_param.compute_plaquette = QudaBoolean.QUDA_BOOLEAN_FALSE
        return self.obs_param.plaquette

    def polyakovLoop(self):
        self.obs_param.compute_polyakov_loop = QudaBoolean.QUDA_BOOLEAN_TRUE
        gaugeObservablesQuda(self.obs_param)
        self.obs_param.compute_polyakov_loop = QudaBoolean.QUDA_BOOLEAN_FALSE
        return self.obs_param.ploop

    def energy(self):
        self.obs_param.compute_qcharge = QudaBoolean.QUDA_BOOLEAN_TRUE
        gaugeObservablesQuda(self.obs_param)
        self.obs_param.compute_qcharge = QudaBoolean.QUDA_BOOLEAN_FALSE
        return self.obs_param.energy

    def qcharge(self):
        self.obs_param.compute_qcharge = QudaBoolean.QUDA_BOOLEAN_TRUE
        gaugeObservablesQuda(self.obs_param)
        self.obs_param.compute_qcharge = QudaBoolean.QUDA_BOOLEAN_FALSE
        return self.obs_param.qcharge

    def qchargeDensity(self):
        qcharge_density = LatticeReal(self.latt_info)
        self.obs_param.qcharge_density = qcharge_density.data_void_ptr
        self.obs_param.compute_qcharge_density = QudaBoolean.QUDA_BOOLEAN_TRUE
        gaugeObservablesQuda(self.obs_param)
        self.obs_param.compute_qcharge_density = QudaBoolean.QUDA_BOOLEAN_TRUE
        return qcharge_density
