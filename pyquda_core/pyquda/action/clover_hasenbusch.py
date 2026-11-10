from typing import Optional

import math
import numpy

from pyquda_comm import getLogger
from ..field import LatticeInfo, LatticeGauge, LatticeMom, LatticeFermion, MultiLatticeFermion
from ..quda import (
    MatQuda,
    cloverQuda,
    computeCloverForceV2Quda,
    computeCloverPairForceV2Quda,
    dslashQuda,
    computeTMCloverForceV2Quda,
    invertQuda,
    loadCloverQuda,
    loadGaugeQuda,
)
from ..gamma import Gamma
from ..enum_quda import (
    QudaDagType,
    QudaDslashType,
    QudaInverterType,
    QudaMassNormalization,
    QudaMatPCType,
    QudaParity,
    QudaSolutionType,
    QudaSolveType,
    QudaTwistFlavorType,
)
from ..dirac import CloverWilsonDirac
from .abstract import FermionAction

nullptr = numpy.empty((0, 0), "<c16")
nullptr_clover = numpy.empty((0), "<c16")


class CloverWilsonLogDetAction(FermionAction):
    """Chroma symmetric EO clover 对角 logdet monomial。

    action 对应 `S=-Nf*(logdet A_even + logdet A_odd)`。QUDA 当前没有公开的
    logdet-force C API；这里复用 `computeCloverForceV2Quda` 的 sigma-trace 分支，
    传入零 pseudofermion 向量，使双线性 force 为零，只保留 clover trace force。
    """

    dirac: CloverWilsonDirac

    def __init__(
        self,
        latt_info: LatticeInfo,
        mass: float,
        n_flavor: int,
        tol: float,
        maxiter: int,
        clover_csw: float,
    ) -> None:
        if latt_info.anisotropy != 1.0:
            getLogger().critical("anisotropy != 1.0 not implemented", NotImplementedError)
        super().__init__(latt_info, CloverWilsonDirac(latt_info, mass, tol, maxiter, clover_csw, 1, None))

        kappa = 1 / (2 * (mass + latt_info.Nd))
        self.coeff = numpy.array([0.0], "<f8")
        self.kappa2 = -(kappa**2)
        self.ck = -kappa * clover_csw / 8
        self.nvector = 1
        self.multiplicity = n_flavor
        # Chroma 的 direct clover 比 QUDA 归一化 clover 多乘 (mass + Nd) = 1/(2*kappa)。
        # logdet 因此需要补 12*V*log(mass+Nd)；该常数不改变 force。
        self.chroma_logdet_constant = (
            -self.multiplicity
            * self.latt_info.Ns
            * self.latt_info.Nc
            * self.latt_info.global_volume
            * math.log(mass + self.latt_info.Nd)
        )
        self.quark = MultiLatticeFermion(latt_info, self.nvector)
        self.force_tmp = LatticeGauge(latt_info)

        self.dirac.setPrecision(False)
        self.invert_param.inv_type = QudaInverterType.QUDA_CG_INVERTER
        self.invert_param.solution_type = QudaSolutionType.QUDA_MATPCDAG_MATPC_SOLUTION
        self.invert_param.solve_type = QudaSolveType.QUDA_NORMOP_PC_SOLVE
        self.invert_param.mass_normalization = QudaMassNormalization.QUDA_KAPPA_NORMALIZATION

    def updateClover(self):
        loadGaugeQuda(nullptr, self.gauge_param)
        loadCloverQuda(nullptr_clover, nullptr_clover, self.invert_param)

    def sample(self):
        # logdet monomial 没有 pseudofermion；零向量只用于屏蔽 QUDA force 的双线性项。
        self.quark.data = 0

    def action(self) -> float:
        self.invert_param.compute_clover_trlog = 1
        self.updateClover()
        self.invert_param.compute_clover_trlog = 0
        return (
            -self.multiplicity * (self.invert_param.trlogA[0] + self.invert_param.trlogA[1])
            + self.chroma_logdet_constant
        )

    def _add_sigma_trace_force(self, force: LatticeGauge, matpc_type: QudaMatPCType) -> None:
        old_matpc = self.invert_param.matpc_type
        old_dagger = self.invert_param.dagger
        self.invert_param.matpc_type = matpc_type
        self.invert_param.dagger = QudaDagType.QUDA_DAG_YES
        self.force_tmp.data = 0
        computeCloverForceV2Quda(
            self.force_tmp.data_ptrs,
            1.0,
            self.quark.even_ptrs,
            self.coeff,
            self.kappa2,
            self.ck,
            self.nvector,
            self.multiplicity,
            self.gauge_param,
            self.invert_param,
        )
        force.data += self.force_tmp.data
        self.invert_param.dagger = old_dagger
        self.invert_param.matpc_type = old_matpc

    def force_v2(self, dt, force: Optional[LatticeGauge] = None):
        self.updateClover()
        self.quark.data = 0
        if force is not None:
            self.gauge_param.use_resident_force = 0
            self.gauge_param.return_result_force = 1
            # `computeCloverForce` 的 sigma trace 作用在 active parity 的另一半；
            # 因此分别调用 even-even / odd-odd asymmetric 来覆盖 A_odd 和 A_even。
            self._add_sigma_trace_force(force, QudaMatPCType.QUDA_MATPC_EVEN_EVEN_ASYMMETRIC)
            self._add_sigma_trace_force(force, QudaMatPCType.QUDA_MATPC_ODD_ODD_ASYMMETRIC)
            self.gauge_param.use_resident_force = 1
            self.gauge_param.return_result_force = 0
        else:
            self.invert_param.matpc_type = QudaMatPCType.QUDA_MATPC_EVEN_EVEN_ASYMMETRIC
            computeCloverForceV2Quda(
                nullptr,
                dt,
                self.quark.even_ptrs,
                self.coeff,
                self.kappa2,
                self.ck,
                self.nvector,
                self.multiplicity,
                self.gauge_param,
                self.invert_param,
            )

    def force(self, dt, mom: Optional[LatticeMom] = None):
        getLogger().critical(
            "CloverWilsonLogDetAction.force is not available; use force_v2 through stout HMC",
            NotImplementedError,
        )


class CloverWilsonHasenbuschTwistAction(FermionAction):
    """Clover Wilson pseudofermion action using QUDA's Hasenbusch-twist operator.

    该类支持两种 monomial：
    1. cancel determinant：单个 shifted operator M(mu)，对应 Chroma 的 multihasen cancel 项。
    2. determinant ratio：相邻 shifted operator M(mu) / M(mu_prec)，按 Chroma ratio monomial
       的 refresh/action/force 数据流构造 RHS，再调用 QUDA 的 detratio force kernel。
    """

    dirac: CloverWilsonDirac

    def __init__(
        self,
        latt_info: LatticeInfo,
        mass: float,
        mu: float,
        tol: float,
        maxiter: int,
        clover_csw: float,
        *,
        md_tol: Optional[float] = None,
        action_tol: Optional[float] = None,
        determinant_ratio: bool = False,
        constdet: bool = False,
        mu_prec: Optional[float] = None,
        force_sign: float = 1.0,
        force_coeff_scale: float = 1.0,
        force_output_scale: float = 1.0,
        force_dagger: str = "yes",
        ratio_force_source: str = "phi",
        ratio_force_mode: str = "quda_detratio",
        cancel_force_mode: str = "quda_tmclover",
        cancel_sample_scale: float = 1.0,
        sym_odd_clover_scale: float = 1.0,
        matpc_type: str = "even_even_asymmetric",
        label: str = "",
    ) -> None:
        if latt_info.anisotropy != 1.0:
            getLogger().critical("anisotropy != 1.0 not implemented", NotImplementedError)
        super().__init__(latt_info, CloverWilsonDirac(latt_info, mass, tol, maxiter, clover_csw, 1, None))

        self.mass = mass
        self.mu = mu
        self.mu_prec = mu_prec
        # Chroma 分别配置 MD force/refresh 与接受 Hamiltonian 的残差目标。
        self.md_tol = float(tol if md_tol is None else md_tol)
        self.action_tol = float(tol if action_tol is None else action_tol)
        self.determinant_ratio = determinant_ratio
        # Chroma 的 multihasen cancel 使用 const-det 预条件化：不包含 clover trlog/sigma-trace。
        self.constdet = constdet
        self.force_sign = force_sign
        self.force_coeff_scale = force_coeff_scale
        self.force_output_scale = force_output_scale
        self.force_dagger = (
            QudaDagType.QUDA_DAG_YES if force_dagger == "yes" else QudaDagType.QUDA_DAG_NO
        )
        self.ratio_force_source = ratio_force_source
        self.ratio_force_mode = ratio_force_mode
        # cancel_force_mode=pc_pair 保留 Chroma 系数探针；pc_quda_norm 使用 QUDA MatPC 真实归一化。
        self.cancel_force_mode = cancel_force_mode
        # Chroma multihasen cancel refresh 中 eta 乘 sqrt(0.5)；默认 1.0 保持旧行为。
        self.cancel_sample_scale = float(cancel_sample_scale)
        # 诊断旋钮：只缩放 symmetric odd-odd 展开中的 active odd clover 导数项。
        # 默认 1.0 保持原实现，用于定位 A_oo insertion 的 QUDA/Chroma 归一化差异。
        self.sym_odd_clover_scale = float(sym_odd_clover_scale)
        kappa = 1 / (2 * (mass + latt_info.Nd))
        # QUDA MatPC 的 Clover/Wilson force 基与 Chroma direct clover 相差
        # clover_csw/kappa = 1/v0^3；pc_quda_norm 的 base 和 shift 项都需要该因子。
        self.quda_norm_force_scale = clover_csw / kappa
        matpc_map = {
            "even_even_asymmetric": QudaMatPCType.QUDA_MATPC_EVEN_EVEN_ASYMMETRIC,
            "odd_odd_asymmetric": QudaMatPCType.QUDA_MATPC_ODD_ODD_ASYMMETRIC,
            "even_even": QudaMatPCType.QUDA_MATPC_EVEN_EVEN,
            "odd_odd": QudaMatPCType.QUDA_MATPC_ODD_ODD,
        }
        if matpc_type not in matpc_map:
            getLogger().critical(
                "matpc_type must be even_even_asymmetric, odd_odd_asymmetric, even_even or odd_odd", ValueError
            )
        self.matpc_type_label = matpc_type
        self.matpc_type = matpc_map[matpc_type]
        if matpc_type == "even_even":
            raise NotImplementedError(
                "symmetric even_even Hasenbusch force is not implemented; use odd_odd or an asymmetric MatPC"
            )
        if not determinant_ratio and not constdet and matpc_type == "odd_odd":
            raise NotImplementedError(
                "non-constdet symmetric odd_odd action has no validated clover trlog decomposition"
            )
        if self.determinant_ratio and self.mu_prec is None:
            getLogger().critical("determinant_ratio=True requires mu_prec", ValueError)
        if self.determinant_ratio and self.constdet:
            getLogger().critical("determinant_ratio and constdet cannot both be True", ValueError)
        if self.ratio_force_source not in ("phi", "rhs"):
            getLogger().critical("ratio_force_source must be 'phi' or 'rhs'", ValueError)
        if self.ratio_force_mode not in ("quda_detratio", "pair", "pc_pair", "pc_quda_twist", "pc_quda_norm"):
            getLogger().critical(
                "ratio_force_mode must be quda_detratio, pair, pc_pair, pc_quda_twist or pc_quda_norm",
                ValueError,
            )
        # QUDA detratio 接口的 x0 有固定含义：它必须是 ratio 的源 pseudofermion phi，
        # 不能把求解右端 rhs 当作 x0；rhs 只属于 Python 侧的显式 PC 展开诊断。
        if self.ratio_force_mode == "quda_detratio" and self.ratio_force_source != "phi":
            raise ValueError("quda_detratio requires ratio_force_source='phi'; rhs is not a valid QUDA x0")
        if self.cancel_force_mode not in ("quda_tmclover", "pc_pair", "pc_constdet", "pc_quda_twist", "pc_quda_norm"):
            getLogger().critical(
                "cancel_force_mode must be quda_tmclover, pc_pair, pc_constdet, pc_quda_twist or pc_quda_norm",
                ValueError,
            )
        if force_dagger not in ("yes", "no"):
            getLogger().critical("force_dagger must be 'yes' or 'no'", ValueError)
        self.label = label or ("ratio" if determinant_ratio else "det")
        # QUDA TMClover force test 使用 4*kappa^2 作为 Hasenbusch/twisted-clover force coeff。
        # 这里的 coeff 只进入 force kernel；pseudofermion refresh/action 仍由 M(mu) 和 M(mu_prec) 定义。
        self.coeff = numpy.array(
            [
                self.force_sign
                * self.force_coeff_scale
                * 4.0
                * self.invert_param.kappa
                * self.invert_param.kappa
            ],
            "<f8",
        )
        k_csw_ov_8 = self.invert_param.kappa * clover_csw / 8.0
        self.ck = -k_csw_ov_8
        self.epsilon_even = numpy.array([k_csw_ov_8 * self.coeff[0]], "<f8")
        self.epsilon_odd = numpy.array([k_csw_ov_8 * self.coeff[0] / (self.invert_param.kappa**2)], "<f8")
        self.nvector = 1
        self.multiplicity = 2

        self.quark = MultiLatticeFermion(self.latt_info, self.nvector)
        self.source = MultiLatticeFermion(self.latt_info, self.nvector)
        self.pair_x = MultiLatticeFermion(self.latt_info, self.nvector)
        self.pair_p = MultiLatticeFermion(self.latt_info, self.nvector)
        self.phi = LatticeFermion(latt_info)
        self.eta = LatticeFermion(latt_info)
        self.rhs = LatticeFermion(latt_info)
        self.tmp = LatticeFermion(latt_info)
        self.y = LatticeFermion(latt_info)
        # pc_pair 模式使用这些临时场显式构造 Chroma e/o 预条件化导数四项。
        self.pc_tmp1 = LatticeFermion(latt_info)
        self.pc_tmp2 = LatticeFermion(latt_info)
        self.pc_tmp3 = LatticeFermion(latt_info)
        self.pc_tmp4 = LatticeFermion(latt_info)
        self.pc_tmp5 = LatticeFermion(latt_info)
        self.force_tmp = LatticeGauge(latt_info)
        # 调试用：按单个 bilinear 项过滤 force，便于有限差分逐项定位。
        self.force_term_filter: Optional[str] = None
        self.force_term_history = []
        # HMC 长跑默认关闭逐 pair INFO，诊断脚本仍可读取 force_term_history。
        self.force_debug_logging = False

        self.dirac.setPrecision(False)
        self.invert_param.inv_type = QudaInverterType.QUDA_CG_INVERTER
        self.invert_param.solution_type = QudaSolutionType.QUDA_MATPCDAG_MATPC_SOLUTION
        self.invert_param.solve_type = QudaSolveType.QUDA_NORMOP_PC_SOLVE
        self.invert_param.matpc_type = self.matpc_type
        self.invert_param.mass_normalization = QudaMassNormalization.QUDA_KAPPA_NORMALIZATION
        self.invert_param.dslash_type = QudaDslashType.QUDA_CLOVER_HASENBUSCH_TWIST_DSLASH
        self.invert_param.mu = mu
        self.invert_param.twist_flavor = QudaTwistFlavorType.QUDA_TWIST_SINGLET

    def updateClover(self):
        loadGaugeQuda(nullptr, self.gauge_param)
        loadCloverQuda(nullptr_clover, nullptr_clover, self.invert_param)

    def _mat(self, out: LatticeFermion, inp: LatticeFermion, dagger: QudaDagType, mu: Optional[float] = None):
        old_mu = self.invert_param.mu
        old_dagger = self.invert_param.dagger
        if mu is not None:
            self.invert_param.mu = mu
        self.invert_param.dagger = dagger
        MatQuda(out.even_ptr, inp.even_ptr, self.invert_param)
        self.invert_param.dagger = old_dagger
        self.invert_param.mu = old_mu

    def _solve(self, rhs: Optional[LatticeFermion] = None, mu: Optional[float] = None, compute_action: bool = False):
        old_mu = self.invert_param.mu
        old_tol = self.invert_param.tol
        if mu is not None:
            self.invert_param.mu = mu
        # action 用严格残差保证 Metropolis 能量；refresh/force 使用头文件的 MD 残差。
        self.invert_param.tol = self.action_tol if compute_action else self.md_tol
        self.invert_param.compute_action = int(compute_action)
        invertQuda(self.quark[0].even_ptr, (rhs or self.phi).even_ptr, self.invert_param)
        self.dirac.performance()
        self.invert_param.compute_action = 0
        self.invert_param.tol = old_tol
        self.invert_param.mu = old_mu

    def sample(self):
        self.sampleEta()
        self.updateClover()
        if self.determinant_ratio:
            # Chroma ratio refresh:
            # eta_tmp = M(mu)^dag eta; phi = M(mu_prec) [M(mu_prec)^dag M(mu_prec)]^-1 eta_tmp.
            self._mat(self.tmp, self.eta, QudaDagType.QUDA_DAG_YES, self.mu)
            self._solve(self.tmp, self.mu_prec, compute_action=False)
            self._mat(self.phi, self.quark[0], QudaDagType.QUDA_DAG_NO, self.mu_prec)
        else:
            self._mat(self.phi, self.eta, QudaDagType.QUDA_DAG_YES, self.mu)
            if self.cancel_sample_scale != 1.0:
                self.phi.data *= self.cancel_sample_scale
        self.source[0] = self.phi

    def action(self) -> float:
        if self.determinant_ratio:
            # S = phi^dag M(mu_prec) [M(mu)^dag M(mu)]^-1 M(mu_prec)^dag phi.
            self.updateClover()
            self._mat(self.rhs, self.phi, QudaDagType.QUDA_DAG_YES, self.mu_prec)
            self._solve(self.rhs, self.mu, compute_action=True)
            return self.invert_param.action[0]
        else:
            if not self.constdet:
                # 普通 twisted-clover determinant force 包含 clover sigma-trace；action 也要包含 trlogA。
                self.invert_param.compute_clover_trlog = 1
                self.updateClover()
                self.invert_param.compute_clover_trlog = 0
            else:
                self.updateClover()
            self._solve(self.phi, self.mu, compute_action=True)
            if not self.constdet:
                # 非 const-det 分支保留原来的零点扣除和 trlog 项。
                action = (
                    self.invert_param.action[0]
                    - self.latt_info.global_volume // 2 * self.latt_info.Ns * self.latt_info.Nc
                )
                # asymmetric Schur complement 要补回被消去 checkerboard 的 clover determinant。
                trlog_index = 1 if self._active_parity() == QudaParity.QUDA_EVEN_PARITY else 0
                action -= self.multiplicity * self.invert_param.trlogA[trlog_index]
            else:
                # Chroma const-det cancel monomial 没有零点常数扣除。
                action = self.invert_param.action[0]
            return action

    def _parity_field(self, field: LatticeFermion, parity: QudaParity) -> LatticeFermion:
        # 返回只含一个 checkerboard 的全场；pair force 用它隔离 Chroma 的 PC block。
        out = LatticeFermion(self.latt_info)
        out.data = 0
        if parity == QudaParity.QUDA_EVEN_PARITY:
            out.even = field.even
        elif parity == QudaParity.QUDA_ODD_PARITY:
            out.odd = field.odd
        else:
            getLogger().critical("_parity_field only accepts even/odd parity", ValueError)
        return out

    def _active_pc_field(self, field: LatticeFermion) -> LatticeFermion:
        # QUDA MatPC 输入/输出总是经 even_ptr 传递；odd_odd 时要把这半格搬到物理 odd checkerboard。
        active = self._active_parity()
        out = LatticeFermion(self.latt_info)
        out.data = 0
        if active == QudaParity.QUDA_EVEN_PARITY:
            out.even = field.even
        elif active == QudaParity.QUDA_ODD_PARITY:
            out.odd = field.even
        else:
            getLogger().critical("_active_pc_field only accepts even/odd active parity", ValueError)
        return out

    def _dslash_block(
        self,
        out: LatticeFermion,
        inp: LatticeFermion,
        out_parity: QudaParity,
        dagger: QudaDagType,
        scale: float = 1.0,
        preconditioned: bool = False,
    ) -> None:
        # preconditioned=False 对应 Chroma 裸 Wilson D.apply；True 对应 QUDA clover PC 的 A^{-1}D。
        old_dagger = self.invert_param.dagger
        old_dslash_type = self.invert_param.dslash_type
        self.invert_param.dagger = dagger
        self.invert_param.dslash_type = (
            QudaDslashType.QUDA_CLOVER_WILSON_DSLASH
            if preconditioned
            else QudaDslashType.QUDA_WILSON_DSLASH
        )
        out.data = 0
        if out_parity == QudaParity.QUDA_EVEN_PARITY:
            dslashQuda(out.even_ptr, inp.odd_ptr, self.invert_param, QudaParity.QUDA_EVEN_PARITY)
            out.even = out.even * scale
        elif out_parity == QudaParity.QUDA_ODD_PARITY:
            dslashQuda(out.odd_ptr, inp.even_ptr, self.invert_param, QudaParity.QUDA_ODD_PARITY)
            out.odd = out.odd * scale
        else:
            getLogger().critical("_dslash_block only accepts even/odd parity", ValueError)
        self.invert_param.dslash_type = old_dslash_type
        self.invert_param.dagger = old_dagger

    def _active_parity(self) -> QudaParity:
        # Chroma 的 SEOPREC Hasenbusch 作用在 odd checkerboard；这里允许显式选择 symmetric/asymmetric。
        if self.invert_param.matpc_type in (
            QudaMatPCType.QUDA_MATPC_EVEN_EVEN,
            QudaMatPCType.QUDA_MATPC_EVEN_EVEN_ASYMMETRIC,
        ):
            return QudaParity.QUDA_EVEN_PARITY
        if self.invert_param.matpc_type in (
            QudaMatPCType.QUDA_MATPC_ODD_ODD,
            QudaMatPCType.QUDA_MATPC_ODD_ODD_ASYMMETRIC,
        ):
            return QudaParity.QUDA_ODD_PARITY
        getLogger().critical("unsupported matpc_type for Hasenbusch force probe", NotImplementedError)

    def _other_parity(self, parity: QudaParity) -> QudaParity:
        if parity == QudaParity.QUDA_EVEN_PARITY:
            return QudaParity.QUDA_ODD_PARITY
        if parity == QudaParity.QUDA_ODD_PARITY:
            return QudaParity.QUDA_EVEN_PARITY
        getLogger().critical("_other_parity only accepts even/odd parity", ValueError)

    def _is_symmetric_matpc(self) -> bool:
        return self.invert_param.matpc_type in (
            QudaMatPCType.QUDA_MATPC_EVEN_EVEN,
            QudaMatPCType.QUDA_MATPC_ODD_ODD,
        )

    def _clover_inv_parity(
        self, out: LatticeFermion, inp: LatticeFermion, parity: QudaParity, dagger: QudaDagType
    ) -> None:
        # Chroma 的 A^{-1} block 是未 shifted 的 base clover inverse，作用在 inactive checkerboard。
        old_dagger = self.invert_param.dagger
        old_dslash_type = self.invert_param.dslash_type
        self.invert_param.dagger = dagger
        self.invert_param.dslash_type = QudaDslashType.QUDA_CLOVER_WILSON_DSLASH
        out.data = 0
        if parity == QudaParity.QUDA_EVEN_PARITY:
            cloverQuda(out.even_ptr, inp.even_ptr, self.invert_param, parity, 1)
        elif parity == QudaParity.QUDA_ODD_PARITY:
            cloverQuda(out.odd_ptr, inp.odd_ptr, self.invert_param, parity, 1)
        else:
            getLogger().critical("_clover_inv_parity only accepts even/odd parity", ValueError)
        self.invert_param.dslash_type = old_dslash_type
        self.invert_param.dagger = old_dagger

    def _clover_parity(
        self, out: LatticeFermion, inp: LatticeFermion, parity: QudaParity, dagger: QudaDagType
    ) -> None:
        # QUDA Hasenbusch twist active block 含 A 和 A^2；这里显式应用 base clover A。
        old_dagger = self.invert_param.dagger
        old_dslash_type = self.invert_param.dslash_type
        self.invert_param.dagger = dagger
        self.invert_param.dslash_type = QudaDslashType.QUDA_CLOVER_WILSON_DSLASH
        out.data = 0
        if parity == QudaParity.QUDA_EVEN_PARITY:
            cloverQuda(out.even_ptr, inp.even_ptr, self.invert_param, parity, 0)
        elif parity == QudaParity.QUDA_ODD_PARITY:
            cloverQuda(out.odd_ptr, inp.odd_ptr, self.invert_param, parity, 0)
        else:
            getLogger().critical("_clover_parity only accepts even/odd parity", ValueError)
        self.invert_param.dslash_type = old_dslash_type
        self.invert_param.dagger = old_dagger

    def _i_gamma5_parity(
        self, out: LatticeFermion, inp: LatticeFermion, parity: QudaParity, scale: complex
    ) -> None:
        # 对单个 checkerboard 应用 scale * i * gamma5；用于 QUDA A^2 twist 项的候选导数。
        gamma5 = Gamma(15)
        out.data = 0
        if parity == QudaParity.QUDA_EVEN_PARITY:
            for spin in range(4):
                src_spin = gamma5.sparse_indices[spin]
                out.even.data[..., spin, :] = scale * 1j * gamma5.sparse_data[spin] * inp.even.data[..., src_spin, :]
        elif parity == QudaParity.QUDA_ODD_PARITY:
            for spin in range(4):
                src_spin = gamma5.sparse_indices[spin]
                out.odd.data[..., spin, :] = scale * 1j * gamma5.sparse_data[spin] * inp.odd.data[..., src_spin, :]
        else:
            getLogger().critical("_i_gamma5_parity only accepts even/odd parity", ValueError)

    def clear_force_term_history(self) -> None:
        # 清空最近一次 force_v2 记录的 bilinear 项列表；不影响物理路径。
        self.force_term_history = []

    def set_force_term_filter(self, term_label: Optional[str]) -> None:
        # term_label=None 表示不过滤；否则只计算匹配 full/raw label 的单项 force。
        self.force_term_filter = term_label
        self.clear_force_term_history()

    def _add_quda_active_twist_force(
        self,
        force: LatticeGauge,
        chi_active: LatticeFermion,
        psi_active: LatticeFermion,
        dagger: QudaDagType,
        sign: float,
        label: str,
    ) -> None:
        # QUDA Hasenbusch twist 的 active block 是 A*x + i*mu*gamma5*A^2*x；
        # Chroma asymmetric shift 是 i*mu*gamma5*x，因此这两项只用于 QUDA 自洽 force 探针。
        if self.invert_param.mu == 0.0:
            return
        active = self._active_parity()
        bdag = 0.5 * self.invert_param.mu if dagger == QudaDagType.QUDA_DAG_YES else -0.5 * self.invert_param.mu

        self._i_gamma5_parity(self.pc_tmp4, chi_active, active, bdag)
        self._clover_parity(self.pc_tmp5, psi_active, active, dagger)
        self._add_pair_force_v2(
            force,
            self._parity_field(self.pc_tmp4, active),
            self._parity_field(self.pc_tmp5, active),
            sign,
            f"{label}: quda twist dA Apsi",
        )

        self._clover_parity(self.pc_tmp5, self.pc_tmp4, active, dagger)
        self._add_pair_force_v2(
            force,
            self._parity_field(self.pc_tmp5, active),
            psi_active,
            sign,
            f"{label}: quda twist A dA psi",
        )

    def _add_eoprec_deriv_force(
        self,
        force: LatticeGauge,
        chi: LatticeFermion,
        psi: LatticeFermion,
        dagger: QudaDagType,
        sign: float,
        label: str,
        include_quda_twist: bool = False,
    ) -> None:
        # 复刻 e/o PC 导数：A'_aa - D'_ai Aii^-1 D_ia + Aii' 插入项 - D'_ia。
        active = self._active_parity()
        inactive = self._other_parity(active)
        msign = QudaDagType.QUDA_DAG_NO if dagger == QudaDagType.QUDA_DAG_YES else QudaDagType.QUDA_DAG_YES
        if active == QudaParity.QUDA_ODD_PARITY:
            # QUDA MatPC half-vector 总通过 even_ptr 暴露；odd_odd 时要先搬到物理 odd。
            chi_active = self._active_pc_field(chi)
            psi_active = self._active_pc_field(psi)
        else:
            chi_active = self._parity_field(chi, active)
            psi_active = self._parity_field(psi, active)

        self._add_pair_force_v2(force, chi_active, psi_active, sign, f"{label}: A_active prime")
        if include_quda_twist:
            self._add_quda_active_twist_force(force, chi_active, psi_active, dagger, sign, label)
        self._dslash_block(self.pc_tmp1, psi_active, inactive, dagger, -0.5)
        self._clover_inv_parity(self.pc_tmp2, self.pc_tmp1, inactive, dagger)
        self._add_pair_force_v2(
            force,
            chi_active,
            self._parity_field(self.pc_tmp2, inactive),
            -sign * 0.5,
            f"{label}: -D_active,inactive prime Aii^-1 D_inactive,active",
        )

        self._dslash_block(self.pc_tmp1, chi_active, inactive, msign, -0.5)
        self._clover_inv_parity(self.pc_tmp3, self.pc_tmp1, inactive, msign)
        self._add_pair_force_v2(
            force,
            self._parity_field(self.pc_tmp3, inactive),
            self._parity_field(self.pc_tmp2, inactive),
            sign,
            f"{label}: +A_inactive prime insertion",
        )

        self._add_pair_force_v2(
            force,
            self._parity_field(self.pc_tmp3, inactive),
            psi_active,
            -sign * 0.5,
            f"{label}: -D_inactive,active prime",
        )

    def _add_eoprec_constdet_deriv_force(
        self,
        force: LatticeGauge,
        chi: LatticeFermion,
        psi: LatticeFermion,
        dagger: QudaDagType,
        sign: float,
        label: str,
    ) -> None:
        """只实现 Chroma EvenOddPrecConstDetLinearOperator::deriv 的两项 D-link 导数。

        const-det 预条件化把 inactive checkerboard 的 det A 视为 gauge-independent 常数，
        因此这里不加入 A_active prime 或 A_inactive prime insertion 项。
        """
        active = self._active_parity()
        inactive = self._other_parity(active)
        msign = QudaDagType.QUDA_DAG_NO if dagger == QudaDagType.QUDA_DAG_YES else QudaDagType.QUDA_DAG_YES
        if active == QudaParity.QUDA_ODD_PARITY:
            # const-det 分支同样面对 QUDA half-vector 布局，不能直接读 field.odd。
            chi_active = self._active_pc_field(chi)
            psi_active = self._active_pc_field(psi)
        else:
            chi_active = self._parity_field(chi, active)
            psi_active = self._parity_field(psi, active)

        self._dslash_block(self.pc_tmp1, psi_active, inactive, dagger, -0.5)
        self._clover_inv_parity(self.pc_tmp2, self.pc_tmp1, inactive, dagger)
        self._add_pair_force_v2(
            force,
            chi_active,
            self._parity_field(self.pc_tmp2, inactive),
            -sign * 0.5,
            f"{label}: constdet -D_active,inactive prime Aii^-1 D_inactive,active",
            include_clover=False,
        )

        self._dslash_block(self.pc_tmp1, chi_active, inactive, msign, -0.5)
        self._clover_inv_parity(self.pc_tmp3, self.pc_tmp1, inactive, msign)
        self._add_pair_force_v2(
            force,
            self._parity_field(self.pc_tmp3, inactive),
            psi_active,
            -sign * 0.5,
            f"{label}: constdet -D_inactive,active prime",
            include_clover=False,
        )

    def _add_sym_oddodd_chroma_deriv_force(
        self,
        force: LatticeGauge,
        chi: LatticeFermion,
        psi: LatticeFermion,
        dagger: QudaDagType,
        sign: float,
        label: str,
        include_shift: bool = True,
    ) -> None:
        """按 Chroma SymEvenOddPrecCloverLinOp::deriv 展开 odd_odd symmetric force。

        chi/psi 是 QUDA MatPC half-vector，物理 odd 数据仍存放在 even_ptr；
        这里先搬到 odd checkerboard，再显式展开 A^{-1}、裸 Wilson D 和 clover A' 项。
        """
        active = self._active_parity()
        if active != QudaParity.QUDA_ODD_PARITY:
            getLogger().critical("symmetric Chroma force branch currently supports only odd_odd", NotImplementedError)

        plus = dagger == QudaDagType.QUDA_DAG_NO
        chi_odd = self._active_pc_field(chi)
        psi_odd = self._active_pc_field(psi)

        if plus:
            # M_eo_psi = -0.5 * A_ee^{-1} D_eo psi。
            self._dslash_block(self.pc_tmp1, psi_odd, QudaParity.QUDA_EVEN_PARITY, QudaDagType.QUDA_DAG_NO)
            self._clover_inv_parity(self.pc_tmp2, self.pc_tmp1, QudaParity.QUDA_EVEN_PARITY, QudaDagType.QUDA_DAG_NO)
            self.pc_tmp2.even = self.pc_tmp2.even * -0.5

            # M_oe^dag_chi = -0.5 * D_oe^dag A_oo^{-dag} chi。
            self._clover_inv_parity(self.pc_tmp3, chi_odd, QudaParity.QUDA_ODD_PARITY, QudaDagType.QUDA_DAG_YES)
            self._dslash_block(
                self.pc_tmp4,
                self.pc_tmp3,
                QudaParity.QUDA_EVEN_PARITY,
                QudaDagType.QUDA_DAG_YES,
                -0.5,
            )

            # active odd A^{-1} 导数项；Chroma 最后整体乘 -0.5。
            self._dslash_block(self.pc_tmp1, self.pc_tmp2, QudaParity.QUDA_ODD_PARITY, QudaDagType.QUDA_DAG_NO)
            self._clover_inv_parity(self.pc_tmp5, self.pc_tmp1, QudaParity.QUDA_ODD_PARITY, QudaDagType.QUDA_DAG_NO)
            self._add_pair_force_v2(
                force,
                self._parity_field(self.pc_tmp3, QudaParity.QUDA_ODD_PARITY),
                self._parity_field(self.pc_tmp5, QudaParity.QUDA_ODD_PARITY),
                sign * -0.5 * self.sym_odd_clover_scale,
                f"{label}: sym PLUS A_oo^-1 insertion",
                include_hopping=False,
                swap_pair_fields=not plus,
            )

            # D_oe 导数项：pre-final 为负号，乘最终 -0.5 后为 +0.5。
            self._add_pair_force_v2(
                force,
                self._parity_field(self.pc_tmp3, QudaParity.QUDA_ODD_PARITY),
                self._parity_field(self.pc_tmp2, QudaParity.QUDA_EVEN_PARITY),
                sign * 0.5,
                f"{label}: sym PLUS D_oe prime",
                include_clover=False,
                swap_pair_fields=not plus,
            )

            # inactive even A^{-1} 导数项。
            self._dslash_block(self.pc_tmp1, psi_odd, QudaParity.QUDA_EVEN_PARITY, QudaDagType.QUDA_DAG_NO)
            self._clover_inv_parity(self.pc_tmp5, self.pc_tmp1, QudaParity.QUDA_EVEN_PARITY, QudaDagType.QUDA_DAG_NO)
            self._clover_inv_parity(self.pc_tmp3, self.pc_tmp4, QudaParity.QUDA_EVEN_PARITY, QudaDagType.QUDA_DAG_NO)
            self._add_pair_force_v2(
                force,
                self._parity_field(self.pc_tmp3, QudaParity.QUDA_EVEN_PARITY),
                self._parity_field(self.pc_tmp5, QudaParity.QUDA_EVEN_PARITY),
                sign * -0.5,
                f"{label}: sym PLUS A_ee^-1 insertion",
                include_hopping=False,
                swap_pair_fields=not plus,
            )

            # D_eo 导数项。
            self._clover_inv_parity(self.pc_tmp3, self.pc_tmp4, QudaParity.QUDA_EVEN_PARITY, QudaDagType.QUDA_DAG_YES)
            self._add_pair_force_v2(
                force,
                self._parity_field(self.pc_tmp3, QudaParity.QUDA_EVEN_PARITY),
                psi_odd,
                sign * 0.5,
                f"{label}: sym PLUS D_eo prime",
                include_clover=False,
                swap_pair_fields=not plus,
            )
        else:
            # M_eo_psi = -0.5 * D_eo^dag A_oo^{-dag} psi。
            self._clover_inv_parity(self.pc_tmp1, psi_odd, QudaParity.QUDA_ODD_PARITY, QudaDagType.QUDA_DAG_YES)
            self._dslash_block(
                self.pc_tmp2,
                self.pc_tmp1,
                QudaParity.QUDA_EVEN_PARITY,
                QudaDagType.QUDA_DAG_YES,
                -0.5,
            )

            # M_oe^dag_chi = -0.5 * A_ee^{-1} D_eo chi。
            self._dslash_block(self.pc_tmp1, chi_odd, QudaParity.QUDA_EVEN_PARITY, QudaDagType.QUDA_DAG_NO)
            self._clover_inv_parity(self.pc_tmp4, self.pc_tmp1, QudaParity.QUDA_EVEN_PARITY, QudaDagType.QUDA_DAG_NO)
            self.pc_tmp4.even = self.pc_tmp4.even * -0.5

            # inactive even A^{-1} 导数项。
            self._dslash_block(self.pc_tmp1, chi_odd, QudaParity.QUDA_EVEN_PARITY, QudaDagType.QUDA_DAG_NO)
            self._clover_inv_parity(self.pc_tmp5, self.pc_tmp1, QudaParity.QUDA_EVEN_PARITY, QudaDagType.QUDA_DAG_NO)
            self._clover_inv_parity(self.pc_tmp3, self.pc_tmp2, QudaParity.QUDA_EVEN_PARITY, QudaDagType.QUDA_DAG_YES)
            self._add_pair_force_v2(
                force,
                self._parity_field(self.pc_tmp5, QudaParity.QUDA_EVEN_PARITY),
                self._parity_field(self.pc_tmp3, QudaParity.QUDA_EVEN_PARITY),
                sign * -0.5,
                f"{label}: sym MINUS A_ee^-1 insertion",
                include_hopping=False,
                swap_pair_fields=not plus,
            )

            # D_oe^dag 导数项。
            self._add_pair_force_v2(
                force,
                chi_odd,
                self._parity_field(self.pc_tmp3, QudaParity.QUDA_EVEN_PARITY),
                sign * 0.5,
                f"{label}: sym MINUS D_oe prime",
                include_clover=False,
                swap_pair_fields=not plus,
            )

            # D_eo^dag 导数项。
            self._clover_inv_parity(self.pc_tmp1, psi_odd, QudaParity.QUDA_ODD_PARITY, QudaDagType.QUDA_DAG_YES)
            self._add_pair_force_v2(
                force,
                self._parity_field(self.pc_tmp4, QudaParity.QUDA_EVEN_PARITY),
                self._parity_field(self.pc_tmp1, QudaParity.QUDA_ODD_PARITY),
                sign * 0.5,
                f"{label}: sym MINUS D_eo prime",
                include_clover=False,
                swap_pair_fields=not plus,
            )

            # active odd A^{-1} 导数项。
            self._dslash_block(self.pc_tmp3, self.pc_tmp4, QudaParity.QUDA_ODD_PARITY, QudaDagType.QUDA_DAG_NO)
            self._clover_inv_parity(self.pc_tmp5, self.pc_tmp3, QudaParity.QUDA_ODD_PARITY, QudaDagType.QUDA_DAG_NO)
            self._clover_inv_parity(self.pc_tmp1, psi_odd, QudaParity.QUDA_ODD_PARITY, QudaDagType.QUDA_DAG_YES)
            self._add_pair_force_v2(
                force,
                self._parity_field(self.pc_tmp5, QudaParity.QUDA_ODD_PARITY),
                self._parity_field(self.pc_tmp1, QudaParity.QUDA_ODD_PARITY),
                sign * -0.5 * self.sym_odd_clover_scale,
                f"{label}: sym MINUS A_oo^-1 insertion",
                include_hopping=False,
                swap_pair_fields=not plus,
            )

        if include_shift:
            self._add_symmetric_shift_deriv_force(
                force,
                chi_odd,
                psi_odd,
                dagger,
                sign,
                f"{label}: sym shifted i_mu_gamma5_A_oo prime",
                self.sym_odd_clover_scale,
            )

    def _add_symmetric_shift_deriv_force(
        self,
        force: LatticeGauge,
        chi_odd: LatticeFermion,
        psi_odd: LatticeFermion,
        dagger: QudaDagType,
        sign: float,
        label: str,
        scale: float = 1.0,
    ) -> None:
        """加入 symmetric odd-odd 中 ``delta[(+-) i mu gamma5 A_oo / 2]`` 的单项导数。

        该项只对 active odd clover block 求导。dagger 分支的共轭双线性由
        ``computeCloverPairForceV2Quda`` 的左右场约定处理；这里显式保留
        dagger 对应的 shift 符号，避免把 ``mu_prec`` 或相反符号静态写死。
        """
        active_mu = self.invert_param.mu
        if active_mu == 0.0:
            return
        plus = dagger == QudaDagType.QUDA_DAG_NO
        shift_alpha = (-active_mu if plus else active_mu) * scale
        self._i_gamma5_parity(
            self.pc_tmp1,
            chi_odd,
            QudaParity.QUDA_ODD_PARITY,
            shift_alpha,
        )
        self._add_pair_force_v2(
            force,
            self._parity_field(self.pc_tmp1, QudaParity.QUDA_ODD_PARITY),
            psi_odd,
            sign,
            label,
            include_hopping=False,
        )

    def _add_pair_force_v2_quda_norm(
        self,
        force: LatticeGauge,
        x_field: LatticeFermion,
        p_field: LatticeFermion,
        alpha: float,
        label: str,
        include_clover: bool = True,
        include_hopping: bool = True,
        swap_pair_fields: bool = False,
    ) -> None:
        """按 QUDA MatPC 的数学系数 alpha 直接构造 pair force。

        computeCloverPairForceV2Quda 直接使用传入的 link/clover 导数系数；
        这里把 `alpha * D` 或 `alpha * A` 的数学系数转换成 kernel 需要的
        force_coeff/epsilon，避免复用旧 Chroma `-0.5` 归一化探针。
        """
        # pair V2 是低层非 dagger 外积接口；dagger 分支需要显式交换左右场。
        if swap_pair_fields:
            self.pair_x[0] = p_field
            self.pair_p[0] = x_field
        else:
            self.pair_x[0] = x_field
            self.pair_p[0] = p_field
        self.force_tmp.data = 0
        coeff = numpy.array([-alpha if include_hopping else 0.0], "<f8")
        epsilon_value = -self.ck * alpha if include_clover else 0.0
        epsilon_even = numpy.array([epsilon_value], "<f8")
        epsilon_odd = numpy.array([epsilon_value], "<f8")
        full_label = f"{self.label}: {label}"
        enabled = self.force_term_filter is None or self.force_term_filter in (label, full_label)
        if not enabled:
            if self.force_debug_logging:
                getLogger().info(f"{self.label}: skip quda-norm pair force {label} by force_term_filter")
            return
        computeCloverPairForceV2Quda(
            self.force_tmp.data_ptrs,
            self.pair_x.even_ptrs,
            self.pair_x.odd_ptrs,
            self.pair_p.even_ptrs,
            self.pair_p.odd_ptrs,
            coeff,
            epsilon_even,
            epsilon_odd,
            self.nvector,
            self.gauge_param,
            self.invert_param,
        )
        if self.force_debug_logging:
            getLogger().info(f"{self.label}: add quda-norm pair force {label} alpha={alpha:.17g}")
        self.force_term_history.append(
            {
                "label": full_label,
                "raw_label": label,
                "sign": float(alpha),
                "include_clover": bool(include_clover),
                "include_hopping": bool(include_hopping),
                "force_output_scale": float(self.force_output_scale),
                "swap_pair_fields": bool(swap_pair_fields),
                "coeff": float(coeff[0]),
                "epsilon_even": float(epsilon_even[0]),
                "epsilon_odd": float(epsilon_odd[0]),
            }
        )
        force.data += self.force_output_scale * self.force_tmp.data

    def _add_sym_oddodd_quda_norm_deriv_force(
        self,
        force: LatticeGauge,
        chi: LatticeFermion,
        psi: LatticeFermion,
        dagger: QudaDagType,
        sign: float,
        label: str,
        norm_scale: Optional[float] = None,
    ) -> None:
        """展开 QUDA symmetric odd-odd MatPC 的真实归一化导数。

        operator probe 已确认 QUDA MatPC 半格算子为
        `I - kappa^2 * base + (±mu/2) * i gamma5 A`。
        因此这里直接使用 `-kappa^2` 和 `±mu/2`，不再沿用 Chroma
        symmetric linop 的 `-0.5` hopping 归一化。
        两个系数还需乘 clover_csw/kappa = 1/v0^3，才能落在 Chroma
        direct clover 的 force 基上。
        """
        active = self._active_parity()
        if active != QudaParity.QUDA_ODD_PARITY:
            getLogger().critical("pc_quda_norm symmetric branch currently supports only odd_odd", NotImplementedError)

        plus = dagger == QudaDagType.QUDA_DAG_NO
        chi_odd = self._active_pc_field(chi)
        psi_odd = self._active_pc_field(psi)
        quda_norm_scale = self.quda_norm_force_scale if norm_scale is None else norm_scale
        base_alpha = sign * (-(self.invert_param.kappa**2)) * quda_norm_scale
        active_mu = self.invert_param.mu
        # ratio force 中 D_prec 分支会临时切到 mu_prec；shift 导数必须跟随 active mu。
        shift_alpha = (
            sign * (0.5 * active_mu if plus else -0.5 * active_mu)
            * quda_norm_scale
        )

        if plus:
            # base = A_oo^-1 D_oe A_ee^-1 D_eo psi。
            self._dslash_block(self.pc_tmp1, psi_odd, QudaParity.QUDA_EVEN_PARITY, QudaDagType.QUDA_DAG_NO)
            self._clover_inv_parity(self.pc_tmp2, self.pc_tmp1, QudaParity.QUDA_EVEN_PARITY, QudaDagType.QUDA_DAG_NO)
            self._dslash_block(self.pc_tmp3, self.pc_tmp2, QudaParity.QUDA_ODD_PARITY, QudaDagType.QUDA_DAG_NO)
            self._clover_inv_parity(self.pc_tmp5, self.pc_tmp3, QudaParity.QUDA_ODD_PARITY, QudaDagType.QUDA_DAG_NO)

            self._clover_inv_parity(self.pc_tmp4, chi_odd, QudaParity.QUDA_ODD_PARITY, QudaDagType.QUDA_DAG_YES)
            self._add_pair_force_v2_quda_norm(
                force,
                self._parity_field(self.pc_tmp4, QudaParity.QUDA_ODD_PARITY),
                self._parity_field(self.pc_tmp5, QudaParity.QUDA_ODD_PARITY),
                -base_alpha,
                f"{label}: quda PLUS A_oo^-1 insertion",
                include_hopping=False,
            )

            self._add_pair_force_v2_quda_norm(
                force,
                self._parity_field(self.pc_tmp4, QudaParity.QUDA_ODD_PARITY),
                self._parity_field(self.pc_tmp2, QudaParity.QUDA_EVEN_PARITY),
                base_alpha,
                f"{label}: quda PLUS D_oe prime",
                include_clover=False,
            )

            self._dslash_block(self.pc_tmp1, self.pc_tmp4, QudaParity.QUDA_EVEN_PARITY, QudaDagType.QUDA_DAG_YES)
            self._clover_inv_parity(self.pc_tmp3, self.pc_tmp1, QudaParity.QUDA_EVEN_PARITY, QudaDagType.QUDA_DAG_YES)
            self._add_pair_force_v2_quda_norm(
                force,
                self._parity_field(self.pc_tmp3, QudaParity.QUDA_EVEN_PARITY),
                self._parity_field(self.pc_tmp2, QudaParity.QUDA_EVEN_PARITY),
                -base_alpha,
                f"{label}: quda PLUS A_ee^-1 insertion",
                include_hopping=False,
            )

            self._add_pair_force_v2_quda_norm(
                force,
                self._parity_field(self.pc_tmp3, QudaParity.QUDA_EVEN_PARITY),
                psi_odd,
                base_alpha,
                f"{label}: quda PLUS D_eo prime",
                include_clover=False,
            )
        else:
            # base_dag = D_oe^dag A_ee^-dag D_eo^dag A_oo^-dag psi。
            self._clover_inv_parity(self.pc_tmp1, psi_odd, QudaParity.QUDA_ODD_PARITY, QudaDagType.QUDA_DAG_YES)
            self._dslash_block(self.pc_tmp2, self.pc_tmp1, QudaParity.QUDA_EVEN_PARITY, QudaDagType.QUDA_DAG_YES)
            self._clover_inv_parity(self.pc_tmp3, self.pc_tmp2, QudaParity.QUDA_EVEN_PARITY, QudaDagType.QUDA_DAG_YES)

            self._add_pair_force_v2_quda_norm(
                force,
                chi_odd,
                self._parity_field(self.pc_tmp3, QudaParity.QUDA_EVEN_PARITY),
                base_alpha,
                f"{label}: quda MINUS D_oe prime",
                include_clover=False,
                swap_pair_fields=True,
            )

            self._dslash_block(self.pc_tmp4, chi_odd, QudaParity.QUDA_EVEN_PARITY, QudaDagType.QUDA_DAG_NO)
            self._clover_inv_parity(self.pc_tmp5, self.pc_tmp4, QudaParity.QUDA_EVEN_PARITY, QudaDagType.QUDA_DAG_NO)
            self._add_pair_force_v2_quda_norm(
                force,
                self._parity_field(self.pc_tmp5, QudaParity.QUDA_EVEN_PARITY),
                self._parity_field(self.pc_tmp3, QudaParity.QUDA_EVEN_PARITY),
                -base_alpha,
                f"{label}: quda MINUS A_ee^-1 insertion",
                include_hopping=False,
                swap_pair_fields=True,
            )

            self._add_pair_force_v2_quda_norm(
                force,
                self._parity_field(self.pc_tmp5, QudaParity.QUDA_EVEN_PARITY),
                self._parity_field(self.pc_tmp1, QudaParity.QUDA_ODD_PARITY),
                base_alpha,
                f"{label}: quda MINUS D_eo prime",
                include_clover=False,
                swap_pair_fields=True,
            )

            self._dslash_block(self.pc_tmp4, self.pc_tmp5, QudaParity.QUDA_ODD_PARITY, QudaDagType.QUDA_DAG_NO)
            self._clover_inv_parity(self.pc_tmp5, self.pc_tmp4, QudaParity.QUDA_ODD_PARITY, QudaDagType.QUDA_DAG_NO)
            self._add_pair_force_v2_quda_norm(
                force,
                self._parity_field(self.pc_tmp5, QudaParity.QUDA_ODD_PARITY),
                self._parity_field(self.pc_tmp1, QudaParity.QUDA_ODD_PARITY),
                -base_alpha,
                f"{label}: quda MINUS A_oo^-1 insertion",
                include_hopping=False,
                swap_pair_fields=True,
            )

        if active_mu != 0.0:
            # shift = (plus/minus mu/2) i gamma5 A_oo；dagger 分支交换双线性左右场。
            shift_alpha *= -1.0
            self._i_gamma5_parity(self.pc_tmp1, chi_odd, QudaParity.QUDA_ODD_PARITY, shift_alpha)
            self._add_pair_force_v2_quda_norm(
                force,
                self._parity_field(self.pc_tmp1, QudaParity.QUDA_ODD_PARITY),
                psi_odd,
                1.0,
                f"{label}: quda shifted i_mu_over_2_gamma5_A_oo prime",
                include_hopping=False,
                swap_pair_fields=not plus,
            )

    def _add_pair_force_v2(
        self,
        force: LatticeGauge,
        x_field: LatticeFermion,
        p_field: LatticeFermion,
        sign: float,
        label: str,
        include_clover: bool = True,
        include_hopping: bool = True,
        symmetric_clover: bool = False,
        swap_pair_fields: bool = False,
    ) -> None:
        # 该 helper 只做一个 Chroma deriv(left,right,±) 的 bilinear force 项；
        # 四项 ratio 组合和整体符号在调用处显式列出，便于 finite-difference 对照。
        if swap_pair_fields:
            self.pair_x[0] = p_field
            self.pair_p[0] = x_field
        else:
            self.pair_x[0] = x_field
            self.pair_p[0] = p_field
        self.force_tmp.data = 0
        # pc_pair 的数学系数已经由 sign 显式给出；这里直接映射到 pair-force kernel，
        # 不再重复乘 4*kappa^2 或使用旧 epsilon_even/epsilon_odd 归一化。
        # 有限差分显示 pc_pair 的 Chroma -0.5 展开还需乘 QUDA kappa 归一化。
        # 保留原公共诊断倍率的作用，但不重复原 coeff 中的 4*kappa^2。
        pair_scale = sign * self.force_sign * self.force_coeff_scale
        coeff = numpy.array([-pair_scale * self.invert_param.kappa if include_hopping else 0.0], "<f8")
        epsilon_value = -self.ck * pair_scale * self.invert_param.kappa if include_clover else 0.0
        epsilon_even = numpy.array([epsilon_value], "<f8")
        epsilon_odd = numpy.array([epsilon_value], "<f8")
        full_label = f"{self.label}: {label}"
        enabled = self.force_term_filter is None or self.force_term_filter in (label, full_label)
        if not enabled:
            if self.force_debug_logging:
                getLogger().info(f"{self.label}: skip pair force {label} by force_term_filter")
            return
        computeCloverPairForceV2Quda(
            self.force_tmp.data_ptrs,
            self.pair_x.even_ptrs,
            self.pair_x.odd_ptrs,
            self.pair_p.even_ptrs,
            self.pair_p.odd_ptrs,
            coeff,
            epsilon_even,
            epsilon_odd,
            self.nvector,
            self.gauge_param,
            self.invert_param,
        )
        if self.force_debug_logging:
            getLogger().info(f"{self.label}: add pair force {label} sign={sign}")
        self.force_term_history.append(
            {
                "label": full_label,
                "raw_label": label,
                "sign": float(sign),
                "include_clover": bool(include_clover),
                "include_hopping": bool(include_hopping),
                "force_output_scale": float(self.force_output_scale),
                "coeff": float(self.coeff[0]),
                "epsilon_even": float(epsilon_even[0]),
                "epsilon_odd": float(epsilon_odd[0]),
            }
        )
        force.data += self.force_output_scale * self.force_tmp.data

    def force_v2(self, dt, force: Optional[LatticeGauge] = None):
        self.updateClover()
        if self.determinant_ratio:
            # Chroma ratio:
            # X=(M^dag M)^-1 M_prec^dag phi, Y=M X,
            # dS = +D_prec(phi,X) -D(X,Y) -D(Y,X) +D_prec(X,phi)。
            self._mat(self.rhs, self.phi, QudaDagType.QUDA_DAG_YES, self.mu_prec)
            self._solve(self.rhs, self.mu, compute_action=False)
            self._mat(self.y, self.quark[0], QudaDagType.QUDA_DAG_NO, self.mu)
        else:
            self._solve(self.phi, self.mu, compute_action=False)
            if self.constdet:
                # const-det cancel 只需要伪费米子力；传入零 x0 让 QUDA 跳过 sigma-trace 且不添加 detratio source。
                self.source.data = 0
            else:
                self.source[0] = self.phi
        old_dagger = self.invert_param.dagger
        self.invert_param.dagger = self.force_dagger
        old_mu = self.invert_param.mu
        self.invert_param.mu = self.mu
        force_out = force
        if (
            force is not None
            and not self.determinant_ratio
            and self.cancel_force_mode not in ("pc_pair", "pc_constdet", "pc_quda_twist", "pc_quda_norm")
            and self.force_output_scale != 1.0
        ):
            # QUDA TMC determinant force 中 clover sigma-trace 项不随 coeff 缩放；
            # 需要整体缩放时先写入临时 force，再把整个 monomial 贡献加回总 force。
            self.force_tmp.data = 0
            force_out = self.force_tmp
        if force_out is not None:
            self.gauge_param.use_resident_force = 0
            self.gauge_param.return_result_force = 1
            if self.force_debug_logging:
                getLogger().info(
                    f"{self.label}: force_v2 host output flags "
                    f"use_resident_force={self.gauge_param.use_resident_force}, "
                    f"return_result_force={self.gauge_param.return_result_force}, "
                    f"make_resident_force={self.gauge_param.make_resident_force}"
                )
        if self.determinant_ratio and force_out is not None and self.ratio_force_mode == "pair":
            # pair 模式保留为调试入口；默认不使用，因为它没有 PC block 重构。
            self._add_pair_force_v2(force_out, self.phi, self.quark[0], +1.0, "D_prec(phi,X)")
            self._add_pair_force_v2(force_out, self.quark[0], self.y, -1.0, "D(X,Y)")
            self._add_pair_force_v2(force_out, self.y, self.quark[0], -1.0, "D(Y,X)")
            self._add_pair_force_v2(force_out, self.quark[0], self.phi, +1.0, "D_prec(X,phi)")
        elif self.determinant_ratio and force_out is not None and self.ratio_force_mode in ("pc_pair", "pc_quda_twist"):
            # pc_pair 显式对应 Chroma ratio force：D_prec(phi,X)-D(X,Y)-D(Y,X)+D_prec(X,phi)。
            include_quda_twist = self.ratio_force_mode == "pc_quda_twist"
            old_mu_for_pc = self.invert_param.mu
            self.invert_param.mu = self.mu_prec
            if self._is_symmetric_matpc():
                self._add_sym_oddodd_chroma_deriv_force(force_out, self.phi, self.quark[0], QudaDagType.QUDA_DAG_NO, -1.0, "D_prec(phi,X)")
                self._add_sym_oddodd_chroma_deriv_force(force_out, self.quark[0], self.phi, QudaDagType.QUDA_DAG_YES, -1.0, "D_prec(X,phi)")
            else:
                self._add_eoprec_deriv_force(force_out, self.phi, self.quark[0], QudaDagType.QUDA_DAG_NO, -1.0, "D_prec(phi,X)", include_quda_twist)
                self._add_eoprec_deriv_force(force_out, self.quark[0], self.phi, QudaDagType.QUDA_DAG_YES, -1.0, "D_prec(X,phi)", include_quda_twist)
            self.invert_param.mu = self.mu
            if self._is_symmetric_matpc():
                self._add_sym_oddodd_chroma_deriv_force(force_out, self.quark[0], self.y, QudaDagType.QUDA_DAG_YES, +1.0, "-D(X,Y)")
                self._add_sym_oddodd_chroma_deriv_force(force_out, self.y, self.quark[0], QudaDagType.QUDA_DAG_NO, +1.0, "-D(Y,X)")
            else:
                self._add_eoprec_deriv_force(force_out, self.quark[0], self.y, QudaDagType.QUDA_DAG_YES, +1.0, "-D(X,Y)", include_quda_twist)
                self._add_eoprec_deriv_force(force_out, self.y, self.quark[0], QudaDagType.QUDA_DAG_NO, +1.0, "-D(Y,X)", include_quda_twist)
            self.invert_param.mu = old_mu_for_pc
        elif self.determinant_ratio and force_out is not None and self.ratio_force_mode == "pc_quda_norm":
            if not self._is_symmetric_matpc():
                getLogger().critical("pc_quda_norm ratio requires symmetric odd_odd/even_even MatPC", NotImplementedError)
            old_mu_for_pc = self.invert_param.mu
            # determinant ratio 的作用量导数是四项大数相消；
            # pc_quda_norm 必须完整保留 dagger 与 non-dagger 两边，不能只用 PLUS 半边近似。
            # 这里底层 helper 返回 dS/dU；HMC.loadGaugeMom 会把 force_v2 直接加到动量，
            # 因此 determinant-ratio monomial 的动量力需要整体取 -dS/dU。
            label_prefix = "D_prec symmetric"
            d_label_prefix = "D symmetric"
            self.invert_param.mu = self.mu_prec
            self._add_sym_oddodd_quda_norm_deriv_force(
                force_out,
                self.phi,
                self.quark[0],
                QudaDagType.QUDA_DAG_NO,
                -1.0,
                f"{label_prefix} +(phi,X)",
                norm_scale=1.0,
            )
            self._add_sym_oddodd_quda_norm_deriv_force(
                force_out,
                self.quark[0],
                self.phi,
                QudaDagType.QUDA_DAG_YES,
                -1.0,
                f"{label_prefix} +(X,phi)dag",
                norm_scale=1.0,
            )
            self.invert_param.mu = self.mu
            self._add_sym_oddodd_quda_norm_deriv_force(
                force_out,
                self.quark[0],
                self.y,
                QudaDagType.QUDA_DAG_YES,
                +1.0,
                f"{d_label_prefix} -(X,Y)dag",
                norm_scale=1.0,
            )
            self._add_sym_oddodd_quda_norm_deriv_force(
                force_out,
                self.y,
                self.quark[0],
                QudaDagType.QUDA_DAG_NO,
                +1.0,
                f"{d_label_prefix} -(Y,X)",
                norm_scale=1.0,
            )
            self.invert_param.mu = old_mu_for_pc
        elif (not self.determinant_ratio) and force_out is not None and self.cancel_force_mode in ("pc_pair", "pc_quda_twist"):
            # Chroma cancel: X=(M^dag M)^-1 phi, Y=M X，源码公式为
            # -[deriv(X,Y,MINUS)+deriv(Y,X,PLUS)]。QUDA pair kernel 的单个
            # PLUS 调用已经包含前向/后向 link 的共轭组合，因此只调用一次并取 -1。
            include_quda_twist = self.cancel_force_mode == "pc_quda_twist"
            self._mat(self.y, self.quark[0], QudaDagType.QUDA_DAG_NO, self.mu)
            if self._is_symmetric_matpc():
                self._add_sym_oddodd_chroma_deriv_force(
                    force_out,
                    self.y,
                    self.quark[0],
                    QudaDagType.QUDA_DAG_NO,
                    -1.0,
                    "cancel pc_pair -M(Y,X)",
                )
            else:
                self._add_eoprec_deriv_force(
                    force_out,
                    self.y,
                    self.quark[0],
                    QudaDagType.QUDA_DAG_NO,
                    -1.0,
                    "cancel pc_pair -M(Y,X)",
                    include_quda_twist,
                )
        elif (not self.determinant_ratio) and force_out is not None and self.cancel_force_mode == "pc_quda_norm":
            # QUDA-normalized cancel: X=(M^dag M)^-1 phi, Y=M X。
            # Chroma 的显式力写作 -[M^dag'(X,Y)+M'(Y,X)]；但 QUDA pair-force kernel
            # 对一个 bilinear 已经生成反厄米 momentum force 所需的共轭组合。有限差分显示
            # 再加 Mdag 分支会重复/错加，正确闭合对应只保留 PLUS 组并反号。
            if not self._is_symmetric_matpc():
                getLogger().critical("pc_quda_norm requires symmetric odd_odd/even_even MatPC", NotImplementedError)
            self._mat(self.y, self.quark[0], QudaDagType.QUDA_DAG_NO, self.mu)
            # Chroma cancel: -[deriv(X,Y,MINUS)+deriv(Y,X,PLUS)]。
            # 这里按同一 QUDA MatPC 导数基展开为两项，不再使用 +2 单项近似。
            self._add_sym_oddodd_quda_norm_deriv_force(
                force_out,
                self.quark[0],
                self.y,
                QudaDagType.QUDA_DAG_YES,
                +1.0,
                "cancel quda-kernel -Mdag(X,Y)",
                norm_scale=1.0,
            )
            self._add_sym_oddodd_quda_norm_deriv_force(
                force_out,
                self.y,
                self.quark[0],
                QudaDagType.QUDA_DAG_NO,
                +1.0,
                "cancel quda-kernel -M(Y,X)",
                norm_scale=1.0,
            )
        elif (not self.determinant_ratio) and force_out is not None and self.cancel_force_mode == "pc_constdet":
            # Chroma const-det cancel: diffM.deriv(F,X,Y,MINUS)+diffM.deriv(F_tmp,Y,X,PLUS)，再整体乘 -1。
            # 该模式只用于验证 const-det cancel 力是否闭合，不包含 clover trlog/sigma-trace 项。
            self._mat(self.y, self.quark[0], QudaDagType.QUDA_DAG_NO, self.mu)
            self._add_eoprec_constdet_deriv_force(force_out, self.quark[0], self.y, QudaDagType.QUDA_DAG_YES, -1.0, "cancel constdet -Ddag(X,Y)")
            self._add_eoprec_constdet_deriv_force(force_out, self.y, self.quark[0], QudaDagType.QUDA_DAG_NO, -1.0, "cancel constdet -D(Y,X)")
        else:
            if self._is_symmetric_matpc():
                getLogger().critical(
                    "symmetric MatPC force requires pc_pair, pc_quda_twist or pc_quda_norm explicit force mode",
                    NotImplementedError,
                )
            if self.determinant_ratio:
                # QUDA detratio force 用 x0 交叉项表示 ratio 的 D_prec(phi,X)+D_prec(X,phi)。
                self.source[0] = self.phi if self.ratio_force_source == "phi" else self.rhs
            computeTMCloverForceV2Quda(
                nullptr if force_out is None else force_out.data_ptrs,
                self.quark.even_ptrs,
                self.source.even_ptrs,
                self.coeff,
                self.nvector,
                self.gauge_param,
                self.invert_param,
                int(self.determinant_ratio or self.constdet),
            )
        if force is not None and force_out is self.force_tmp:
            force.data += self.force_output_scale * self.force_tmp.data
        if force_out is not None:
            self.gauge_param.use_resident_force = 1
            self.gauge_param.return_result_force = 0
        self.invert_param.mu = old_mu
        self.invert_param.dagger = old_dagger

    def force(self, dt, mom: Optional[LatticeMom] = None):
        getLogger().critical(
            "CloverWilsonHasenbuschTwistAction.force is not available; use force_v2 through stout HMC",
            NotImplementedError,
        )
