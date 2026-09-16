from typing import Optional
import numpy

from pyquda_comm import getLogger
from ..field import LatticeInfo, LatticeGauge, LatticeMom, LatticeFermion, MultiLatticeFermion
from ..quda import MatQuda, computeCloverForceV2Quda, computeCloverForceQuda, loadCloverQuda, loadGaugeQuda
from ..enum_quda import (
    QudaDagType,
    QudaInverterType,
    QudaMassNormalization,
    QudaMatPCType,
    QudaParity,
    QudaSolutionType,
    QudaSolveType,
)
from ..dirac import CloverWilsonDirac

nullptr = numpy.empty((0, 0), "<c16")
nullptr_clover = numpy.empty((0), "<f8")

from .abstract import RationalParam, FermionAction


class CloverWilsonAction(FermionAction):
    dirac: CloverWilsonDirac

    def __init__(
        self,
        latt_info: LatticeInfo,
        rational_param: RationalParam,
        mass: float,
        n_flavor: int,
        tol: float,
        maxiter: int,
        clover_csw: float,
        use_action_rational: bool = False,
        matpc_type: str = "even_even_asymmetric",
    ) -> None:
        if latt_info.anisotropy != 1.0:
            getLogger().critical("anisotropy != 1.0 not implemented", NotImplementedError)
        super().__init__(latt_info, CloverWilsonDirac(latt_info, mass, tol, maxiter, clover_csw, 1, None))

        kappa = 1 / (2 * (mass + latt_info.Nd))
        self.setForceParam(rational_param, kappa, clover_csw, n_flavor)
        # Metropolis action 可选使用 action rational；force 仍使用 force rational。
        self.use_action_rational = bool(use_action_rational)
        matpc_map = {
            "even_even_asymmetric": QudaMatPCType.QUDA_MATPC_EVEN_EVEN_ASYMMETRIC,
            "odd_odd_asymmetric": QudaMatPCType.QUDA_MATPC_ODD_ODD_ASYMMETRIC,
            "even_even": QudaMatPCType.QUDA_MATPC_EVEN_EVEN,
            "odd_odd": QudaMatPCType.QUDA_MATPC_ODD_ODD,
        }
        if matpc_type not in matpc_map:
            getLogger().critical("matpc_type must be even_even_asymmetric, odd_odd_asymmetric, even_even or odd_odd", ValueError)
        self.matpc_type = matpc_map[matpc_type]
        self.active_parity = (
            QudaParity.QUDA_EVEN_PARITY if matpc_type.startswith("even_even") else QudaParity.QUDA_ODD_PARITY
        )
        self.quark = MultiLatticeFermion(self.latt_info, self.max_num_offset)
        self.phi = LatticeFermion(latt_info)
        self.eta = LatticeFermion(latt_info)
        self.label = "strange"
        self._sym_force_helper = None
        if matpc_type == "odd_odd":
            self.y = MultiLatticeFermion(latt_info, self.nvector)
            from .clover_hasenbusch import CloverWilsonHasenbuschTwistAction

            self._sym_force_helper = CloverWilsonHasenbuschTwistAction(
                latt_info,
                mass,
                0.0,
                tol,
                maxiter,
                clover_csw,
                determinant_ratio=False,
                constdet=True,
                matpc_type="odd_odd",
                label="strange_symmetric_force_helper",
            )

        self.dirac.setPrecision(n_flavor != 2)  # TODO: Check if it's better or worse
        self.invert_param.inv_type = QudaInverterType.QUDA_CG_INVERTER
        self.invert_param.solution_type = QudaSolutionType.QUDA_MATPCDAG_MATPC_SOLUTION
        self.invert_param.solve_type = QudaSolveType.QUDA_NORMOP_PC_SOLVE  # This is set to compute action
        self.invert_param.matpc_type = self.matpc_type
        self.invert_param.mass_normalization = QudaMassNormalization.QUDA_KAPPA_NORMALIZATION

    def setForceParam(self, rational_param: RationalParam, kappa: float, clover_csw: float, n_flavor: int):
        self.kappa = kappa
        self.coeff = numpy.array(rational_param.residue_force, "<f8")
        self.kappa2 = -(kappa**2)
        self.ck = -kappa * clover_csw / 8
        self.nvector = len(rational_param.offset_force)
        self.multiplicity = n_flavor
        self.max_num_offset = max(
            len(rational_param.offset_force),
            len(rational_param.offset_sample),
            len(rational_param.offset_action),
        )
        self.rational_param = rational_param

    def updateClover(self):
        loadGaugeQuda(nullptr, self.gauge_param)
        loadCloverQuda(nullptr_clover, nullptr_clover, self.invert_param)

    def sample(self):
        self.sampleEta()
        self.updateClover()
        self.invertMultiShift("sample")

    def sampleEta(self):
        super().sampleEta()
        # QUDA MatPC 只使用 data[0] 槽位；odd-odd 时该槽位对应物理 odd half。
        if self.active_parity == QudaParity.QUDA_ODD_PARITY:
            self.eta.data[0] = self.eta.data[1]
        self.eta.data[1] = 0

    def action(self) -> float:
        self.invert_param.compute_clover_trlog = 1
        self.updateClover()
        self.invert_param.compute_clover_trlog = 0
        if self.use_action_rational:
            # invertMultiShift("action") 以 eta 为源、phi 为输出；
            # 为计算 S[phi]，临时把 eta 替换为 phi，并保留原始场。
            eta_backup = self.eta.data.copy()
            phi_backup = self.phi.data.copy()
            self.eta.data = self.phi.data.copy()
            self.invertMultiShift("action")
            # `invertMultiShift("action")` 会把 norm*phi + sum_i residue_i*X_i
            # 重构到 self.eta；这里必须使用 eta，否则 action 只依赖固定 phi。
            action = (
                self.eta.norm2()
                - self.latt_info.global_volume // 2 * self.latt_info.Ns * self.latt_info.Nc
            )
            self.eta.data = eta_backup
            self.phi.data = phi_backup
            return action
        self.invert_param.compute_action = 1
        self.invertMultiShift("force")
        self.invert_param.compute_action = 0
        return (
            self.invert_param.action[0]
            - self.latt_info.global_volume // 2 * self.latt_info.Ns * self.latt_info.Nc  # volume_cb2 here
            - self.multiplicity * self.invert_param.trlogA[
                1 if self.active_parity == QudaParity.QUDA_EVEN_PARITY else 0
            ]
        )

    def force_v2(self, dt, force: Optional[LatticeGauge] = None):
        self.updateClover()
        self.invertMultiShift("force")
        if self._sym_force_helper is not None and force is not None:
            helper = self._sym_force_helper
            helper.invert_param = self.invert_param
            helper.gauge_param = self.gauge_param
            helper.ck = self.ck
            # HMC momentum update 需要 -dS/dU；symmetric Chroma helper 的
            # QUDA kernel 归一化再乘 -4*kappa 才能得到正确符号和幅度。
            helper.force_output_scale = -4.0 * self.kappa
            helper.force_term_filter = None
            self.gauge_param.use_resident_force = 0
            self.gauge_param.return_result_force = 1

            old_dagger = self.invert_param.dagger
            self.invert_param.dagger = QudaDagType.QUDA_DAG_NO
            for i in range(self.nvector):
                MatQuda(self.y[i].even_ptr, self.quark[i].even_ptr, self.invert_param)
                coeff_i = float(self.coeff[i])
                if i == 0:
                    getLogger().info(
                        f"strange symmetric force debug: quark_norm2={self.quark[i].norm2():.17g}, "
                        f"y_norm2={self.y[i].norm2():.17g}, coeff={coeff_i:.17g}"
                    )
                helper._add_sym_oddodd_chroma_deriv_force(
                    force,
                    self.quark[i],
                    self.y[i],
                    QudaDagType.QUDA_DAG_YES,
                    -coeff_i,
                    f"strange pole {i} X^dag dM^dag Y",
                )
                helper._add_sym_oddodd_chroma_deriv_force(
                    force,
                    self.y[i],
                    self.quark[i],
                    QudaDagType.QUDA_DAG_NO,
                    -coeff_i,
                    f"strange pole {i} Y^dag dM X",
                )
            self.invert_param.dagger = old_dagger
            self.gauge_param.use_resident_force = 1
            self.gauge_param.return_result_force = 0
            return
        # Some conventions force the dagger to be YES here
        self.invert_param.dagger = QudaDagType.QUDA_DAG_YES
        if force is not None:
            self.gauge_param.use_resident_force = 0
            self.gauge_param.return_result_force = 1
        computeCloverForceV2Quda(
            nullptr if force is None else force.data_ptrs,
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
        if force is not None:
            self.gauge_param.use_resident_force = 1
            self.gauge_param.return_result_force = 0
        self.invert_param.dagger = QudaDagType.QUDA_DAG_NO

    def force(self, dt, mom: Optional[LatticeMom] = None):
        self.updateClover()
        self.invertMultiShift("force")
        # Some conventions force the dagger to be YES here
        self.invert_param.dagger = QudaDagType.QUDA_DAG_YES
        if mom is not None:
            self.gauge_param.use_resident_mom = 0
            self.gauge_param.make_resident_gauge = 0
            self.gauge_param.return_result_mom = 1
        computeCloverForceQuda(
            nullptr if mom is None else mom.data_ptrs,
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
        if mom is not None:
            self.gauge_param.use_resident_mom = 1
            self.gauge_param.make_resident_gauge = 1
            self.gauge_param.return_result_mom = 0
        self.invert_param.dagger = QudaDagType.QUDA_DAG_NO
