#!/usr/bin/env python3
"""12x32 clover logdet force 有限差分诊断。

用途：验证 Chroma `N_FLAVOR_LOGDET_DIAG_FERM_MONOMIAL` 对应的
`S=-Nf*(logdet A_even + logdet A_odd)` 是否能用 QUDA 现有
`computeCloverForceV2Quda` 的 sigma-trace 路径构造出闭合 force。

运行前置：在 p100 上先 source
`/home/jyg/stout_smear/PyQUDA_stout_smeared_force/set_PyQUDA_env.sh`。
输出：只写 `hasenbusch_operation/csv` 与 `hasenbusch_operation/logs`。
"""

import argparse
import csv
import logging
import math
import sys
from pathlib import Path

import numpy


OP_DIR = Path(__file__).resolve().parents[1]
SCRIPT_DIR = OP_DIR / "scripts"
LOG_DIR = OP_DIR / "logs"
CSV_DIR = OP_DIR / "csv"
CACHE_ROOT = OP_DIR / ".cache" / "force_check"
TESTS_DIR = OP_DIR.parent
PROJECT_DIR = TESTS_DIR.parent

sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(TESTS_DIR))

from pyquda.action import GaugeAction
from pyquda.action.abstract import FermionAction
from pyquda.dirac import CloverWilsonDirac
from pyquda.enum_quda import (
    QudaDagType,
    QudaInverterType,
    QudaMassNormalization,
    QudaMatPCType,
    QudaSolutionType,
    QudaSolveType,
    QudaVerbosity,
)
from pyquda.field import LatticeGauge, LatticeMom, MultiLatticeFermion
from pyquda.hmc import HMC, O2Nf1Ng0V
from pyquda.quda import computeCloverForceV2Quda, loadCloverQuda, loadGaugeQuda
from pyquda_utils import core, io
from pyquda_utils.hmc_param import symanzikTreeGaugeLoopParam as loopParam


nullptr = numpy.empty((0, 0), "<c16")
nullptr_clover = numpy.empty((0), "<c16")
FROM_UPSTREAM_TESTS = Path(
    "/home/jyg/stout_smear/PyQUDA_stout_smeared_force_fromUpstream/PyQUDA/tests"
)


class CloverLogDetAction(FermionAction):
    """Chroma symmetric EO clover 对角 logdet monomial 的诊断实现。"""

    def __init__(self, latt_info, mass, n_flavor, tol, maxiter, clover_csw):
        super().__init__(latt_info, CloverWilsonDirac(latt_info, mass, tol, maxiter, clover_csw, 1, None))
        kappa = 1 / (2 * (mass + latt_info.Nd))
        self.kappa2 = -(kappa**2)
        self.ck = -kappa * clover_csw / 8
        self.coeff = numpy.array([0.0], "<f8")
        self.nvector = 1
        self.multiplicity = n_flavor
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
        # logdet monomial 没有 pseudofermion；清零只是保证 force kernel 的 dummy x 不贡献双线性项。
        self.quark.data = 0

    def action(self) -> float:
        self.invert_param.compute_clover_trlog = 1
        self.updateClover()
        self.invert_param.compute_clover_trlog = 0
        return -self.multiplicity * (self.invert_param.trlogA[0] + self.invert_param.trlogA[1])

    def _add_sigma_trace_force_for_matpc(self, force, matpc_type):
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

    def force_v2(self, dt, force=None):
        self.updateClover()
        self.quark.data = 0
        if force is not None:
            self.gauge_param.use_resident_force = 0
            self.gauge_param.return_result_force = 1
            # 分别取 odd/even parity 的 sigma-trace；QUDA force 内部对 active 的另一半做 trace。
            self._add_sigma_trace_force_for_matpc(force, QudaMatPCType.QUDA_MATPC_EVEN_EVEN_ASYMMETRIC)
            self._add_sigma_trace_force_for_matpc(force, QudaMatPCType.QUDA_MATPC_ODD_ODD_ASYMMETRIC)
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

    def force(self, dt, mom=None):
        # 该诊断只走 stout HMC 的 force_v2 路径；普通 momentum force 入口不用于验证。
        raise NotImplementedError("CloverLogDetAction.force is not used; call force_v2 through stout HMC")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", choices=["1000", "10001"], default="1000")
    parser.add_argument("--seed", type=int, default=10086)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--eps", default="1e-5,3e-5,1e-4")
    return parser.parse_args()


def copy_mom_like(latt_info, data) -> LatticeMom:
    mom = LatticeMom(latt_info)
    mom.data = data.copy()
    return mom


def kinetic_inner_product(hmc, latt_info, a: LatticeMom, b: LatticeMom) -> float:
    plus = copy_mom_like(latt_info, a.data + b.data)
    minus = copy_mom_like(latt_info, a.data - b.data)
    hmc.loadMom(plus)
    k_plus = hmc.momAction()
    hmc.loadMom(minus)
    k_minus = hmc.momAction()
    return (k_plus - k_minus) / 2.0


def action_at_shift(hmc, gauge0: LatticeGauge, direction: LatticeMom, eps: float) -> float:
    hmc.loadGauge(gauge0)
    hmc.loadMom(direction)
    hmc.updateGauge(eps)
    return hmc.fermionAction()


def main() -> int:
    args = parse_args()
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(args.run_id)
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(LOG_DIR / f"force_check_{args.run_id}.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)

    latt_size = [12, 12, 12, 32]
    u_0, v_0 = 0.855453, 0.951479
    ml_bare = -0.2770
    clover_csw = 1 / v_0**3
    eps_values = [float(item) for item in args.eps.split(",") if item.strip()]
    gauge_path = FROM_UPSTREAM_TESTS / f"beta6.20_mu-0.2770_ms-0.2400_L12x32_cfg_{args.cfg}.lime"

    core.init(None, latt_size, resource_path=str(CACHE_ROOT / args.run_id), enable_force_monitor=True)
    latt_info = core.LatticeInfo(latt_size, t_boundary=-1, anisotropy=1.0)
    logdet = CloverLogDetAction(latt_info, ml_bare, 2, 1e-10, 5000, clover_csw)
    hmc = HMC(latt_info, [GaugeAction(latt_info, loopParam(u_0), 6.2), logdet], O2Nf1Ng0V(1))
    hmc.setFermionVerbosity(QudaVerbosity.QUDA_SILENT)
    gauge = io.readQIOGauge(str(gauge_path))
    hmc.initialize(args.seed, gauge)
    hmc.samplePhi()

    gauge0 = LatticeGauge(latt_info)
    hmc.saveGauge(gauge0)
    hmc.gaussMom()
    direction = LatticeMom(latt_info)
    hmc.saveMom(direction)

    zero_mom = LatticeMom(latt_info)
    force_mom = LatticeMom(latt_info)
    hmc.loadGauge(gauge0)
    hmc.loadMom(zero_mom)
    hmc.fermionForce(1.0)
    hmc.saveMom(force_mom)
    force_dot = kinetic_inner_product(hmc, latt_info, force_mom, direction)
    force_norm = math.sqrt(float((force_mom.data * force_mom.data).sum()))
    logger.info(f"force_dot_direction = {force_dot:.17g}")
    logger.info(f"force_norm = {force_norm:.17g}")

    csv_file = CSV_DIR / f"force_check_{args.run_id}.csv"
    with csv_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run_id",
                "cfg",
                "eps",
                "S_plus",
                "S_minus",
                "finite_difference",
                "force_dot_direction",
                "fd_plus_force_dot",
                "relative_error_opposite_sign",
                "force_norm",
            ],
        )
        writer.writeheader()
        for eps in eps_values:
            s_plus = action_at_shift(hmc, gauge0, direction, eps)
            s_minus = action_at_shift(hmc, gauge0, direction, -eps)
            fd = (s_plus - s_minus) / (2 * eps)
            denom = max(abs(fd), abs(force_dot), 1.0)
            writer.writerow(
                {
                    "run_id": args.run_id,
                    "cfg": args.cfg,
                    "eps": eps,
                    "S_plus": s_plus,
                    "S_minus": s_minus,
                    "finite_difference": fd,
                    "force_dot_direction": force_dot,
                    "fd_plus_force_dot": fd + force_dot,
                    "relative_error_opposite_sign": abs(fd + force_dot) / denom,
                    "force_norm": force_norm,
                }
            )
            logger.info(f"eps={eps:.3e} fd={fd:.17g} fd+force={fd + force_dot:.17g}")
    print(csv_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
