# ===============log config===========================================
import logging
from pathlib import Path
from datetime import datetime
import os

# 创建缓存目录
os.makedirs(".cache/stout", exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = Path("./log_test_hmc_clover")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"test_hmc_clover_24x64_{timestamp}.log"  

logging.basicConfig(
    level=logging.INFO,
    format="%(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_file, encoding='utf-8')]
)
# ======================================================================


from math import exp
from time import perf_counter

from pyquda_utils import core, io
from pyquda.hmc import HMC, O4Nf5Ng0V
from pyquda.action import GaugeAction, CloverWilsonAction
from pyquda.enum_quda import QudaVerbosity
from pyquda_utils.hmc_param import (
    symanzikTreeGaugeLoopParam as loopParam,
    fermionRationalParam as rationalParam,
)

from check_pyquda import test_dir
from check_pyquda import field_24x64, data

# beta, u_0 = 7.4, 0.890
beta, u_0, v_0 = 6.2, 0.855453, 0.951479
ml_bare, ms_bare = -0.277, -0.2310  
# clover_csw = 1 / u_0**3
clover_csw = 1 / v_0**3
tol, maxiter = 1e-6, 1000
start, stop, warm, save =  0, 10, 5, 5
# t = 1.0
t = 0.1  #积分步长

latt_size = [24, 24, 24, 64]
# grid_size = [2, 2, 1, 1]  # 4-GPU 划分
# core.init(grid_size, latt_size, resource_path=".cache/stout", enable_force_monitor=True)
core.init(None, latt_size, resource_path=".cache/stout", enable_force_monitor=True)
latt_info = core.LatticeInfo(latt_size, t_boundary=1, anisotropy=1.0)

monomials = [
    GaugeAction(latt_info, loopParam(u_0), beta),
    CloverWilsonAction(latt_info, rationalParam(2, 12, 15, 7e-4, 32, 50), ml_bare, 2, tol, maxiter, clover_csw),
    CloverWilsonAction(latt_info, rationalParam(1, 12, 15, 7e-4, 32, 50), ms_bare, 1, tol, maxiter, clover_csw),
]

# hmc_inner = HMC(latt_info, monomials[:1], O4Nf5Ng0V(4))
# hmc = HMC(latt_info, monomials[1:], O4Nf5Ng0V(3), hmc_inner)

# hmc_inner = HMC(latt_info, monomials[:1], O4Nf5Ng0V(8))
# hmc = HMC(latt_info, monomials[1:], O4Nf5Ng0V(4), hmc_inner)

hmc = HMC(latt_info, monomials, O4Nf5Ng0V(20))
# hmc = HMC(core.LatticeInfo(latt_size, t_boundary=1, anisotropy=1.0), monomials, O4Nf5Ng0V(20))


hmc.setFermionVerbosity(QudaVerbosity.QUDA_SILENT)

# gauge = core.LatticeGauge(latt_info)
# gauge = io.readQIOGauge(field_24x64)
gauge = io.readQIOGauge("/home/jyg/stout_smear/PyQUDA_stout_smeared_force/PyQUDA/tests/beta6.20_mu-0.2770_ms-0.2310_L24x64_cfg_20050.lime")
# gauge.gauss(1024, 0.05)

hmc.initialize(10086, gauge)
plaq = hmc.plaquette()
core.getLogger().info(f"Trajectory {start}:\n" f"Plaquette = {plaq}\n")
for i in range(start, stop):
    s = perf_counter()

    hmc.gaussMom()
    hmc.samplePhi()

    kinetic_old, potential_old = hmc.momAction(), hmc.gaugeAction() + hmc.fermionAction()
    energy_old = kinetic_old + potential_old

    hmc.integrate(t, 2e-15)

    kinetic, potential = hmc.momAction(), hmc.gaugeAction() + hmc.fermionAction()
    energy = kinetic + potential

    accept = hmc.accept(energy - energy_old)
    if accept or i < warm:
        hmc.saveGauge(gauge)
    else:
        hmc.loadGauge(gauge)

    plaq = hmc.plaquette()
    core.getLogger().info(
        f"Trajectory {i + 1}:\n"
        f"Plaquette = {plaq}\n"
        f"P_old = {potential_old}, K_old = {kinetic_old}\n"
        f"P = {potential}, K = {kinetic}\n"
        f"Delta_P = {potential - potential_old}, Delta_K = {kinetic - kinetic_old}\n"
        f"Delta_E = {energy - energy_old}\n"
        f"acceptance rate = {exp(min(energy_old - energy, 0)) * 100:.2f}%\n"
        f"accept? {accept}\n"
        f"warmup? {i < warm}\n"
        f"HMC time = {perf_counter() - s:.3f} secs\n"
    )

    # if (i + 1) % save == 0:
    #     gauge.save(f"./DATA/cfg/cfg_{i + 1}.npy")

# mpiexec -n 4 --bind-to none python3 test_hmc_clover_24x64.py
# mpirun -np 4 python3 test_hmc_clover_24x64.py