from pyquda_utils import core, io
from pyquda.hmc import HMC, O4Nf5Ng0V
from pyquda.action import GaugeAction, CloverWilsonAction
from pyquda_utils.hmc_param import symanzikTreeGaugeLoopParam as loopParam, fermionRationalParam as rationalParam
from pyquda.enum_quda import QudaVerbosity
from check_pyquda import field_12x32

latt_size = [12,12,12,32]
core.init(None, latt_size, resource_path=".cache/stout", enable_force_monitor=True)
latt_info = core.LatticeInfo(latt_size, t_boundary=1, anisotropy=1.0)

monomials = [
    GaugeAction(latt_info, loopParam(0.855453), 6.2),
    CloverWilsonAction(latt_info, rationalParam(2,12,15,7e-4,32,50), -0.277, 2, 1e-6, 1000, 1/0.855453**3),
]

hmc = HMC(latt_info, monomials[1:], O4Nf5Ng0V(1))
hmc.setFermionVerbosity(QudaVerbosity.QUDA_SILENT)

gauge = io.readQIOGauge(field_12x32)
print("gauge type:", type(gauge))
try:
    print("plaquette (before init):", gauge.plaquette())
except Exception as e:
    print("plaquette() error:", e)

hmc.initialize(10086, gauge)
print("initialized")
hmc.gaussMom()
print("gaussMom done")
print("momAction:", hmc.momAction())