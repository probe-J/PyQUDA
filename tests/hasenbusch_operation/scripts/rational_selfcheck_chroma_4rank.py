import sys
import logging
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, "/home/jyg/stout_smear/PyQUDA_stout_smeared_force/PyQUDA/tests/hasenbusch_operation/scripts")
sys.path.insert(0, "/home/jyg/stout_smear/PyQUDA_stout_smeared_force/PyQUDA")

import run_hmc_hasenbusch_12x32 as hr
from chroma_rng_adapter import attach_chroma_rng
from pyquda.enum_quda import QudaVerbosity
from pyquda_utils import core, io


args = SimpleNamespace(
    cfg="1000",
    gauge_path=None,
    t_boundary=-1,
    trajectory_length=0.1,
    chroma_traj_length=None,
    stop=1,
    start=0,
    warm=0,
    seed=20260910,
    run_id="rational_selfcheck_chroma_4rank",
    tol=1e-10,
    nsteps_inner=2,
    nsteps_outer=14,
    placement="cancel_inner",
    integrator_inner="STSForceGrad",
    integrator_outer="STSForceGrad",
    hasenbusch_mu="0,0.0005,0.0015,0.0035,0.0075,0.0155",
    hasenbusch_force_sign=1.0,
    hasenbusch_force_coeff_scale=1.0,
    hasenbusch_force_output_scale=1.0,
    sym_odd_clover_scale=1.0,
    hasenbusch_force_dagger="yes",
    monomial_subset="full",
    ratio_force_source="phi",
    ratio_force_mode="pc_quda_norm",
    cancel_force_mode="pc_quda_norm",
    cancel_sample_scale=1.0,
    cancel_constdet="yes",
    matpc_type="odd_odd",
    stout_dim=4,
    stout_rho=0.125,
    stout_n_smear=1,
    strange_matpc_type="odd_odd",
    strange_action_rational=True,
)

config = hr.base_config(args)
cache = Path("/home/jyg/stout_smear/PyQUDA_stout_smeared_force/PyQUDA/tests/hasenbusch_operation/rng_results/.cache/rational_selfcheck_chroma_4rank")
cache.mkdir(parents=True, exist_ok=True)
core.init(None, config["latt_size"], resource_path=str(cache), enable_force_monitor=False)
latt_info = core.LatticeInfo(config["latt_size"], t_boundary=config["t_boundary"], anisotropy=1.0)
hmc = hr.make_hmc(latt_info, config, logging.getLogger("rational_selfcheck_chroma_4rank"))
hmc.setFermionVerbosity(QudaVerbosity.QUDA_SILENT)
gauge = io.readQIOGauge(str(config["gauge_path"]))
hmc.initialize(config["seed"], gauge)
handle = attach_chroma_rng(hmc, seed=config["seed"], precision="double", use_chroma_accept=False)
hmc.gaussMom()
hmc.samplePhi()


def iter_fermion(hmc):
    for monomial in hmc.fermion_monomials:
        yield monomial
    inner = getattr(hmc, "hmc_inner", None)
    if inner is not None:
        yield from iter_fermion(inner)


def find_strange(hmc):
    for monomial in iter_fermion(hmc):
        if type(monomial).__name__ == "CloverWilsonAction":
            return monomial
    raise RuntimeError("strange not found")


monomial = find_strange(hmc)
constant = latt_info.volume // 2 * latt_info.Ns * latt_info.Nc
eta_norm2 = float(monomial.eta.norm2())
phi_norm2 = float(monomial.phi.norm2())
hmc.loadGaugeMomSmeared()
action = float(monomial.action())
hmc.loadGaugeMom()
print("eta_norm2", eta_norm2)
print("phi_norm2", phi_norm2)
print("constant", constant)
print("action", action)
print("action_plus_constant", action + constant)
print("ratio_action_plus_constant_over_eta_norm2", (action + constant) / eta_norm2)
handle.detach()
