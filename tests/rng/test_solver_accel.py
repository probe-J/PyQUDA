"""CG/MG 与 HMC 生命周期的 CPU 契约测试，不导入实际 QUDA/CUDA。

输入为 NumPy 半场、复数对角算子和库映射桩；检查原方程残差、FG 冻结、
配置/异常恢复、共享池清理及 runner 所需证据。数值力门禁仍需 p100 隔离运行。
在仓库根目录运行 python -B -m unittest discover -s tests/rng。
"""

import ast
from contextlib import ExitStack, contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np


CORE = Path(__file__).resolve().parents[2] / "pyquda_core/pyquda"
PACKAGE = "_cpu_solver_accel"


class Half(np.ndarray):
    data = property(lambda self: self)

    def norm2(self):
        return float(np.vdot(self, self).real)


class Field:
    """half 数据保存在首个槽；暴露 NumPy 指针替身，不分配 GPU。"""

    def __init__(self, li=None):
        self._data = np.zeros((2, 2, 4, 3), dtype=complex)

    data = property(lambda self: self._data, lambda self, value: self._data.__setitem__(slice(None), value))
    even = property(lambda self: self._data[0].view(Half),
                    lambda self, value: self._data.__setitem__(0, value))
    even_ptr = property(lambda self: self.even)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_methods(path, name, methods, namespace):
    # AST 保留真实方法实现；去掉导入/基类，防止测试触发 GPU 初始化。
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    cls.bases = []
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in methods]
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    tree = ast.fix_missing_locations(ast.Module(body=[future, cls], type_ignores=[]))
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace[name]


class TestSolverAccel(unittest.TestCase):
    def setUp(self):
        self.modules = patch.dict(sys.modules)
        self.modules.start()
        self.addCleanup(self.modules.stop)
        package = ModuleType(PACKAGE)
        package.__path__ = [str(CORE)]
        sys.modules[PACKAGE] = package
        # enum_quda.py 由构建生成；从被测模块使用的名称创建纯 CPU 枚举替身。
        self.enums = ModuleType(PACKAGE + ".enum_quda")
        tree = ast.parse((CORE / "solver_accel.py").read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id.startswith("Quda"):
                cls = getattr(self.enums, node.value.id, None)
                if cls is None:
                    cls = SimpleNamespace()
                    setattr(self.enums, node.value.id, cls)
                if not hasattr(cls, node.attr):
                    setattr(cls, node.attr, len(vars(cls)))
        self.enums.QUDA_MAX_DIM, self.enums.QUDA_MAX_MG_LEVEL = 6, 5
        sys.modules[self.enums.__name__] = self.enums
        field = ModuleType(PACKAGE + ".field")
        field.LatticeFermion = Field
        sys.modules[field.__name__] = field
        kernel = ModuleType(PACKAGE + ".quda")
        self.calls = []

        def mat(out, inp, param):
            z = 2.0 + 1j * param.mu
            out[:] = (z.conjugate() if param.dagger == self.enums.QudaDagType.QUDA_DAG_YES else z) * inp

        def invert(out, inp, param):
            self.calls.append((param.mu, param.compute_action, param.solve_type))
            z = 2.0 + 1j * param.mu
            normal = param.solution_type == self.enums.QudaSolutionType.QUDA_MATPCDAG_MATPC_SOLUTION
            out[:] = inp / (abs(z)**2 if normal else z)
            param.iter = 3

        kernel.MatQuda, kernel.invertQuda = Mock(side_effect=mat), Mock(side_effect=invert)
        kernel.cloverQuda = Mock(side_effect=lambda out, inp, *args: out.__setitem__(slice(None), inp))
        sys.modules[kernel.__name__] = kernel
        self.kernel = kernel
        gamma = ModuleType(PACKAGE + ".gamma")
        gamma.Gamma = lambda _: SimpleNamespace(sparse_data=[1, 1, -1, -1], sparse_indices=[0, 1, 2, 3])
        sys.modules[gamma.__name__] = gamma
        self.s = load_module(PACKAGE + ".solver_accel", CORE / "solver_accel.py")
        self.mgs = []
        mgs = self.mgs

        class FakeMG:
            def __init__(self, param, invert_param):
                self.param, self.invert_param = param, invert_param
                self.instance = None
                self.updates = []
                self.destroy = Mock()
                mgs.append(self)

            def new(self):
                self.instance = object()

            def update(self, thin):
                self.updates.append((thin, tuple(self.param.setup_maxiter_refresh)))

        general = ModuleType(PACKAGE + ".dirac.general")
        general.Multigrid = FakeMG
        general.newQudaMultigridParam = lambda *args: (SimpleNamespace(n_level=2), SimpleNamespace())
        sys.modules[general.__name__] = general
        self.seed_patcher = patch.object(self.s, "_set_mg_setup_seed")
        self.seed = self.seed_patcher.start()
        self.addCleanup(self.seed_patcher.stop)
        self.li = SimpleNamespace(size=[8]*4, global_size=[8]*4, grid_size=[1]*4)
        self.param = self.make_param()
        self.action = SimpleNamespace(latt_info=self.li, label="ratio:numer",
                                      dirac=SimpleNamespace(setPrecision=Mock()))
        self.manager = self.s.SolverAcceleration(self.action, {"multigrid": {"enabled": True}})
        self.rhs, self.out = Field(), Field()
        self.rhs.even = 1 + 2j
        self.performance = Mock()

    def make_param(self):
        fields = self.s._BASE_FIELDS + self.s._PRECISION_FIELDS
        names = ("inv_type inv_type_precondition dslash_type_precondition compute_action compute_true_res "
                 "gcrNkrylov tol_restart reliable_delta tol_precondition maxiter_precondition schwarz_type "
                 "precondition_cycle omega preconditioner chrono_use_resident chrono_make_resident "
                 "chrono_replace_last use_init_guess iter").split()
        p = SimpleNamespace(**dict.fromkeys(fields + tuple(names), 0))
        e = self.enums
        p.mass, p.kappa, p.mu, p.tol = -0.1, 0.125, 0.2, 1e-12
        p.matpc_type = e.QudaMatPCType.QUDA_MATPC_ODD_ODD
        p.solution_type = e.QudaSolutionType.QUDA_MATPCDAG_MATPC_SOLUTION
        p.solve_type = e.QudaSolveType.QUDA_NORMOP_PC_SOLVE
        p.dslash_type = e.QudaDslashType.QUDA_CLOVER_HASENBUSCH_TWIST_DSLASH
        p.gamma_basis = e.QudaGammaBasis.QUDA_DEGRAND_ROSSI_GAMMA_BASIS
        p.dagger = e.QudaDagType.QUDA_DAG_NO
        return p

    def solve(self):
        self.manager.solve(self.out, self.rhs, self.param, self.performance)

    def test_config_fail_closed_and_precision_application(self):
        for config in ({"predictor": {"enabled": True}}, {"predictor": {"formulation": "chroma_quda_two_step"}},
                       {"multigrid": {"seed_backend": "builtin_fixed1234"}}, {"telemetry": True}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                self.s.SolverAccelConfig.from_dict(config)
        self.manager.configure({"plain_precision": "double"})
        self.action.dirac.setPrecision.assert_called_once()
        with self.assertRaisesRegex(ValueError, "configure"):
            self.manager.config = {}
        self.manager.solve_plain(self.out, self.rhs, self.param, self.performance)
        with self.assertRaisesRegex(ValueError, "first solve"):
            self.manager.configure({})

    def test_plain_action_and_exception_restore(self):
        before = vars(self.param).copy()
        self.manager.solve_plain(self.out, self.rhs, self.param, self.performance, compute_action=True)
        self.assertEqual(self.calls[-1][1], 1)
        self.assertEqual({k: v for k, v in vars(self.param).items() if k != "iter"},
                         {k: v for k, v in before.items() if k != "iter"})
        self.kernel.invertQuda.side_effect = RuntimeError("invert failure")
        before = vars(self.param).copy()
        with self.assertRaisesRegex(RuntimeError, "invert failure"):
            self.manager.solve_plain(self.out, self.rhs, self.param, self.performance)
        self.assertEqual(vars(self.param), before)

    def test_two_directions_original_residual_and_evidence(self):
        self.s.configure_solver_accel_telemetry(enabled=True)
        before = vars(self.param).copy()
        self.solve()
        np.testing.assert_allclose(self.out.even, self.rhs.even / 4.04)
        self.assertLess(self.manager.last_validation["true_residual"], 1e-12)
        self.assertEqual(self.calls[0][0], -0.2)
        self.assertEqual(self.calls[1][0], 0.2)
        for key in before.keys() - {"iter"}:
            self.assertEqual(getattr(self.param, key), before[key], key)
        events = self.s.get_solver_accel_telemetry()["events"]
        solves = [e for e in events if e["event"] == "solve"]
        self.assertEqual([e["direction"] for e in solves], ["Mdag", "M"])
        self.assertTrue(all(e["multigrid_used"] and not e["fallback"] for e in solves))
        self.assertEqual(solves[0]["physical_shift"], 0.2)
        # 使用主 runner 的真实 verifier，避免仅对镜像断言自证。
        support = load_module("_solver_evidence_support", CORE.parents[1] / "tests/rng/chroma_hmc_support.py")
        with self.manager.suspended():
            self.solve()
        support.verify_mg_evidence(self.s.get_solver_accel_telemetry(), [self.action.label], temporary=True)

    def test_cg_evidence_preserves_md_sample_action_and_temporary_scope(self):
        self.manager.configure({})
        self.s.configure_solver_accel_telemetry(enabled=True)
        self.s.set_solver_accel_validation(True)
        self.solve()
        self.assertEqual(self.manager.last_solve["scope"], "md")
        self.manager.solve_plain(self.out, self.rhs, self.param, self.performance)
        self.manager.solve_plain(self.out, self.rhs, self.param, self.performance, compute_action=True)
        with self.manager.suspended():
            self.solve()
            self.assertEqual(self.manager.last_solve["scope"], "temporary")
        # solve 与原方程残差必须使用同一来源分类，FG 显式 MD 仍应归 temporary。
        events = self.s.get_solver_accel_telemetry()["events"]
        for kind in ("solve", "normal_residual"):
            selected = [event for event in events if event["event"] == kind]
            self.assertEqual([event["scope"] for event in selected], ["md", "sample", "action", "temporary"])
        self.assertTrue(all(not event["multigrid_used"] for event in events if event["event"] == "solve"))

    def test_shared_pool_identity_and_last_user_cleanup(self):
        self.solve()
        other = self.s.SolverAcceleration(self.action, self.manager.config)
        self.param.mu = 0.4
        other.solve(self.out, self.rhs, self.param, self.performance)
        self.assertIs(self.manager._entry, other._entry)
        self.assertEqual(len(self.mgs), 1)
        self.manager.close()
        self.mgs[0].destroy.assert_not_called()
        other.close()
        self.mgs[0].destroy.assert_called_once()
        self.assertFalse(self.s._MG_POOL)
        self.assertIsNone(self.s._MG_ACTIVE)

    def test_pool_isolates_mass_and_stout_context(self):
        self.solve()
        original = self.manager._entry
        other = self.s.SolverAcceleration(self.action, self.manager.config)
        other.operator_context = (1, 0.125, 3, 1.0)
        other.solve(self.out, self.rhs, self.param, self.performance)
        self.assertIsNot(original, other._entry)
        self.param.mass = 0.3
        other.solve(self.out, self.rhs, self.param, self.performance)
        self.assertEqual(len(self.mgs), 3)

    def test_fg_cold_refusal_and_no_refresh_rebuild(self):
        with self.manager.suspended(), self.assertRaisesRegex(RuntimeError, "existing healthy"):
            self.solve()
        self.assertFalse(self.mgs)
        self.solve()
        self.manager._refresh_pending = True
        with self.manager.suspended():
            with self.manager.suspended():
                self.solve()
            with self.assertRaisesRegex(RuntimeError, "temporary"):
                self.manager.configure({})
            other = self.s.SolverAcceleration(self.action, self.manager.config)
            with self.assertRaisesRegex(RuntimeError, "shared temporary"):
                other.solve(self.out, self.rhs, self.param, self.performance)
            self.assertTrue(self.manager._refresh_pending)
        self.assertEqual(len(self.mgs), 1)
        self.assertTrue(all(not thin and not any(refresh) for thin, refresh in self.mgs[0].updates))
        self.solve()
        self.assertEqual(set(self.mgs[0].updates[-1][1]), {100})
        self.assertFalse(self.manager._refresh_pending)

    def test_fg_failure_defers_then_normal_rebuilds(self):
        self.solve()
        original = self.mgs[0]
        with self.manager.suspended(), patch.object(self.manager, "_residual", return_value=float("nan")):
            with self.assertRaises(self.s.SolverValidationError):
                self.solve()
        original.destroy.assert_not_called()
        self.assertTrue(self.manager._entry.deferred_rebuild)
        self.solve()
        original.destroy.assert_called_once()
        self.assertEqual(len(self.mgs), 2)

    def test_normal_failure_one_retry_and_cleanup(self):
        with patch.object(self.manager, "_residual", return_value=float("nan")):
            with self.assertRaises(self.s.SolverValidationError):
                self.solve()
        self.assertEqual(len(self.mgs), 2)
        for mg in self.mgs:
            mg.destroy.assert_called_once()
        self.assertFalse(self.s._MG_POOL)
        self.assertIsNone(self.manager._entry)

    def test_adjoint_failure_rejects_before_second_direction(self):
        self.kernel.cloverQuda.side_effect = lambda out, inp, *args: out.__setitem__(slice(None), 0)
        with self.assertRaises(self.s.SolverValidationError) as failure:
            self.solve()
        self.assertEqual(failure.exception.detail["equation"], "original_adjoint")
        self.assertEqual(len(self.calls), 2)  # 两次 attempt 均只执行第一方向。

    def test_topology_and_nonfinite_target_fail_before_allocation(self):
        self.li.size = [6, 8, 8, 8]
        with self.assertRaises(self.s.UnsupportedMultigridTopology):
            self.solve()
        self.li.size = [8]*4
        self.param.tol = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            self.solve()
        self.assertFalse(self.mgs)

    def test_builtin_attestation_api_and_loaded_mapping_checks(self):
        # 使用本地临时假库，哈希/设备/inode 均来自真实文件，不加载动态库。
        self.seed_patcher.stop()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "libquda.so"
            payload = b"CPU attestation fixture"
            path.write_bytes(payload)
            stat = path.stat()
            maps = f"1-2 r-xp 0 {os.major(stat.st_dev):x}:{os.minor(stat.st_dev):x} {stat.st_ino} {path}"
            config = self.s.MultigridConfig(seed_backend="builtin_fixed1234", setup_seed=1234,
                                            library_path=str(path), library_sha256=hashlib.sha256(payload).hexdigest())
            with patch.object(self.s.Path, "read_text", return_value=maps), patch.object(self.s.ctypes, "CDLL") as cdll:
                self.s._set_mg_setup_seed(config)
                cdll.assert_not_called()
                with patch.object(self.s.hashlib, "sha256", return_value=SimpleNamespace(update=lambda _: None, hexdigest=lambda: "0"*64)):
                    with self.assertRaisesRegex(RuntimeError, "sha256"):
                        self.s._set_mg_setup_seed(config)
                with patch.object(self.s.Path, "read_text", return_value=maps + " (deleted)"):
                    with self.assertRaisesRegex(RuntimeError, "deleted"):
                        self.s._mapped_quda_library_path()
                with patch.object(self.s.os, "stat", return_value=SimpleNamespace(st_ino=stat.st_ino + 1, st_dev=stat.st_dev)):
                    with self.assertRaisesRegex(RuntimeError, "on-disk"):
                        self.s._mapped_quda_library_path()
                cdll.side_effect = OSError("no API")
                with self.assertRaisesRegex(RuntimeError, "requires hmc_quda_set"):
                    self.s._set_mg_setup_seed(self.s.MultigridConfig())
                setter = Mock(return_value=1)
                cdll.side_effect = None
                cdll.return_value = SimpleNamespace(hmc_quda_set_mg_setup_seed=setter)
                with self.assertRaisesRegex(RuntimeError, "status=1"):
                    self.s._set_mg_setup_seed(self.s.MultigridConfig())
                setter.return_value = 0
                self.s._set_mg_setup_seed(self.s.MultigridConfig(setup_seed=17))
                setter.assert_called_with(17)

    def test_telemetry_opt_in_copy_strict_json_and_rank0_write(self):
        self.s._emit_telemetry("ignored")
        self.assertFalse(self.s.get_solver_accel_telemetry()["events"])
        comm = ModuleType("pyquda_comm")
        rank = Mock(return_value=1)
        comm.getMPIComm = lambda: SimpleNamespace(Get_rank=rank)
        with tempfile.TemporaryDirectory() as temp, patch.dict(sys.modules, pyquda_comm=comm):
            path = Path(temp) / "evidence.json"
            self.s.configure_solver_accel_telemetry(path)
            self.s.set_solver_accel_trajectory(7)
            self.s._emit_telemetry("normal_residual", true_residual=float("nan"))
            data = self.s.get_solver_accel_telemetry()
            self.assertIsNone(data["events"][0]["true_residual"])
            data["events"].clear()
            self.s.write_solver_accel_telemetry()
            self.assertFalse(path.exists())
            rank.return_value = 0
            self.s.write_solver_accel_telemetry()
            self.assertEqual(json.loads(path.read_text())["events"][0]["trajectory"], 7)

    def test_action_md_dispatch_and_parameter_restore(self):
        ns = {"parameter_scope": self.s.parameter_scope, "MatQuda": Mock(side_effect=RuntimeError("mat"))}
        cls = load_methods(CORE / "action/clover_hasenbusch.py", "CloverWilsonHasenbuschTwistAction",
                           {"_solve", "_mat"}, ns)
        obj = cls()
        obj.invert_param, obj.phi, obj.quark = self.param, self.rhs, [self.out]
        obj.md_tol, obj.action_tol = 1e-10, 1e-12
        obj.dirac, obj.solver_accel = self.action.dirac, SimpleNamespace(solve=Mock(), solve_plain=Mock())
        obj.dirac.performance = Mock()
        for md, action, method in ((False, False, "solve_plain"), (True, False, "solve"), (True, True, "solve_plain")):
            callback = getattr(obj.solver_accel, method)
            callback.side_effect = RuntimeError("dispatch")
            before = vars(self.param).copy()
            with self.assertRaisesRegex(RuntimeError, "dispatch"):
                obj._solve(mu=0.7, md=md, compute_action=action)
            self.assertEqual(vars(self.param), before)
            callback.side_effect = None
        before = vars(self.param).copy()
        with self.assertRaisesRegex(RuntimeError, "mat"):
            obj._mat(self.out, self.rhs, self.enums.QudaDagType.QUDA_DAG_YES, 0.8)
        self.assertEqual(vars(self.param), before)

    def test_hmc_nested_hooks_and_fg_state_restore(self):
        ns = {"__package__": PACKAGE, "ExitStack": ExitStack, "LatticeGauge": Field, "LatticeMom": Field}
        cls = load_methods(CORE / "hmc.py", "HMC", {"_solverAccelerators", "configureSolverAccel",
                           "invalidateSolverAccel", "beginTrajectory", "closeSolverAccel", "forceGradient"}, ns)
        root, inner = cls(), cls()
        root.hmc_inner, inner.hmc_inner = inner, None
        root.stout_config = inner.stout_config = (1, .125, 4, 1.)
        manager = SimpleNamespace(configure=Mock(), invalidate=Mock(), close=Mock(), suspended=Mock())
        root.fermion_monomials = [SimpleNamespace(solver_accel=manager)]
        inner.fermion_monomials = [SimpleNamespace(solver_accel=manager)]
        self.assertEqual(list(root._solverAccelerators()), [manager])
        root.configureSolverAccel({})
        manager.configure.assert_called_once()
        root.beginTrajectory(3)
        self.assertEqual(self.s._TRAJECTORY, 3)
        root.closeSolverAccel()
        manager.close.assert_called_once()
        root.latt_info = self.li
        state = {"gauge": 5., "mom": 7., "frozen": 0}

        @contextmanager
        def suspended():
            state["frozen"] += 1
            try:
                yield
            finally:
                state["frozen"] -= 1

        manager.suspended.side_effect = suspended
        root.saveGauge = lambda field: setattr(field, "data", state["gauge"])
        root.saveMom = lambda field: setattr(field, "data", state["mom"])
        root.loadGauge = lambda field: state.update(gauge=float(field.data.flat[0].real))
        root.loadMom = lambda field: state.update(mom=float(field.data.flat[0].real))
        root.updateGauge = lambda dt: state.update(gauge=99.)

        def update(dt):
            self.assertEqual(state["frozen"], 1)
            state["mom"] += dt

        root.updateMom = update
        root.forceGradient(.1, .2)
        self.assertEqual(state, {"gauge": 5., "mom": 7.2, "frozen": 0})
        root.updateMom = Mock(side_effect=[None, RuntimeError("force")])
        with self.assertRaisesRegex(RuntimeError, "force"):
            root.forceGradient(.1, .2)
        self.assertEqual(state, {"gauge": 5., "mom": 7.2, "frozen": 0})

    def test_fg_load_gauge_invalidation_and_recovery_failures(self):
        # 使用真实 loadGauge/invalidate 和 MG manager，覆盖冻结内恢复与双重恢复失败。
        class Gauge(Field):
            data_ptrs = property(lambda self: self.data)

            def copy(self):
                copied = Gauge()
                copied.data = self.data
                return copied

        state = {"gauge": 5., "mom": 7.}
        gauge_error, momentum_error = RuntimeError("gauge restore"), RuntimeError("momentum restore")
        fail_gauge = fail_momentum = fail_force = False
        restoring_gauge = False
        rollback_attempts = []

        def load_gauge(ptr, param):
            nonlocal restoring_gauge
            restoring_gauge = True
            self.assertEqual(self.manager._suspended, 1)
            self.assertTrue(self.manager._entry.dirty)
            if fail_gauge:
                raise gauge_error
            state["gauge"] = float(ptr.flat[0].real)

        ns = {"__package__": PACKAGE, "ExitStack": ExitStack, "LatticeGauge": Gauge, "LatticeMom": Field,
              "QudaTboundary": SimpleNamespace(QUDA_ANTI_PERIODIC_T=1), "loadGaugeQuda": load_gauge}
        cls = load_methods(CORE / "hmc.py", "HMC", {"_solverAccelerators", "invalidateSolverAccel",
                           "forceGradient", "loadGauge"}, ns)
        root = cls()
        root.latt_info, root.hmc_inner, root.stout_config = self.li, None, (1, .125, 4, 1.)
        root.fermion_monomials = [SimpleNamespace(solver_accel=self.manager)]
        root.is_staggered = False
        root.gauge_param = SimpleNamespace(t_boundary=0, use_resident_gauge=1)
        list(root._solverAccelerators())
        self.solve()
        basis = self.manager._mg
        root.saveGauge = lambda field: setattr(field, "data", state["gauge"])
        root.saveMom = lambda field: setattr(field, "data", state["mom"])

        def load_mom(field):
            if restoring_gauge:
                rollback_attempts.append(float(field.data.flat[0].real))
                if fail_momentum:
                    raise momentum_error
            state["mom"] = float(field.data.flat[0].real)

        def update(dt):
            self.manager.gauge_updated()
            self.solve()
            state["mom"] += dt
            if fail_force and dt == .2:
                raise RuntimeError("force")

        root.loadMom, root.updateMom = load_mom, update
        root.updateGauge = lambda dt: state.update(gauge=99.)
        for fail_force, fail_gauge, fail_momentum in ((False, False, False), (True, False, False),
                                                      (False, True, False), (True, True, False),
                                                      (False, True, True)):
            with self.subTest(force=fail_force, gauge=fail_gauge, momentum=fail_momentum):
                state.update(gauge=5., mom=7.)
                restoring_gauge = False
                rollback_attempts.clear()
                if fail_gauge:
                    with self.assertRaises(RuntimeError) as failure:
                        root.forceGradient(.1, .2)
                    self.assertIs(failure.exception, gauge_error)
                    self.assertEqual(rollback_attempts, [7.])
                    if fail_momentum:
                        self.assertIs(failure.exception.__cause__, momentum_error)
                elif fail_force:
                    with self.assertRaisesRegex(RuntimeError, "force"):
                        root.forceGradient(.1, .2)
                else:
                    root.forceGradient(.1, .2)
                if not fail_momentum:
                    self.assertEqual(state["mom"], 7. if fail_force or fail_gauge else 7.2)
                if not fail_gauge:
                    self.assertEqual(state["gauge"], 5.)
                self.assertEqual(self.manager._suspended, 0)
                self.assertTrue(self.manager._entry.dirty)
                self.assertIs(self.manager._mg, basis)
                basis.destroy.assert_not_called()
                self.assertTrue(all(not any(refresh) for _, refresh in basis.updates))
                if not fail_gauge:
                    # gauge 恢复成功后，普通求解消费 dirty，仍复用正常 basis。
                    self.solve()
                    self.assertFalse(self.manager._entry.dirty)
                    self.assertIs(self.manager._mg, basis)


if __name__ == "__main__":
    unittest.main()
