"""质量比作用量的 CPU 契约测试：AST 加载库实现，以 NumPy 场替换 QUDA。

在仓库根目录运行 python -B -m unittest discover -s tests/rng -p test_mass_ratio.py -v。
输入为标量算子桩、可注入故障的导数桩；检查采样顺序、四项配对、求解分派、
步长与状态恢复。仅输出 unittest 结果，不初始化 GPU、不验证物理力或有限差分。
"""

import ast
from contextlib import contextmanager
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np


ACTION_PATH = Path(__file__).resolve().parents[2] / "pyquda_core/pyquda/action/clover_mass_ratio.py"
DAG = SimpleNamespace(QUDA_DAG_NO="no", QUDA_DAG_YES="yes")
PRECISION = SimpleNamespace(QUDA_DOUBLE_PRECISION="double")


class Field:
    """保持 data 赋值的场语义，标量赋值清零原缓冲而不替换数组。"""

    def __init__(self, shape=(2, 1)):
        self._data = np.zeros(shape, dtype=complex)

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self._data[...] = value

    @property
    def data_ptrs(self):
        return self.data


class Gauge(Field):
    def __init__(self, li):
        super().__init__((1, 3, 3))


class Mom(Field):
    def __init__(self, li):
        super().__init__((1, 10))


class BaseAction:
    def __init__(self, li, dirac):
        self.latt_info = li
        self.dirac = dirac
        self.invert_param = dirac.invert_param
        self.gauge_param = dirac.gauge_param

    def sampleEta(self):
        raise AssertionError("RNG adapter stub must bind outer sampleEta")


class Manager:
    def __init__(self, name, events):
        self.name, self.events = name, events
        self.operator_context = None
        self.configure = Mock()
        self.invalidate = Mock()
        self.close = Mock()

    @contextmanager
    def suspended(self):
        self.events.append((self.name, "suspend"))
        try:
            yield
        finally:
            self.events.append((self.name, "resume"))


class Helper:
    """A=2、B=3 的实数算子桩；导数返回人为权重，仅验证控制流。"""

    def __init__(self, name, events, with_manager):
        self.name, self.events = name, events
        self.factor = 2.0 if name == "A" else 3.0
        self.mu = 0.0
        self.invert_param = SimpleNamespace(mu=0.0, action=[0.0])
        self.gauge_param = SimpleNamespace(use_resident_force=7, return_result_force=9)
        self.dirac = SimpleNamespace(
            invert_param=self.invert_param, gauge_param=self.gauge_param, setPrecision=Mock(),
        )
        self.eta, self.phi, self.rhs, self.y = (Field() for _ in range(4))
        self.quark, self.source = [Field()], [Field()]
        self.force_term_history = []
        self.setVerbosity = Mock()
        self.fail_term = None
        if with_manager:
            self.solver_accel = Manager(name, events)

    def updateClover(self):
        self.events.append((self.name, "clover"))

    def _active_parity(self):
        return "physical_odd"

    def _mat(self, out, inp, dagger, mu):
        self.events.append((self.name, "mat", out, inp, dagger, mu))
        out.data = self.factor * inp.data

    def _solve(self, rhs, mu, compute_action=False, **kwargs):
        self.events.append((self.name, "solve", rhs, mu, compute_action, kwargs))
        self.quark[0].data = rhs.data / self.factor**2
        if compute_action:
            self.invert_param.action[0] = np.vdot(rhs.data[0], self.quark[0].data[0]).real

    def _add_sym_oddodd_quda_norm_deriv_force(self, force, left, right, dagger, sign, name, *, norm_scale):
        self.events.append((self.name, "term", left, right, dagger, sign, name, norm_scale))
        assert (self.gauge_param.use_resident_force, self.gauge_param.return_result_force) == (0, 1)
        self.force_term_history.append(name)
        weights = {"B(phi,x)": 1, "Bdag(x,phi)": 2, "Adag(x,y)": 3, "A(y,x)": 4}
        force.data += sign * weights[name]
        if name == self.fail_term:
            raise RuntimeError("injected derivative failure")


def load_module(helper_factory):
    # 只替换顶层库导入；真实方法、相对 kernel 导入及别名均保留在 AST 中。
    tree = ast.parse(ACTION_PATH.read_text())
    tree.body = [node for node in tree.body if not (isinstance(node, ast.ImportFrom) and node.level)]
    namespace = {
        "__name__": "pyquda.action.clover_mass_ratio", "__package__": "pyquda.action",
        "FermionAction": BaseAction, "CloverWilsonHasenbuschTwistAction": helper_factory,
        "QudaDagType": DAG, "QudaPrecision": PRECISION, "LatticeGauge": Gauge, "LatticeMom": Mom,
    }
    exec(compile(tree, str(ACTION_PATH), "exec"), namespace)
    return namespace


class TestMassRatio(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.helpers = []
        self.with_managers = False

        def factory(*args, **kwargs):
            name = "A" if not self.helpers else "B"
            helper = Helper(name, self.events, self.with_managers)
            self.helpers.append(helper)
            return helper

        self.factory = Mock(side_effect=factory)
        self.ns = load_module(self.factory)
        self.cls = self.ns["CloverWilsonScalarMassRatioAction"]
        self.li = SimpleNamespace(Nd=4, anisotropy=1.0)

    def make_action(self, **kwargs):
        options = dict(li=self.li, mass=-0.2, mass_precond=-0.1, tol=1e-9, maxiter=100,
                       clover_csw=1.2, md_tol=1e-8, action_tol=1e-12, label="ratio")
        options.update(kwargs)
        return self.cls(**options)

    def sample(self, action):
        def eta():
            self.events.append(("outer", "eta"))
            action.eta.data = [[6], [99]]

        action.sampleEta = Mock(side_effect=eta)
        action.sample()

    def test_constructor_bare_masses_and_cg_compatibility(self):
        action = self.make_action()
        self.assertIs(self.ns["ScalarMassRatioAction"], self.cls)
        self.assertIsNone(action.solver_accel)
        for call, mass, suffix, helper in zip(self.factory.call_args_list, (-0.2, -0.1),
                                              ("numer", "denom"), self.helpers):
            self.assertEqual(call.args, (self.li, mass, 0.0, 1e-9, 100, 1.2))
            self.assertEqual(call.kwargs, dict(
                md_tol=1e-8, action_tol=1e-12, constdet=True, determinant_ratio=False,
                matpc_type="odd_odd", cancel_force_mode="pc_quda_norm", cancel_sample_scale=1.0,
                label=f"ratio:{suffix}",
            ))
            helper.dirac.setPrecision.assert_called_once_with(
                cuda="double", sloppy="double", refinement_sloppy="double",
                precondition="double", eigensolver="double",
            )
        for name in ("eta", "phi", "rhs", "quark", "y", "source"):
            self.assertIs(getattr(action, name), getattr(action.numer, name))
        self.assertEqual(action._active_parity(), "physical_odd")

    def test_invalid_parameters_fail_before_helper_construction(self):
        cases = [dict(mass=-4), dict(mass_precond=float("nan")), dict(clover_csw=0),
                 dict(tol=1), dict(md_tol=0), dict(action_tol=float("inf")),
                 dict(maxiter=0), dict(maxiter=1.5), dict(maxiter=float("inf")),
                 dict(li=SimpleNamespace(Nd=3, anisotropy=1.0)),
                 dict(li=SimpleNamespace(Nd=4, anisotropy=2.0))]
        for kwargs in cases:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.make_action(**kwargs)
        self.factory.assert_not_called()

    def test_sample_order_single_eta_and_no_rescaling(self):
        action = self.make_action()
        self.sample(action)
        action.sampleEta.assert_called_once_with()
        self.assertEqual(self.events, [
            ("outer", "eta"), ("A", "clover"),
            ("A", "mat", action.rhs, action.eta, "yes", 0.0), ("B", "clover"),
            ("B", "solve", action.rhs, 0.0, False, {}),
            ("B", "mat", action.phi, action.denom.quark[0], "no", 0.0),
        ])
        np.testing.assert_array_equal(action.eta.data, [[6], [0]])
        np.testing.assert_allclose(action.phi.data, [[4], [0]])
        np.testing.assert_array_equal(action.source[0].data, action.phi.data)

    def test_action_bdag_then_a_cg_without_constant(self):
        action = self.make_action()
        action.phi.data = [[4], [0]]
        self.assertEqual(action.action(), 36.0)
        self.assertEqual(self.events, [
            ("B", "clover"), ("B", "mat", action.rhs, action.phi, "yes", 0.0),
            ("A", "clover"), ("A", "solve", action.rhs, 0.0, True, {}),
        ])

    def test_force_four_terms_and_mass_activation(self):
        action = self.make_action()
        action.phi.data = [[4], [0]]
        result = action.force_v2(1.0)
        x = action.quark[0]
        terms = [event for event in self.events if event[1] == "term"]
        self.assertEqual(terms, [
            ("B", "term", action.phi, x, "no", -1.0, "B(phi,x)", 1.0),
            ("B", "term", x, action.phi, "yes", -1.0, "Bdag(x,phi)", 1.0),
            ("A", "term", x, action.y, "yes", 1.0, "Adag(x,y)", 1.0),
            ("A", "term", action.y, x, "no", 1.0, "A(y,x)", 1.0),
        ])
        self.assertEqual([e for e in self.events if e[1] == "clover"],
                         [("B", "clover"), ("A", "clover"), ("B", "clover"), ("A", "clover")])
        np.testing.assert_array_equal(action.y.data, [[6], [0]])
        np.testing.assert_array_equal(result.data, np.full((1, 3, 3), 4))
        for helper in self.helpers:
            self.assertEqual((helper.gauge_param.use_resident_force, helper.gauge_param.return_result_force), (7, 9))

    def test_force_scales_only_new_contribution(self):
        action = self.make_action()
        for dt in (0, 0.25, 1, -0.5):
            with self.subTest(dt=dt):
                force = Gauge(self.li)
                force.data = np.arange(9).reshape(1, 3, 3)
                before = force.data.copy()
                self.assertIs(action.force_v2(dt, force), force)
                np.testing.assert_array_equal(force.data, before + dt * 4)
                self.assertEqual(action.numer.force_term_history, ["Adag(x,y)", "A(y,x)"])
                self.assertEqual(action.denom.force_term_history, ["B(phi,x)", "Bdag(x,phi)"])

    def test_partial_force_failure_restores_flags_and_caller(self):
        action = self.make_action()
        for helper, name in ((action.denom, "Bdag(x,phi)"), (action.numer, "A(y,x)")):
            with self.subTest(name=name):
                force = Gauge(self.li)
                force.data = 11
                helper.fail_term = name
                with self.assertRaisesRegex(RuntimeError, "injected derivative"):
                    action.force_v2(0.25, force)
                np.testing.assert_array_equal(force.data, np.full((1, 3, 3), 11))
                for h in self.helpers:
                    self.assertEqual((h.gauge_param.use_resident_force, h.gauge_param.return_result_force), (7, 9))
                helper.fail_term = None

    def test_invalid_force_inputs_do_no_operator_work(self):
        action = self.make_action()
        for method in (action.force, action.force_v2):
            for dt in (float("nan"), float("inf"), -float("inf")):
                with self.subTest(method=method.__name__, dt=dt), self.assertRaises(ValueError):
                    method(dt)
        with self.assertRaises(TypeError):
            action.force_v2(1, Mom(self.li))
        with self.assertRaises(TypeError):
            action.force(1, Gauge(self.li))
        self.assertEqual(self.events, [])

    def test_activate_preserves_mu_zero_guard_and_clover_rebuild(self):
        action = self.make_action()
        action.updateClover()
        action.updateClover()
        self.assertEqual(self.events, [("A", "clover"), ("A", "clover")])
        self.assertEqual(action.numer.invert_param.compute_clover, 1)
        self.assertEqual(action.numer.invert_param.compute_clover_inverse, 1)
        self.assertEqual((action.numer.md_tol, action.numer.action_tol), (1e-8, 1e-12))
        for obj in (action.numer, action.numer.invert_param):
            obj.mu = 0.1
            with self.assertRaisesRegex(RuntimeError, "mu=0"):
                action.updateClover()
            obj.mu = 0.0
        action.setVerbosity("verbose")
        for helper in self.helpers:
            helper.setVerbosity.assert_called_once_with("verbose")

    def test_acceleration_only_for_force_and_config_is_copied(self):
        self.with_managers = True
        config = {"multigrid": {"enabled": True}}
        action = self.make_action(solver_accel=config)
        self.assertEqual(config, {"multigrid": {"enabled": True}})
        for call, helper in zip(self.factory.call_args_list, self.helpers):
            self.assertEqual(call.kwargs["solver_accel"],
                             {"multigrid": {"enabled": True}, "plain_precision": "double"})
            helper.dirac.setPrecision.assert_not_called()
        self.sample(action)
        action.action()
        action.force_v2(1)
        solves = [e for e in self.events if e[1] == "solve"]
        self.assertEqual([(e[0], e[4], e[5]) for e in solves],
                         [("B", False, {}), ("A", True, {"md": False}), ("A", False, {"md": True})])
        for helper in self.helpers:
            helper.solver_accel.invalidate.assert_called_once_with("scalar sample")

    def test_explicit_acceleration_never_silently_falls_back(self):
        with self.assertRaisesRegex(RuntimeError, "requires helper.solver_accel"):
            self.make_action(solver_accel={"multigrid": {"enabled": True}})

    def test_parsed_solver_config_is_forwarded_unchanged(self):
        self.with_managers = True
        config = SimpleNamespace(multigrid=SimpleNamespace(enabled=True), plain_precision="double")
        self.make_action(solver_accel=config)
        for call in self.factory.call_args_list:
            self.assertIs(call.kwargs["solver_accel"], config)

    def test_manager_lifecycle_for_both_masses(self):
        self.with_managers = True
        action = self.make_action()
        manager = action.solver_accel
        context = object()
        manager.operator_context = context
        self.assertIs(manager.operator_context, context)
        config = object()
        manager.configure(config)
        manager.invalidate("gauge", operator_changed=True)
        with self.assertRaisesRegex(RuntimeError, "FG"):
            with manager.suspended():
                raise RuntimeError("FG")
        self.assertEqual(self.events, [("A", "suspend"), ("B", "suspend"), ("B", "resume"), ("A", "resume")])
        manager.close()
        for helper in self.helpers:
            self.assertIs(helper.solver_accel.operator_context, context)
            helper.solver_accel.configure.assert_called_once_with(config)
            helper.solver_accel.invalidate.assert_called_once_with("gauge", operator_changed=True)
            helper.solver_accel.close.assert_called_once_with()

    def test_momentum_projection_dt_once_and_padding(self):
        action = self.make_action()
        links = np.diag([1j, -1j, 1])[None, ...]
        link_force = Gauge(self.li)
        link_force.data = np.array([[1j, 2 + 3j, 4 + 5j], [0, 2j, 6 + 7j], [0, 0, 3j]])
        action.force_v2 = Mock(return_value=link_force)
        quda = ModuleType("pyquda.quda")

        def save(out, param):
            self.assertIs(param, action.gauge_param)
            out[...] = links

        quda.saveGaugeQuda = Mock(side_effect=save)
        # 预先给出 U*F 的反厄米无迹部分及 MILC 十实数布局，不调用生产投影求期望值。
        expected = np.array([[-1.5, 1, -2.5, 2, 3.5, -3, -1, -1, 2, 0]])
        with patch.dict(sys.modules, {"pyquda.quda": quda}):
            for dt in (0, 0.25, 1, -0.5):
                with self.subTest(dt=dt):
                    delta = action.force(dt)
                    np.testing.assert_array_equal(delta.data, dt * expected)
                    mom = Mom(self.li)
                    mom.data = 17
                    self.assertIs(action.force(dt, mom), mom)
                    np.testing.assert_array_equal(mom.data, 17 + dt * expected)
        self.assertTrue(all(call.args == (1.0,) for call in action.force_v2.call_args_list))
        self.assertEqual(quda.saveGaugeQuda.call_count, 8)

    def test_module_has_no_experiment_imports_or_fd_gate(self):
        tree = ast.parse(ACTION_PATH.read_text())
        imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
        for node in imports:
            if isinstance(node, ast.ImportFrom):
                if node.module != "contextlib":
                    self.assertGreater(node.level, 0)
            else:
                self.assertTrue(all(alias.name in ("math", "numpy", "cupy") for alias in node.names))
        action = self.make_action()
        for name in ("fd_gate_passed", "fd_report", "finite_difference_gate", "require_fd_passed"):
            self.assertFalse(hasattr(action, name))


if __name__ == "__main__":
    unittest.main()
