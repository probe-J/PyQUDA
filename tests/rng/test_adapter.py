"""用标准库假对象验证适配器，不导入 GPU、QUDA、NumPy 或真实 MPI。

从仓库根目录运行：python -B -m unittest discover -s tests/rng -p test_adapter.py -v
验证替换时机、共享流推进、缓存隔离、slot0、接受规则和 detach；不验证随机数值内核。
"""

import importlib.util
import math
from pathlib import Path
import sys
from types import MethodType, ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch


def load_adapter():
    """从同目录加载测试支持模块，不经过 pyquda 包初始化。"""
    spec = importlib.util.spec_from_file_location("rng_test_adapter", Path(__file__).with_name("chroma_rng_adapter.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeArray:
    """最小双奇偶数组，支持深拷贝、行赋值及标量广播。"""

    def __init__(self, rows=None):
        self.rows = [list(row) for row in (rows if rows is not None else [[-1, -2], [-3, -4]])]

    def __getitem__(self, key):
        return self.rows[key]

    def __setitem__(self, key, value):
        self.rows[key] = list(value) if isinstance(value, list) else [value] * len(self.rows[key])

    def copy(self):
        return FakeArray(self.rows)


class FakeField:
    """保留标签及独立数据，用于检查场缓存复制。"""

    def __init__(self, label):
        self.label = label
        self.data = FakeArray()

    def copy(self):
        field = FakeField(self.label)
        field.data = self.data.copy()
        return field


class FakeAction:
    """原生采样仅记录调用次数，便于识别替换遗漏。"""

    def __init__(self, label, parity=None):
        self.eta = FakeField(label)
        self.active_parity = parity
        self.native_samples = 0

    def sampleEta(self):
        self.native_samples += 1


class MethodParityAction(FakeAction):
    """方法奇偶性故意与属性冲突，验证方法优先级。"""

    def _active_parity(self):
        return 1


class FakeLogDetAction(FakeAction):
    """模拟真实 LogDet：继承 sampleEta，但不创建 eta 场。"""

    def __init__(self):
        self.native_samples = 0


class FakeHMC:
    """仅提供适配器使用的原生接口，不提供任何 solver_accel 方法。"""

    def __init__(self, actions=(), inner=None):
        self.latt_info = SimpleNamespace(global_size=["2", "2", "2", "2"])
        self.fermion_monomials = list(actions)
        self.hmc_inner = inner
        self.gauge_param = SimpleNamespace(use_resident_mom=0, make_resident_mom=0, return_result_mom=1)
        self.native_momenta = 0
        self.native_accepts = []
        self.loaded = []

    def gaussMom(self):
        self.native_momenta += 1

    def loadMom(self, momentum):
        self.loaded.append(momentum)

    def accept(self, delta_s):
        self.native_accepts.append(delta_s)
        return "native"


class FakeComm:
    """记录广播输入，可模拟根 rank 覆盖本地结果。"""

    def __init__(self):
        self.rank = 0
        self.calls = []
        self.result = None

    def bcast(self, value):
        self.calls.append(value)
        return value if self.result is None else self.result


class FakeRNG:
    """严格匹配目标构造 API；用固定步长代替数值内核以检查调用次数。"""

    instances = []

    def __init__(self, global_latt_size, precision="double"):
        self.size = global_latt_size
        self.precision = precision
        self.events = []
        self.uniforms = []
        self.instances.append(self)

    def set_seed(self, seed):
        self.seed = seed

    def get_seed(self):
        return self.seed

    def make_momentum(self, latt_info):
        self.events.append(("momentum", self.seed))
        self.seed += 10
        return SimpleNamespace(seed=self.seed, latt_info=latt_info)

    def fill_eta(self, field, parity):
        self.events.append(("eta", field.label, parity, self.seed))
        self.seed += 3
        field.data[0] = 0
        field.data[1] = 0
        field.data[parity] = [self.seed, -self.seed]

    def random_scalar(self):
        self.events.append(("uniform", self.seed))
        self.seed += 1
        return self.uniforms.pop(0)

    def dump_reference(self, path, seed, mode):
        self.events.append(("dump", path, seed, mode))


class AdapterTests(unittest.TestCase):
    """通过假模块隔离真实依赖，检查公开运行时行为。"""

    def setUp(self):
        self.adapter = load_adapter()
        FakeRNG.instances = []
        self.comm = FakeComm()
        pyquda = ModuleType("pyquda")
        pyquda.__path__ = []
        rng_module = ModuleType("pyquda.chroma_rng")
        rng_module.ChromaRNG = FakeRNG
        comm_module = ModuleType("pyquda_comm")
        comm_module.getMPIComm = lambda: self.comm
        self.modules = patch.dict(sys.modules, {"pyquda": pyquda, "pyquda.chroma_rng": rng_module,
                                               "pyquda_comm": comm_module})
        self.modules.start()
        self.addCleanup(self.modules.stop)
        self.outer_action = FakeAction("outer")
        self.inner_action = FakeAction("inner", 1)
        self.inner = FakeHMC([self.inner_action])
        self.hmc = FakeHMC([self.outer_action], self.inner)

    def attach(self, **kwargs):
        handle = self.adapter.attach_chroma_rng(self.hmc, seed=7, **kwargs)
        self.addCleanup(handle.detach)
        return handle

    def test_import_without_pyquda_or_gpu(self):
        with patch.dict(sys.modules, {"pyquda": None, "pyquda.chroma_rng": None,
                                      "pyquda_comm": None, "cupy": None}):
            self.assertTrue(callable(load_adapter().attach_chroma_rng))

    def test_constructor_and_immediate_eta_replacement(self):
        handle = self.attach(precision="single")
        self.assertEqual(FakeRNG.instances, [handle.rng])
        self.assertEqual(handle.rng.size, (2, 2, 2, 2))
        self.assertEqual(handle.rng.precision, "single")
        self.assertEqual(handle.rng.seed, 7)
        self.assertEqual(handle.rng.events, [])
        self.assertEqual(handle.eta_fields, {})
        for action in (self.outer_action, self.inner_action):
            self.assertIn("sampleEta", vars(action))
            with self.assertRaisesRegex(RuntimeError, "top-level gaussMom"):
                action.sampleEta()
            self.assertEqual(action.native_samples, 0)
        with self.assertRaisesRegex(RuntimeError, "top-level gaussMom"):
            self.inner.gaussMom()
        self.assertEqual(handle.rng.seed, 7)

    def test_refresh_order_and_shared_nested_momentum(self):
        handle = self.attach()
        self.hmc.gaussMom()
        self.assertEqual(handle.rng.events, [("momentum", 7), ("eta", "outer", 0, 17),
                                             ("eta", "inner", 1, 20)])
        self.inner.gaussMom()
        self.inner.gaussMom()
        self.assertIs(self.hmc.loaded[-1], self.inner.loaded[-1])
        self.assertEqual(handle.rng.seed, 23)
        for level in (self.hmc, self.inner):
            self.assertEqual(vars(level.gauge_param), {"use_resident_mom": 1, "make_resident_mom": 1,
                                                       "return_result_mom": 0})
            self.assertEqual(level.native_momenta, 0)

    def test_every_top_refresh_advances_rng_and_replaces_all_fields(self):
        handle = self.attach()
        cache = handle.eta_fields
        self.hmc.gaussMom()
        first_fields = dict(cache)
        first_momentum = self.hmc.loaded[-1]
        self.hmc.gaussMom()
        self.assertIs(cache, handle.eta_fields)
        self.assertEqual(handle.rng.seed, 39)
        self.assertEqual(handle.rng.events[3:], [("momentum", 23), ("eta", "outer", 0, 33),
                                               ("eta", "inner", 1, 36)])
        self.assertIsNot(self.hmc.loaded[-1], first_momentum)
        self.inner.gaussMom()
        self.assertIs(self.hmc.loaded[-1], self.inner.loaded[-1])
        for action, old_field in first_fields.items():
            self.assertIsNot(cache[action], old_field)
            self.assertNotEqual(cache[action].data.rows, old_field.data.rows)
            action.sampleEta()
            self.assertEqual(action.eta.data.rows, cache[action].data.rows)
        self.assertEqual(handle.rng.seed, 39)

    def test_sample_eta_copies_cache_without_advancing_rng(self):
        handle = self.attach()
        original = self.outer_action.eta.data.copy().rows
        self.hmc.gaussMom()
        self.assertEqual(self.outer_action.eta.data.rows, original)
        for action in (self.outer_action, self.inner_action):
            cached = handle.eta_fields[action]
            action.sampleEta()
            first_copy = action.eta.data
            self.assertEqual(first_copy.rows, cached.data.rows)
            self.assertIsNot(first_copy, cached.data)
            first_copy[0][0] = 999
            self.assertNotEqual(first_copy.rows, cached.data.rows)
            action.sampleEta()
            self.assertIsNot(action.eta.data, first_copy)
            self.assertEqual(action.eta.data.rows, cached.data.rows)
            self.assertEqual(action.native_samples, 0)
        self.assertEqual(handle.rng.seed, 23)
        self.assertEqual(len(handle.rng.events), 3)

    def test_parity_defaults_method_precedence_and_slot_zero(self):
        actions = [FakeAction("unset"), FakeAction("even", 0), FakeAction("odd", 1),
                   MethodParityAction("method", 0), FakeAction("missing")]
        del actions[-1].active_parity
        self.hmc.fermion_monomials = actions
        self.hmc.hmc_inner = None
        handle = self.attach()
        self.hmc.gaussMom()
        self.assertEqual([event[2] for event in handle.rng.events[1:]], [0, 0, 1, 1, 0])
        for action in actions:
            field = handle.eta_fields[action]
            self.assertNotEqual(field.data[0], [0, 0])
            self.assertEqual(field.data[1], [0, 0])
            action.sampleEta()
            self.assertEqual(action.eta.data.rows, field.data.rows)

    def test_explicit_sampling_order_is_reused_across_trajectories(self):
        self.hmc.chroma_sampling_order = iter([self.inner_action, object(), self.outer_action])
        handle = self.attach()
        for _ in range(2):
            self.hmc.gaussMom()
        self.assertEqual([event[1] for event in handle.rng.events if event[0] == "eta"],
                         ["inner", "outer", "inner", "outer"])
        self.assertEqual(len(handle.eta_fields), 2)

    def test_logdet_and_noncallable_samplers_are_skipped(self):
        logdet = FakeLogDetAction()
        noncallable = FakeAction("noncallable")
        noncallable.sampleEta = None
        empty_eta = FakeAction("empty eta")
        empty_eta.eta = None
        self.hmc.fermion_monomials.extend([logdet, noncallable])
        self.inner.fermion_monomials.append(empty_eta)
        for explicit in (False, True):
            with self.subTest(explicit=explicit):
                if explicit:
                    self.hmc.chroma_sampling_order = [logdet, self.outer_action, noncallable,
                                                      empty_eta, self.inner_action]
                handle = self.attach()
                self.hmc.gaussMom()
                self.assertEqual(handle.rng.events, [("momentum", 7), ("eta", "outer", 0, 17),
                                                     ("eta", "inner", 1, 20)])
                self.assertEqual(set(handle.eta_fields), {self.outer_action, self.inner_action})
                self.assertNotIn("sampleEta", vars(logdet))
                self.assertIs(logdet.sampleEta.__func__, FakeAction.sampleEta)
                self.assertIsNone(noncallable.sampleEta)
                self.assertNotIn("sampleEta", vars(empty_eta))
                self.assertFalse(hasattr(logdet, "eta"))
                self.assertEqual(logdet.native_samples, 0)
                handle.detach()

    def test_invalid_explicit_sampling_order_fails_before_injection(self):
        cases = [([self.outer_action, self.inner_action, self.outer_action], "duplicate"),
                 ([self.outer_action], "exactly all"),
                 ([], "exactly all"),
                 ([self.outer_action, self.inner_action, FakeAction("foreign")], "exactly all")]
        for order, message in cases:
            with self.subTest(order=order):
                self.hmc.chroma_sampling_order = order
                with self.assertRaisesRegex(ValueError, message):
                    self.attach(use_chroma_accept=True)
                self.assertEqual(FakeRNG.instances, [])
                for level in (self.hmc, self.inner):
                    self.assertNotIn("gaussMom", vars(level))
                    self.assertNotIn("accept", vars(level))
                for action in (self.outer_action, self.inner_action):
                    self.assertNotIn("sampleEta", vars(action))

    def test_empty_hmc_only_refreshes_momentum(self):
        self.hmc = FakeHMC()
        handle = self.attach()
        self.hmc.gaussMom()
        self.assertEqual(handle.rng.events, [("momentum", 7)])
        self.assertEqual(handle.eta_fields, {})

    def test_native_accept_is_default_and_does_not_need_comm(self):
        with patch.dict(sys.modules, {"pyquda_comm": None}):
            handle = self.attach()
        self.assertFalse(handle.use_chroma_accept)
        for level in (self.hmc, self.inner):
            self.assertNotIn("accept", vars(level))
            self.assertEqual(level.accept(1), "native")
            self.assertEqual(level.native_accepts, [1])
        self.assertEqual(handle.rng.events, [])
        self.assertEqual(handle.uniform_log, [])

    def test_nonpositive_accept_does_not_consume_or_broadcast(self):
        handle = self.attach(use_chroma_accept=True)
        for level in (self.hmc, self.inner):
            for delta_s in (-1000.0, -0.0, 0.0):
                self.assertIs(level.accept(delta_s), True)
                self.assertEqual(level.last_metropolis,
                                 {"acceptance_probability": 1.0, "metropolis_uniform": None})
        self.assertEqual(handle.uniform_log, [None] * 6)
        self.assertEqual(handle.rng.seed, 7)
        self.assertEqual(handle.rng.events, [])
        self.assertEqual(self.comm.calls, [])

    def test_nonfinite_delta_rejected_without_side_effects(self):
        handle = self.attach(use_chroma_accept=True)
        for level in (self.hmc, self.inner):
            for delta_s in (float("nan"), float("inf"), -float("inf")):
                with self.subTest(level=level, delta_s=delta_s):
                    with self.assertRaisesRegex(ValueError, "delta_s must be finite"):
                        level.accept(delta_s)
                    self.assertFalse(hasattr(level, "last_metropolis"))
        self.assertEqual(handle.rng.seed, 7)
        self.assertEqual(handle.rng.events, [])
        self.assertEqual(handle.uniform_log, [])
        self.assertEqual(self.comm.calls, [])
        self.hmc.accept(0.0)
        diagnostic = self.hmc.last_metropolis
        with self.assertRaisesRegex(ValueError, "finite"):
            self.hmc.accept(float("nan"))
        self.assertIs(self.hmc.last_metropolis, diagnostic)
        self.assertEqual(handle.uniform_log, [None])
        self.assertEqual(handle.rng.seed, 7)
        self.assertEqual(handle.rng.events, [])
        self.assertEqual(self.comm.calls, [])

    def test_positive_accept_boundary_reject_and_shared_advancement(self):
        handle = self.attach(use_chroma_accept=True)
        self.hmc.gaussMom()
        probability = math.exp(-1.0)
        handle.rng.uniforms = [probability, probability + 0.1, 0.0]
        self.assertIs(self.hmc.accept(1.0), True)
        self.assertIs(self.inner.accept(1.0), False)
        self.assertIs(self.hmc.accept(1000.0), True)
        self.assertEqual(handle.uniform_log, [probability, probability + 0.1, 0.0])
        self.assertEqual(self.comm.calls, [(True, probability), (False, probability + 0.1), (True, 0.0)])
        self.assertEqual(handle.rng.seed, 26)
        self.assertEqual(self.inner.last_metropolis,
                         {"acceptance_probability": probability, "metropolis_uniform": probability + 0.1})
        self.assertEqual(self.hmc.last_metropolis, {"acceptance_probability": 0.0, "metropolis_uniform": 0.0})
        self.hmc.gaussMom()
        self.assertEqual(handle.rng.events[-3], ("momentum", 26))
        self.assertEqual(handle.rng.seed, 42)

    def test_nonroot_advances_stream_but_uses_broadcast_decision(self):
        handle = self.attach(use_chroma_accept=True)
        self.comm.rank = 1
        self.comm.result = (False, 0.9)
        handle.rng.uniforms = [0.01]
        self.assertIs(self.hmc.accept(1.0), False)
        self.assertEqual(self.comm.calls, [(True, 0.01)])
        self.assertEqual(handle.uniform_log, [0.9])
        self.assertEqual(self.hmc.last_metropolis["metropolis_uniform"], 0.9)
        self.assertEqual(handle.rng.seed, 8)

    def test_detach_restores_inherited_methods_and_is_idempotent(self):
        handle = self.attach(use_chroma_accept=True)
        self.hmc.gaussMom()
        seed = handle.rng.seed
        handle.detach()
        handle.detach()
        for level in (self.hmc, self.inner):
            self.assertNotIn("gaussMom", vars(level))
            self.assertNotIn("accept", vars(level))
            level.gaussMom()
            self.assertEqual(level.native_momenta, 1)
            self.assertEqual(level.accept(-1), "native")
        for action in (self.outer_action, self.inner_action):
            self.assertNotIn("sampleEta", vars(action))
            action.sampleEta()
            self.assertEqual(action.native_samples, 1)
        self.assertEqual(handle.rng.seed, seed)

    def test_detach_preserves_existing_instance_overrides(self):
        self.hmc.gaussMom = MethodType(lambda self: "custom momentum", self.hmc)
        self.hmc.accept = MethodType(lambda self, delta_s: "custom accept", self.hmc)
        self.outer_action.sampleEta = MethodType(lambda self: "custom eta", self.outer_action)
        originals = [(obj, name, getattr(obj, name)) for obj, name in
                     [(self.hmc, "gaussMom"), (self.hmc, "accept"), (self.outer_action, "sampleEta")]]
        handle = self.attach(use_chroma_accept=True)
        handle.detach()
        for obj, name, original in originals:
            self.assertIs(getattr(obj, name), original)

    def test_unattached_objects_and_class_methods_remain_native(self):
        other_action = FakeAction("other")
        other = FakeHMC([other_action])
        original_methods = (FakeHMC.gaussMom, FakeHMC.accept, FakeAction.sampleEta)
        self.attach(use_chroma_accept=True)
        other.gaussMom()
        other_action.sampleEta()
        self.assertEqual(other.accept(1), "native")
        self.assertEqual((other.native_momenta, other_action.native_samples), (1, 1))
        self.assertEqual((FakeHMC.gaussMom, FakeHMC.accept, FakeAction.sampleEta), original_methods)

    def test_dump_stream_forwards_arguments(self):
        handle = self.attach()
        handle.dump_stream("unused-reference-path", 19, "eta")
        self.assertEqual(handle.rng.events, [("dump", "unused-reference-path", 19, "eta")])


if __name__ == "__main__":
    unittest.main()
