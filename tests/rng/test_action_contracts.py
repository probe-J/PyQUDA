"""CPU action 契约回归：AST 加载真实方法，仅替换场和 QUDA kernel，不导入 GPU。

在仓库根目录运行 python -B -m unittest discover -s tests/rng -p test_action_contracts.py。
输入为小型 NumPy 场及故障桩；断言 determinant、步长和状态恢复，不验证力的数值公式。
"""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np


ACTION_DIR = Path(__file__).resolve().parents[2] / "pyquda_core/pyquda/action"


def enum(*names):
    return SimpleNamespace(**{name: name for name in names})


DAG = enum("QUDA_DAG_NO", "QUDA_DAG_YES")
PARITY = enum("QUDA_EVEN_PARITY", "QUDA_ODD_PARITY")
MATPC = enum(
    "QUDA_MATPC_EVEN_EVEN_ASYMMETRIC", "QUDA_MATPC_ODD_ODD_ASYMMETRIC",
    "QUDA_MATPC_EVEN_EVEN", "QUDA_MATPC_ODD_ODD",
)
SOLUTION = enum("QUDA_MATPC_SOLUTION", "QUDA_MATPCDAG_MATPC_SOLUTION")


def load_class(filename, name, namespace, methods=None):
    # 保留真实方法体及 super() 绑定；仅排除模块导入和未选中的方法。
    path = ACTION_DIR / filename
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)
    if methods is not None:
        cls.bases = []
        cls.body = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in methods]
    module = ast.Module(body=[
        ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), cls,
    ], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[name]


class Field:
    """even_ptr 直接暴露 NumPy 半场，便于 kernel 桩验证源和输出。"""

    def __init__(self, values=(0.0, 0.0)):
        self.data = np.array(values, dtype=float)

    @property
    def even(self):
        return self.data[:1]

    @even.setter
    def even(self, value):
        self.data[:1] = value

    @property
    def even_ptr(self):
        return self.even

    @property
    def data_ptrs(self):
        return self

    def norm2(self):
        return float(np.vdot(self.data, self.data))


class Fields(list):
    even_ptrs = property(lambda self: [field.even_ptr for field in self])
    odd_ptrs = property(lambda self: [field.data[1:] for field in self])


class TestActionContracts(unittest.TestCase):
    def setUp(self):
        self.ns = {
            "numpy": np, "FermionAction": object,
            "QudaDagType": DAG, "QudaParity": PARITY, "QudaMatPCType": MATPC,
            "QudaSolutionType": SOLUTION, "QUDA_MAX_MULTI_SHIFT": 16,
            "MatQuda": Mock(), "computeCloverForceV2Quda": Mock(),
            "CloverWilsonDirac": Mock(side_effect=AssertionError("unexpected GPU construction")),
            "nullptr": object(),
        }
        self.cls = load_class("clover_wilson.py", "CloverWilsonAction", self.ns)
        self.base = load_class("abstract.py", "FermionAction", self.ns, {"invertMultiShift"})
        self.obj = self.cls.__new__(self.cls)
        self.obj.invert_param = SimpleNamespace(
            compute_clover_trlog=0, compute_action=0, dagger=DAG.QUDA_DAG_NO,
            solution_type=SOLUTION.QUDA_MATPCDAG_MATPC_SOLUTION,
            cuda_prec_sloppy=4, cuda_prec_precondition=2, tol=1e-8,
            trlogA=[3.0, 7.0], action=[25.0], kappa=0.125, mu=0.0,
        )
        self.obj.gauge_param = SimpleNamespace(use_resident_force=1, return_result_force=0)
        self.obj.latt_info = SimpleNamespace(global_volume=2, Ns=1, Nc=1, GLt=2)
        self.obj.eta, self.obj.phi = Field((2, 0)), Field((4, 0))
        self.obj.quark = Fields([Field(), Field()])
        self.obj.y = Fields([Field(), Field()])
        self.obj.rational_param = SimpleNamespace(
            offset_action=[0.1, 0.2], residue_action=[0.5, 0.25], norm_action=0.25,
        )
        self.obj.dirac = SimpleNamespace(performance=Mock())
        self.obj.updateClover = Mock()
        self.obj.invertMultiShift = self.base.invertMultiShift.__get__(self.obj)
        self.obj.use_action_rational = True
        self.obj.matpc_type = MATPC.QUDA_MATPC_EVEN_EVEN_ASYMMETRIC
        self.obj.active_parity = PARITY.QUDA_EVEN_PARITY
        self.obj.multiplicity = 2
        self.obj.kappa, self.obj.kappa2, self.obj.ck = 0.125, -0.125**2, -0.02
        self.obj.nvector, self.obj.coeff = 2, np.array([0.5, 0.25])
        self.obj._sym_force_helper = None

    def multi_kernel(self, outputs, source, param):
        np.testing.assert_array_equal(source, [4.0])
        # 求解之前 eta 必须仍是原始场，不能被无必要地替换成 phi。
        np.testing.assert_array_equal(self.obj.eta.data, [2.0, 0.0])
        outputs[0][:] = 4.0
        outputs[1][:] = 8.0

    def test_rational_determinants_and_field_restore(self):
        self.ns["invertMultiShiftQuda"] = self.multi_kernel
        for matpc, parity, logdet in (
            (MATPC.QUDA_MATPC_EVEN_EVEN_ASYMMETRIC, PARITY.QUDA_EVEN_PARITY, 14.0),
            (MATPC.QUDA_MATPC_ODD_ODD_ASYMMETRIC, PARITY.QUDA_ODD_PARITY, 6.0),
            (MATPC.QUDA_MATPC_ODD_ODD, PARITY.QUDA_ODD_PARITY, 0.0),
        ):
            with self.subTest(matpc=matpc):
                self.obj.matpc_type, self.obj.active_parity = matpc, parity
                # 重构 eta=0.25*4+0.5*4+0.25*8=5；odd_odd 不重复计入外部 logdet。
                self.assertEqual(self.obj.action(), 25.0 - 1.0 - logdet)
                np.testing.assert_array_equal(self.obj.eta.data, [2.0, 0.0])
                np.testing.assert_array_equal(self.obj.phi.data, [4.0, 0.0])

    def test_rational_restore_on_failures(self):
        for stage in ("clover", "solve", "norm"):
            with self.subTest(stage=stage):
                self.setUp()
                self.ns["invertMultiShiftQuda"] = self.multi_kernel
                param = self.obj.invert_param
                names = ("compute_clover_trlog", "dagger", "solution_type",
                         "cuda_prec_sloppy", "cuda_prec_precondition")
                before = {name: getattr(param, name) for name in names}

                def fail(*args):
                    self.obj.eta.data[:] = 99
                    self.obj.phi.data[:] = 88
                    for name in names:
                        setattr(param, name, "changed")
                    raise RuntimeError(stage)

                if stage == "clover":
                    self.obj.updateClover = fail
                elif stage == "solve":
                    self.ns["invertMultiShiftQuda"] = fail
                else:
                    self.obj.eta.norm2 = fail
                with self.assertRaisesRegex(RuntimeError, stage):
                    self.obj.action()
                self.assertEqual({name: getattr(param, name) for name in names}, before)
                np.testing.assert_array_equal(self.obj.eta.data, [2, 0])
                np.testing.assert_array_equal(self.obj.phi.data, [4, 0])

    def test_single_pole_restores_actual_solver_state(self):
        self.obj.rational_param.offset_action = [0.0]
        self.obj.rational_param.residue_action = [1.0]
        self.obj.invert_param.dagger = DAG.QUDA_DAG_YES
        before = vars(self.obj.invert_param).copy()

        def solve(output, source, param):
            self.assertEqual(param.solution_type, SOLUTION.QUDA_MATPC_SOLUTION)
            np.testing.assert_array_equal(source, [4.0])
            output[:] = 5.0

        self.ns["invertQuda"] = solve
        self.assertEqual(self.obj.action(), 10.0)
        self.assertEqual(vars(self.obj.invert_param), before)
        self.ns["invertQuda"] = Mock(side_effect=RuntimeError("single solve"))
        with self.assertRaisesRegex(RuntimeError, "single solve"):
            self.obj.action()
        self.assertEqual(vars(self.obj.invert_param), before)
        np.testing.assert_array_equal(self.obj.eta.data, [2, 0])

    def test_unsupported_symmetric_modes_fail_before_gpu(self):
        for matpc, rational in (("even_even", False), ("even_even", True), ("odd_odd", False)):
            with self.subTest(matpc=matpc, rational=rational):
                with self.assertRaises(NotImplementedError):
                    self.cls(None, None, 0.1, 2, 1e-8, 100, 1.0, rational, matpc)
        self.ns["CloverWilsonDirac"].assert_not_called()

    def make_helper(self):
        # 执行真实 symmetric 展开及 pair 累加，只有底层场算子和 kernel 使用桩。
        helper_cls = load_class("clover_hasenbusch.py", "CloverWilsonHasenbuschTwistAction", self.ns, {
            "_add_sym_oddodd_chroma_deriv_force", "_add_pair_force_v2",
            "_add_symmetric_shift_deriv_force",
        })
        helper = helper_cls()
        helper._active_parity = lambda: PARITY.QUDA_ODD_PARITY
        helper._active_pc_field = lambda field: field
        helper._parity_field = lambda field, parity: field
        helper._dslash_block = lambda *args: None
        helper._clover_inv_parity = lambda *args: None
        for index in range(1, 6):
            setattr(helper, f"pc_tmp{index}", Field())
        helper.pair_x, helper.pair_p = Fields([Field()]), Fields([Field()])
        helper.force_tmp = Field()
        helper.nvector, helper.sym_odd_clover_scale = 1, 1.0
        helper.force_sign, helper.force_coeff_scale = 1.0, 1.0
        helper.label, helper.force_debug_logging = "stub", False
        helper.force_term_history, helper.coeff = [], np.array([1.0])
        # 固定非零 kernel 输出让步长缩放与符号抵消无关；不冒充数值验证。
        self.ns["computeCloverPairForceV2Quda"] = lambda out, *args: setattr(out, "data", np.ones(2))
        self.obj._sym_force_helper = helper
        self.obj.invertMultiShift = Mock()
        return helper

    def test_symmetric_dt_scales_only_new_contributions(self):
        helper = self.make_helper()
        for dt in (0.0, 0.25, -0.5, 1.0):
            with self.subTest(dt=dt):
                force = Field((11.0, 13.0))
                self.obj.force_v2(dt, force)
                # 两极点各有 PLUS/MINUS 两组、每组四个 pair 项。
                np.testing.assert_allclose(force.data, np.array([11, 13]) + 16 * (-4 * 0.125) * dt)
                self.assertEqual(helper.force_output_scale, -4 * 0.125 * dt)
        self.ns["computeCloverForceV2Quda"].assert_not_called()

    def test_symmetric_force_restores_saved_flags(self):
        for stage in ("success", "mat", "pair"):
            with self.subTest(stage=stage):
                self.setUp()
                self.make_helper()
                self.obj.invert_param.dagger = DAG.QUDA_DAG_YES
                self.obj.gauge_param.use_resident_force = 0
                self.obj.gauge_param.return_result_force = 1
                before = vars(self.obj.gauge_param).copy()
                if stage != "success":
                    name = "MatQuda" if stage == "mat" else "computeCloverPairForceV2Quda"
                    self.ns[name] = Mock(side_effect=RuntimeError(stage))
                    with self.assertRaisesRegex(RuntimeError, stage):
                        self.obj.force_v2(0.25, Field())
                else:
                    self.obj.force_v2(0.25, Field())
                self.assertEqual(vars(self.obj.gauge_param), before)
                self.assertEqual(self.obj.invert_param.dagger, DAG.QUDA_DAG_YES)

    def test_symmetric_resident_force_fails_before_calls(self):
        self.make_helper()
        with self.assertRaisesRegex(NotImplementedError, "explicit force buffer"):
            self.obj.force_v2(1.0, None)
        self.obj.updateClover.assert_not_called()
        self.obj.invertMultiShift.assert_not_called()
        self.ns["MatQuda"].assert_not_called()
        self.ns["computeCloverForceV2Quda"].assert_not_called()

    def test_symmetric_momentum_force_fails_before_calls(self):
        self.make_helper()
        self.ns["computeCloverForceQuda"] = Mock()
        for mom in (None, Field()):
            with self.subTest(resident=mom is None):
                with self.assertRaisesRegex(NotImplementedError, "explicit force buffer"):
                    self.obj.force(1.0, mom)
        self.obj.updateClover.assert_not_called()
        self.obj.invertMultiShift.assert_not_called()
        self.ns["MatQuda"].assert_not_called()
        self.ns["computeCloverForceQuda"].assert_not_called()

    def test_default_asymmetric_native_behavior(self):
        self.obj.use_action_rational = False
        self.obj.invertMultiShift = Mock()
        self.assertEqual(self.obj.action(), 10.0)
        self.obj.invertMultiShift.assert_called_once_with("force")
        self.assertEqual(self.obj.invert_param.compute_action, 0)
        for force in (None, Field((11, 13))):
            with self.subTest(resident=force is None):
                self.obj.force_v2(0.25, force)
                args = self.ns["computeCloverForceV2Quda"].call_args.args
                self.assertIs(args[0], self.ns["nullptr"] if force is None else force)
                self.assertEqual(args[1], 0.25)
                self.assertIs(args[3], self.obj.coeff)
                self.assertEqual(args[4:8], (self.obj.kappa2, self.obj.ck, 2, 2))
                self.assertEqual(self.obj.invert_param.dagger, DAG.QUDA_DAG_NO)
                self.assertEqual(vars(self.obj.gauge_param), {"use_resident_force": 1, "return_result_force": 0})


if __name__ == "__main__":
    unittest.main()
