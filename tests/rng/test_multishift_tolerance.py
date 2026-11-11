"""CPU 检查 strange 多极点的残差策略与源/目标场；不依赖 CUDA。

使用 AST 执行真实方法，kernel 桩只写固定解，验证原生默认策略不变，
Chroma 显式 uniform 策略对每个极点使用相同容差且不修改求解精度。
"""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

from test_action_contracts import Field, Fields, load_class


class TestMultishiftTolerance(unittest.TestCase):
    def setUp(self):
        self.delegate = Mock(return_value="native")
        base = type("FermionAction", (), {"invertMultiShift": self.delegate})
        self.kernel = Mock(side_effect=self.solve)
        self.ns = dict(FermionAction=base, invertMultiShiftQuda=self.kernel, QUDA_MAX_MULTI_SHIFT=16)
        cls = load_class("clover_wilson.py", "CloverWilsonAction", self.ns)
        self.action = cls.__new__(cls)
        self.action.multishift_tolerance = "uniform"
        self.action.invert_param = SimpleNamespace(tol=1e-9, cuda_prec_sloppy=8, cuda_prec_precondition=8)
        self.action.dirac = SimpleNamespace(performance=Mock())
        self.action.eta, self.action.phi = Field((2, 0)), Field((4, 0))
        self.action.quark = Fields([Field(), Field()])
        self.action.rational_param = SimpleNamespace(**{
            prefix + "_" + mode: values
            for mode in ("sample", "action", "force")
            for prefix, values in (("offset", [.1, .2]), ("residue", [.5, .25]), ("norm", .125))
        })

    def solve(self, outputs, source, param):
        self.source = source.copy()
        outputs[0][:] = 8
        outputs[1][:] = 12

    def test_uniform_source_reconstruction_and_precision(self):
        for mode in ("sample", "action", "force"):
            with self.subTest(mode=mode):
                self.setUp()
                self.action.invertMultiShift(mode)
                p = self.action.invert_param
                self.assertEqual(p.num_offset, 2)
                self.assertEqual(p.tol_offset, [1e-9, 1e-9] + [0.] * 14)
                self.assertEqual(p.offset, [.1, .2] + [0.] * 14)
                self.assertEqual(p.residue, [.5, .25] + [0.] * 14)
                self.assertEqual((p.cuda_prec_sloppy, p.cuda_prec_precondition), (8, 8))
                np.testing.assert_array_equal(self.source, [2 if mode == "sample" else 4])
                if mode == "sample":
                    np.testing.assert_array_equal(self.action.phi.even, [7.25])
                    np.testing.assert_array_equal(self.action.eta.even, [2])
                elif mode == "action":
                    np.testing.assert_array_equal(self.action.eta.even, [7.5])
                    np.testing.assert_array_equal(self.action.phi.even, [4])
                else:
                    np.testing.assert_array_equal(self.action.eta.even, [2])
                    np.testing.assert_array_equal(self.action.phi.even, [4])
                self.delegate.assert_not_called()

    def test_native_default_and_single_pole_delegate(self):
        del self.action.multishift_tolerance
        self.assertEqual(self.action.invertMultiShift("force"), "native")
        self.kernel.assert_not_called()
        self.action.multishift_tolerance = "uniform"
        self.action.rational_param.offset_force = [0.]
        self.action.rational_param.residue_force = [1.]
        self.assertEqual(self.action.invertMultiShift("force"), "native")
        self.assertEqual(self.delegate.call_count, 2)

    def test_invalid_poles_rejected_before_kernel(self):
        for offsets, residues in (([], []), ([0.], [1., 2.]), ([0.] * 17, [1.] * 17)):
            self.action.rational_param.offset_force = offsets
            self.action.rational_param.residue_force = residues
            with self.assertRaises(ValueError):
                self.action.invertMultiShift("force")
        with self.assertRaises(ValueError):
            self.action.invertMultiShift("unknown")
        self.kernel.assert_not_called()
        self.delegate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
