"""无 CUDA 的接口回归：执行源码中的 pair-force 方法，检查倍率而非证明数值力正确。"""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np


class Pair:
    even_ptrs = None
    odd_ptrs = None

    def __setitem__(self, index, value):
        self.value = value


class PairForceTests(unittest.TestCase):
    def call(self, force_sign, force_coeff_scale, include_hopping=True, include_clover=True):
        path = Path(__file__).resolve().parents[2] / "pyquda_core/pyquda/action/clover_hasenbusch.py"
        tree = ast.parse(path.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "CloverWilsonHasenbuschTwistAction")
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_add_pair_force_v2")
        # 仅去掉类型注解，函数体原样执行；记录实际传给 kernel 的系数。
        for arg in method.args.args:
            arg.annotation = None
        method.returns = None
        captures = []
        def kernel(*args):
            captures.append(tuple(a.copy() for a in args[5:8]))
        namespace = {"numpy": np, "computeCloverPairForceV2Quda": kernel}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), str(path), "exec"), namespace)
        obj = SimpleNamespace(pair_x=Pair(), pair_p=Pair(), force_tmp=SimpleNamespace(data=0, data_ptrs=None),
                              invert_param=SimpleNamespace(kappa=0.125), ck=-0.01, force_sign=force_sign,
                              force_coeff_scale=force_coeff_scale, label="test", force_term_filter=None,
                              force_debug_logging=False, nvector=1, gauge_param=None, coeff=np.array([1.0]),
                              force_output_scale=1, force_term_history=[])
        namespace[method.name](obj, SimpleNamespace(data=0), object(), object(), 2, "term",
                               include_hopping=include_hopping, include_clover=include_clover)
        return captures[0]

    def test_pair_controls_not_silently_dropped(self):
        baseline = self.call(1, 1)
        for sign, scale in ((-1, 1), (1, 0), (1, 2.5)):
            actual = self.call(sign, scale)
            for a, b in zip(actual, baseline):
                np.testing.assert_allclose(a, b * sign * scale, atol=0)

    def test_pair_hopping_clover_switches(self):
        hopping = self.call(1, 1, include_clover=False)
        np.testing.assert_array_equal(hopping[1:], 0)
        clover = self.call(1, 1, include_hopping=False)
        np.testing.assert_array_equal(clover[0], 0)


if __name__ == "__main__":
    unittest.main()
