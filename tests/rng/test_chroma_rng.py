"""CPU 回归：LCG 状态、精度、全局坐标映射及 QUDA 动量打包，不需要 GPU。

从源码路径加载模块，避免触发 PyQUDA 的 CUDA 初始化。
"""

import importlib.util
import math
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np

SOURCE = Path(__file__).resolve().parents[2] / "pyquda_core/pyquda/chroma_rng.py"
spec = importlib.util.spec_from_file_location("rng_under_test", SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
ChromaRNG = module.ChromaRNG


class FakeLattice:
    """仅提供场注入所需的本地 checkerboard 全局坐标，支持沿 t 分区。"""

    def __init__(self, size=(4, 2, 2, 2), t_start=0, t_stop=2):
        self.size = size
        self._coords = np.array([
            [(x, y, z, t) for t in range(t_start, t_stop)
             for z in range(size[2]) for y in range(size[1]) for x in range(size[0])
             if (x + y + z + t) % 2 == parity]
            for parity in (0, 1)
        ]).transpose(2, 0, 1)

    def coordinate(self):
        return self._coords


class FakeField:
    def __init__(self, li):
        self.latt_info = li
        self.data = np.zeros((2, li.coordinate().shape[-1], 4, 3), complex)


class FakeMom:
    def __init__(self, li):
        self.data = np.zeros((4, 2, li.coordinate().shape[-1], 10))


def unpack_momentum(packed):
    result = np.zeros(packed.shape[:-1] + (3, 3), complex)
    for index, (row, col) in enumerate(((0, 1), (0, 2), (1, 2))):
        result[..., row, col] = packed[..., 2 * index] + 1j * packed[..., 2 * index + 1]
        result[..., col, row] = -result[..., row, col].conj()
    for index in range(3):
        result[..., index, index] = 1j * packed[..., index + 6]
    return result


class TestChromaRNG(unittest.TestCase):
    def test_scalar_golden_and_state(self):
        # 固定小向量是算法回归值，不冒充外部 Chroma 实测数据。
        rng = ChromaRNG((4, 2, 2, 2))
        rng.set_seed(12345)
        expected = np.array([0.21551288664340973, 0.028680937364697456,
                             0.40981489419937134, 0.4954339563846588,
                             0.23404307663440704], dtype=np.float32)
        actual = np.array([rng.random_scalar() for _ in range(5)], dtype=np.float32)
        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
        self.assertEqual(rng.get_seed(), 12345 * pow(module._MULT_A, 5, 2**47) % 2**47)
        saved = rng.save()
        first = rng.random_scalar()
        self.assertEqual(rng.restore(), saved)
        self.assertEqual(rng.random_scalar(), first)

    def test_field_uniforms_coordinate_formula(self):
        rng = ChromaRNG((4, 2, 2, 2))
        rng.set_seed(20260910)
        pos = rng._parity_pos[1]
        actual = rng._subset_uniforms(pos, 12)
        expected = np.empty_like(actual)
        for i, coord in enumerate(rng.coords[:, pos].T):
            lex = module.lex_exponent(coord, rng.size)
            for k in range(24):
                state = 20260910 * pow(module._MULT_A, lex + k * (rng.volume + 1), 2**47) % 2**47
                expected[i, k] = state / 2**47
        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
        self.assertEqual(rng.get_seed(), 20260910 * pow(module._MULT_A, 24 * 33, 2**47) % 2**47)

    def test_gaussian_precision(self):
        r1 = np.array([[0.13, 0.22, 0.37, 0.49]], dtype=np.float32)
        r2 = np.array([[0.23, 0.18, 0.65, 0.31]], dtype=np.float32)
        radius = np.sqrt(-2 * np.log(r1[:, ::2].astype(float)))
        angle = r2[:, ::2].astype(float) * 6.283185307
        expected = radius * (np.cos(angle) + 1j * np.sin(angle))
        double = ChromaRNG._complex_from_uniforms(r1, r2, 2, "double")
        np.testing.assert_allclose(double, expected, rtol=0, atol=1e-14)
        radius32 = np.sqrt(-np.float32(2) * np.log(r1[:, ::2]))
        angle32 = r2[:, ::2] * np.float32(6.283185307)
        single = ChromaRNG._complex_from_uniforms(r1, r2, 2, "single")
        np.testing.assert_array_equal(single.real, radius32 * np.cos(angle32))
        np.testing.assert_array_equal(single.imag, radius32 * np.sin(angle32))
        self.assertGreater(np.max(np.abs(single - double)), 1e-9)

    def test_eta_partition_and_normalization(self):
        expected_rng = ChromaRNG((4, 2, 2, 2))
        expected_rng.set_seed(12345)
        values, positions = expected_rng.gaussian_fermion_subset(1)
        expected = {tuple(expected_rng.coords[:, p]): values[i] / math.sqrt(2)
                    for i, p in enumerate(positions)}
        with patch.dict(sys.modules, {"cupy": types.SimpleNamespace(asarray=np.asarray)}):
            for start, stop in ((0, 2), (0, 1), (1, 2)):
                li = FakeLattice(t_start=start, t_stop=stop)
                field = FakeField(li)
                rng = ChromaRNG(li.size)
                rng.set_seed(12345)
                rng.fill_eta(field, 1)
                self.assertEqual(rng.get_seed(), expected_rng.get_seed())
                np.testing.assert_array_equal(field.data[0], 0)
                for index, coord in enumerate(li.coordinate()[:, 1].T):
                    np.testing.assert_allclose(field.data[1, index], expected[tuple(coord)], atol=1e-14)

    def test_momentum_projection_pack_partition(self):
        ref = ChromaRNG((4, 2, 2, 2))
        ref.set_seed(12345)
        matrices, _ = ref.gaussian_momentum_matrices()
        np.testing.assert_allclose(matrices + matrices.swapaxes(-1, -2).conj(), 0, atol=1e-14)
        np.testing.assert_allclose(np.trace(matrices, axis1=-2, axis2=-1), 0, atol=1e-14)
        lookup = {tuple(coord): i for i, coord in enumerate(ref.coords.T)}
        fake = {"cupy": types.SimpleNamespace(asarray=np.asarray),
                "pyquda.field": types.SimpleNamespace(LatticeMom=FakeMom)}
        with patch.dict(sys.modules, fake):
            for start, stop in ((0, 2), (0, 1), (1, 2)):
                li = FakeLattice(t_start=start, t_stop=stop)
                rng = ChromaRNG(li.size)
                rng.set_seed(12345)
                packed = rng.make_momentum(li).data
                np.testing.assert_array_equal(packed[..., 9], 0)
                unpacked = unpack_momentum(packed)
                for parity in (0, 1):
                    for index, coord in enumerate(li.coordinate()[:, parity].T):
                        np.testing.assert_allclose(unpacked[:, parity, index], matrices[:, lookup[tuple(coord)]], atol=1e-14)
        chroma = ChromaRNG(ref.size)
        chroma.set_seed(12345)
        scaled, _ = chroma.gaussian_momentum_matrices(True)
        np.testing.assert_allclose(scaled, matrices / math.sqrt(2), atol=1e-14)

    def test_validation_and_selftest(self):
        for size in ((1, 2), (0, 2, 2, 2), (2.5, 2, 2, 2)):
            with self.assertRaises(ValueError):
                ChromaRNG(size)
        with self.assertRaises(ValueError):
            ChromaRNG((2, 2, 2, 2), "half")
        rng = ChromaRNG((2, 2, 2, 2))
        for seed in (-1, 2**47, 1.2):
            with self.assertRaises(ValueError):
                rng.set_seed(seed)
        module.selftest()


if __name__ == "__main__":
    unittest.main()
