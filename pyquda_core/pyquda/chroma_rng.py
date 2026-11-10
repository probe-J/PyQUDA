"""Chroma QDP++ 47-bit LCG 随机数机制的 PyQUDA 移植（仅测试用，独立模块）。

来源（Chroma 所依赖的 QDP++/QDP-JIT；版本及验证限制见 tests/rng/README.md）：
- lib/qdp_random.cc            LCG 核心 / initRNG / setrn / savern / sranf
- include/qdp_primseed.h       种子卷积乘法(=mod 2^47)、seedToFloat
- include/qdp_reality.h        fill_random(RComplex 递归)、fill_gaussian
- include/qdp_parscalar_specific.h  场级 random/gaussian host 参考实现
- include/qdp_scalar_specific.h:274 标量 random

语义约定（务必保持）：
- 种子为 47-bit 整数；乘子 a 的 12-bit 字 = (1645, 712, 1498, 1222)
- ran_mult_n = a^(V+1)，V = 全局格点数
- 格点乘子 = a^(lex)，lex = 1 + x + Lx*(y + Ly*(z + Lz*t))，x 最快，全局坐标
- 场调用 random(field, subset)：每格点 skew = S * a^lex(i)，第 k 个 float 分量
  取 seedToFloat(skew) 后 skew *= a^(V+1)；调用结束后 S *= a^(E*(V+1))，
  E = 每格点 float 分量数（Fermion=24, ColorMatrix=18）
- 标量 random：u = seedToFloat(S*a)，S *= a
- Box-Muller 常数 6.283185307（截断十进制，非 math.tau）；复数元只使用
  r1.re（半径）与 r2.re（相角）；分量顺序 = (re, im)
- 均匀数永远经 float32；BM 中间精度由 precision 决定（默认 double）

本模块不依赖 pyquda 内部实现，只依赖 numpy；对 pyquda 场的写入由
fill_eta / fill_momentum 完成（布局约定见函数 docstring）。
"""

from typing import List, Optional, Sequence, Tuple

import math

import numpy

_MOD = 1 << 47
_MULT_A = 1645 + (712 << 12) + (1498 << 24) + (1222 << 36)
_TWOM11 = 1.0 / 2048.0
_TWOM12 = 1.0 / 4096.0
_TWO_PI_CHROMA = 6.283185307

_FERMION_E = 24  # 4 spin * 3 color * (re, im)
_GAUGEMAT_E = 18  # 3*3 complex * (re, im)


def seed_mul(x: int, y: int) -> int:
    return (x * y) % _MOD


def seed_to_float(s: int) -> float:
    """qdp_primseed.h seedToFloat，中间用 float64（double build）。"""
    w0 = s & 4095
    w1 = (s >> 12) & 4095
    w2 = (s >> 24) & 4095
    w3 = (s >> 36) & 2047
    d = _TWOM12 * w0
    d = _TWOM12 * (w1 + d)
    d = _TWOM12 * (w2 + d)
    d = _TWOM11 * (w3 + d)
    return d


def seed_to_float_single(s: int) -> float:
    """单精度 build 的近似：中间量按 float32 舍入。"""
    w0 = numpy.float32(s & 4095)
    w1 = numpy.float32((s >> 12) & 4095)
    w2 = numpy.float32((s >> 24) & 4095)
    w3 = numpy.float32((s >> 36) & 2047)
    d = numpy.float32(_TWOM12) * w0
    d = numpy.float32(_TWOM12) * (w1 + d)
    d = numpy.float32(_TWOM12) * (w2 + d)
    d = numpy.float32(_TWOM11) * (w3 + d)
    return float(d)


def a_pow(n: int) -> int:
    return pow(_MULT_A, n, _MOD)


def lex_exponent(coord: Sequence[int], latt_size: Sequence[int]) -> int:
    """chroma initRNG 的 lexoc：1 + x + Lx*(y + Ly*(z + Lz*t))，coord=(x,y,z,t)。"""
    x, y, z, t = (int(c) for c in coord)
    lx, ly, lz, _lt = (int(v) for v in latt_size)
    return 1 + x + lx * (y + ly * (z + lz * t))


class ChromaRNG:
    """Chroma 全局 LCG 流的 Python 重放器（每 rank 独立重放同一全局流）。"""

    def __init__(self, global_latt_size: Sequence[int], precision: str = "double") -> None:
        if len(global_latt_size) != 4 or any(int(v) != v or v <= 0 for v in global_latt_size):
            raise ValueError("global_latt_size must contain four positive integers")
        self.size: Tuple[int, int, int, int] = tuple(int(v) for v in global_latt_size)
        lx, ly, lz, lt = self.size
        self.volume = lx * ly * lz * lt
        if precision not in ("double", "single"):
            raise ValueError("precision must be double or single")
        self.precision = precision
        self._s2f = seed_to_float if precision == "double" else seed_to_float_single
        self.mult_n = a_pow(self.volume + 1)
        # 每格点乘子表：mult_of_lex[e] = a^e mod 2^47，e = 0..volume
        max_lex = self.volume  # lex 最大 = V（lex=1+... ≤ V）
        mult_list = [1] * (max_lex + 1)
        g = 1
        for e in range(max_lex + 1):
            mult_list[e] = g
            g = seed_mul(g, _MULT_A)
        self._mult_of_lex = mult_list
        # 内部表按 meshgrid 的 C 序展平（t 最快）；随机数指数仍使用全局 x-fast lex。
        # 场写入始终按坐标映射，不能直接把此表当成 QUDA checkerboard 顺序。
        xs = numpy.arange(lx)
        ys = numpy.arange(ly)
        zs = numpy.arange(lz)
        ts = numpy.arange(lt)
        gx, gy, gz, gt = numpy.meshgrid(xs, ys, zs, ts, indexing="ij")
        self.coords = numpy.stack(
            [gx.reshape(-1), gy.reshape(-1), gz.reshape(-1), gt.reshape(-1)]
        )  # (4, V)，内部坐标表下标
        parity = (gx + gy + gz + gt) % 2
        self.parity = parity.reshape(-1)
        self._parity_pos = {
            0: numpy.nonzero(self.parity == 0)[0],
            1: numpy.nonzero(self.parity == 1)[0],
        }
        self.seed = 11
        self._saved: List[int] = []

    # ---------- 流状态 ----------
    def set_seed(self, seed: int) -> None:
        if int(seed) != seed or not 0 <= seed < _MOD:
            raise ValueError("seed must be an integer in [0, 2**47)")
        self.seed = int(seed) % _MOD

    def get_seed(self) -> int:
        return self.seed

    def save(self) -> int:
        self._saved.append(self.seed)
        return self.seed

    def restore(self) -> int:
        self.seed = self._saved.pop()
        return self.seed

    # ---------- 消耗原语 ----------
    def random_scalar(self) -> float:
        """chroma random(OScalar)：u = f(S*a)，S *= a。"""
        s = seed_mul(self.seed, _MULT_A)
        u = float(numpy.float32(self._s2f(s)))
        self.seed = s
        return u

    def _subset_uniforms(self, site_pos: numpy.ndarray, ncomp: int) -> numpy.ndarray:
        """对 canonical 序中的 site_pos 子集生成 (n_sites, 2*ncomp) 均匀数。

        chroma 复场 fill：每个复分量先 re 后 im（E = 2*ncomp），输出只用每个
        复分量的 re 槽（见 fill_gaussian），但 im 槽同样消耗流。这里返回全部
        E 列，Gaussian 变换自行挑选偶数列。
        """
        e_total = 2 * ncomp
        n_sites = int(site_pos.shape[0])
        out = numpy.empty((n_sites, e_total), dtype="<f4")
        lexes = self.coords[:, site_pos]
        lexes = 1 + lexes[0] + self.size[0] * (
            lexes[1] + self.size[1] * (lexes[2] + self.size[2] * lexes[3])
        )
        mult_table = self._mult_of_lex
        s_list = [seed_mul(self.seed, mult_table[int(e)]) for e in lexes]
        mn = self.mult_n
        for k in range(e_total):
            out[:, k] = [self._s2f(s) for s in s_list]
            s_list = [seed_mul(s, mn) for s in s_list]
        # Chroma 的场级 random 在每个浮点分量之间跨越 V+1 个 seed 槽位，
        # 因此一次完整场调用必须推进 E*(V+1)，而不是只推进 E。
        self.seed = seed_mul(self.seed, a_pow(e_total * (self.volume + 1)))
        return out

    # ---------- 高斯层 ----------
    @staticmethod
    def _complex_from_uniforms(
        r1: numpy.ndarray, r2: numpy.ndarray, ncomp: int, precision: str
    ) -> numpy.ndarray:
        """fill_gaussian(RComplex)：复分量 c 使用 r1[:,2c]（半径）、r2[:,2c]（相角）。

        返回 (n_sites, ncomp) complex128（double build）。
        """
        r1_sel = r1[:, 0 : 2 * ncomp : 2].astype(numpy.float64)
        r2_sel = r2[:, 0 : 2 * ncomp : 2].astype(numpy.float64)
        angle = r2_sel * _TWO_PI_CHROMA
        radius = numpy.sqrt(-2.0 * numpy.log(r1_sel))
        re = radius * numpy.cos(angle)
        im = radius * numpy.sin(angle)
        if precision == "single":
            # single build：中间按 float32
            angle32 = (r2[:, 0 : 2 * ncomp : 2] * numpy.float32(_TWO_PI_CHROMA)).astype(numpy.float32)
            radius32 = numpy.sqrt(-numpy.float32(2.0) * numpy.log(r1[:, 0 : 2 * ncomp : 2]))
            re = (radius32 * numpy.cos(angle32)).astype(numpy.float64)
            im = (radius32 * numpy.sin(angle32)).astype(numpy.float64)
        return re + 1j * im

    def gaussian_fermion_subset(self, parity: int) -> Tuple[numpy.ndarray, numpy.ndarray]:
        """chroma gaussian(eta, rb[p])。

        返回 (values, site_pos)：
        - values: (n_subset, Ns*Nc) complex128，分量顺序 spin 外层、color 内层
        - site_pos: self.coords 内部坐标表的下标，不等同于 lex 下标
        """
        site_pos = self._parity_pos[parity]
        r1 = self._subset_uniforms(site_pos, _FERMION_E // 2)
        r2 = self._subset_uniforms(site_pos, _FERMION_E // 2)
        values = self._complex_from_uniforms(r1, r2, _FERMION_E // 2, self.precision)
        return values.reshape(site_pos.shape[0], 4, 3), site_pos

    def gaussian_momentum_matrices(self, apply_chroma_sqrt_half: bool = False) -> Tuple[numpy.ndarray, numpy.ndarray]:
        """chroma refreshP 的 gaussian 部分：对 mu=0..3 各一次全格调用。

        返回 (momentum, site_pos)：
        - momentum: (Nd, V, 3, 3) complex128，已做 taproj。
          apply_chroma_sqrt_half=True 时输出 Chroma B0 dump 口径；
          False 时输出 QUDA LatticeMom 注入口径。
        - site_pos: canonical 全格序
        """
        lx, ly, lz, lt = self.size
        v = self.volume
        momentum = numpy.empty((4, v, 3, 3), dtype=numpy.complex128)
        site_pos = numpy.arange(v)
        for mu in range(4):
            r1 = self._subset_uniforms(site_pos, _GAUGEMAT_E // 2)
            r2 = self._subset_uniforms(site_pos, _GAUGEMAT_E // 2)
            m = self._complex_from_uniforms(r1, r2, _GAUGEMAT_E // 2, self.precision).reshape(v, 3, 3)
            if apply_chroma_sqrt_half:
                # Chroma refreshP 在 gaussian 后先乘 sqrt(0.5)，再做 taproj。
                m = m * math.sqrt(0.5)
            # QDP++ taproj：先取反厄米部分，再去除反厄米迹，得到 su(3) 动量。
            m = 0.5 * (m - numpy.conjugate(numpy.swapaxes(m, 1, 2)))
            trace = (m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]) / 3.0
            for c in range(3):
                m[:, c, c] -= trace
            momentum[mu] = m
        return momentum, site_pos

    # ---------- PyQUDA 场写入 ----------
    def fill_eta(self, field, parity: int) -> None:
        """把 gaussian(eta, rb[parity]) 写入 pyquda LatticeFermion（支持多卡）。

        field.data 为复数，形状 (2, Lt, Lz, Ly, Lx//2, Ns, Nc)；slot=site parity；
        slot 内存储序 = (t,z,y,x/2) 字典序 = canonical 序过滤该 parity。
        通过 latt_info.coordinate()（全局坐标）建立本地存储序 → canonical 序
        映射，因此与 MPI 分解方式无关。另一 parity 半场置零。
        """
        values, site_pos = self.gaussian_fermion_subset(parity)
        # chroma refreshInternalFields 在 gaussian 后对 eta 乘 sqrt(0.5)；
        # pyquda 原生 sampleEta 的分布也是方差 1/2，这里对齐该约定。
        values = values * math.sqrt(0.5)
        field.data[:] = 0
        li = field.latt_info
        # RNG 值按 site_pos 坐标顺序生成；PyQUDA field 按 checkerboard 存储
        # 顺序排列。必须按全局坐标映射，不能把两种顺序直接互换。
        value_by_coord = {
            tuple(int(v) for v in self.coords[:, pos]): values[index]
            for index, pos in enumerate(site_pos)
        }
        coords = li.coordinate()
        coords = coords.get() if hasattr(coords, "get") else coords
        slot = int(parity)
        view = field.data[slot].reshape(-1, 4, 3)
        flat_coords = coords[:, slot].reshape(4, -1)
        import cupy
        for index in range(flat_coords.shape[1]):
            coord = tuple(int(v) for v in flat_coords[:, index])
            view[index] = cupy.asarray(value_by_coord[coord].reshape(4, 3))

    def make_momentum(self, latt_info):
        """生成 pyquda LatticeMom（chroma 流动量，反厄米投影到 QUDA 布局）。"""
        from pyquda.field import LatticeMom
        import cupy

        # QUDA momAction/HMC 使用未乘 sqrt(0.5) 的 QUDA 动量约定；
        # direct Chroma dump 才使用 apply_chroma_sqrt_half=True。
        momentum, site_pos = self.gaussian_momentum_matrices()
        mom = LatticeMom(latt_info)
        # 本 rank 场内存储序 → 全局 canonical 序映射
        # momentum 同样按全局坐标映射，避免 site_pos 顺序与 field 存储顺序混用。
        coords = latt_info.coordinate()
        coords = coords.get() if hasattr(coords, "get") else coords
        for mu in range(4):
            m = momentum[mu]
            momentum_by_coord = {
                tuple(int(v) for v in self.coords[:, index]): m[index]
                for index in range(self.volume)
            }
            for slot in (0, 1):
                flat_coords = coords[:, slot].reshape(4, -1)
                mm = numpy.empty((flat_coords.shape[1], 3, 3), dtype=m.dtype)
                for index in range(flat_coords.shape[1]):
                    coord = tuple(int(v) for v in flat_coords[:, index])
                    mm[index] = momentum_by_coord[coord]
                anti = 0.5 * (mm - numpy.conj(numpy.swapaxes(mm, 1, 2)))
                # QUDA 的 LatticeMom 每格 10 个实数：三条上三角复元、
                # 三个对角虚部和一个 padding 槽；不能按 9 个复数展开。
                view = mom.data[mu, slot].reshape(-1, 10)
                view[:, 0] = cupy.asarray(anti[:, 0, 1].real)
                view[:, 1] = cupy.asarray(anti[:, 0, 1].imag)
                view[:, 2] = cupy.asarray(anti[:, 0, 2].real)
                view[:, 3] = cupy.asarray(anti[:, 0, 2].imag)
                view[:, 4] = cupy.asarray(anti[:, 1, 2].real)
                view[:, 5] = cupy.asarray(anti[:, 1, 2].imag)
                view[:, 6] = cupy.asarray(anti[:, 0, 0].imag)
                view[:, 7] = cupy.asarray(anti[:, 1, 1].imag)
                view[:, 8] = cupy.asarray(anti[:, 2, 2].imag)
                view[:, 9] = 0.0
        return mom

    # ---------- 参考 dump（与 chroma_rng_dump.cc 同格式） ----------
    def dump_reference(self, path: str, seed: int, mode: str) -> None:
        """mode='eta'（odd subset，未乘 sqrt(0.5)）/ 'mom'（全格，含 taproj）。

        格式：首行 '#' JSON 头；随后每行: x y z t <值...>（float64 文本）。
        """
        import json

        lx, ly, lz, lt = self.size
        self.set_seed(seed)
        if mode == "eta":
            values, site_pos = self.gaussian_fermion_subset(1)
            values = values.reshape(site_pos.shape[0], -1)
        elif mode == "mom":
            momentum, site_pos = self.gaussian_momentum_matrices(apply_chroma_sqrt_half=True)
        elif mode == "uniforms":
            site_pos = self._parity_pos[1]
            values = self._subset_uniforms(site_pos, _FERMION_E // 2).astype(numpy.float64)
        else:
            raise ValueError(mode)
        with open(path, "w") as f:
            header = {
                "format": "chroma_rng_dump_v1",
                "latt": [lx, ly, lz, lt],
                "seed": int(seed),
                "mode": mode,
                "precision": self.precision,
            }
            f.write("# " + json.dumps(header) + "\n")
            coords = self.coords[:, site_pos]
            for i in range(site_pos.shape[0]):
                x, y, z, t = (int(v) for v in coords[:, i])
                if mode == "eta":
                    vals = numpy.column_stack((values[i].real, values[i].imag)).reshape(-1)
                    f.write(f"{x} {y} {z} {t} " + " ".join(f"{v:.17g}" for v in vals) + "\n")
                elif mode == "uniforms":
                    f.write(f"{x} {y} {z} {t} " + " ".join(f"{v:.9g}" for v in values[i]) + "\n")
                else:
                    # 与 Chroma 工具一致：mu 外层，3x3 矩阵行优先，re/im 交错。
                    vals = numpy.column_stack(
                        (momentum[:, i].reshape(-1).real, momentum[:, i].reshape(-1).imag)
                    ).reshape(-1)
                    f.write(f"{x} {y} {z} {t} " + " ".join(f"{v:.17g}" for v in vals) + "\n")


def selftest() -> None:
    """无 GPU 依赖的流自检：确定性、[0,1) 值域、状态保存恢复。"""
    rng = ChromaRNG((4, 4, 4, 4))
    rng.set_seed(12345)
    u0 = [rng.random_scalar() for _ in range(5)]
    rng.set_seed(12345)
    u1 = [rng.random_scalar() for _ in range(5)]
    assert u0 == u1, "同 seed 必须可复现"
    assert all(0.0 <= u < 1.0 for u in u0), "均匀数必须在 [0,1)"
    rng2 = ChromaRNG((4, 4, 4, 4))
    rng2.set_seed(12345)
    [rng2.random_scalar() for _ in range(3)]
    saved = rng2.save()
    expected = [rng2.random_scalar() for _ in range(3)]
    rng2.restore()
    assert rng2.get_seed() == saved
    assert [rng2.random_scalar() for _ in range(3)] == expected
    vals, pos = rng.gaussian_fermion_subset(1)
    assert vals.shape == (128, 4, 3), f"4^4 lattice odd subset = 128 sites, got {vals.shape}"
    assert numpy.isfinite(vals).all()
    print("chroma_rng selftest ok; first uniforms:", u0)


if __name__ == "__main__":
    selftest()
