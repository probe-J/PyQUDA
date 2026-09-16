#!/usr/bin/env python3
"""B0 位级/容差对拍：chroma_rng_dump 输出 vs pyquda ChromaRNG dump_reference。

用法：python3 compare_dump.py <chroma.txt> <pyquda.txt>
判定：
- uniforms：float32 位级一致（文本以 %.9g 写出，按 float32 解析后严格相等）
- eta/mom：double 容差（默认 1e-12 相对 + 1e-12 绝对；单精度 build 时按 1e-6）
"""

import json
import math
import sys
from pathlib import Path

import numpy


def load_dump(path):
    lines = Path(path).read_text().splitlines()
    header = None
    rows = {}
    for line in lines:
        if line.startswith("#"):
            header = json.loads(line[1:].strip())
            continue
        parts = line.split()
        coords = tuple(int(v) for v in parts[:4])
        rows[coords] = numpy.array([float(v) for v in parts[4:]], dtype=numpy.float64)
    assert header is not None, "missing header"
    return header, rows


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    h1, r1 = load_dump(sys.argv[1])
    h2, r2 = load_dump(sys.argv[2])

    assert h1["mode"] == h2["mode"], f"mode mismatch: {h1['mode']} vs {h2['mode']}"
    assert h1["latt"] == h2["latt"], "lattice size mismatch"
    assert h1["seed"] == h2["seed"], "seed mismatch"

    common = sorted(set(r1) & set(r2))
    only1 = set(r1) - set(r2)
    only2 = set(r2) - set(r1)
    print(f"mode={h1['mode']} latt={h1['latt']} seed={h1['seed']}")
    print(f"sites: chroma={len(r1)} pyquda={len(r2)} common={len(common)}")
    if only1 or only2:
        print(f"WARNING: site mismatch! only_chroma={len(only1)} only_pyquda={len(only2)}")
        for c in sorted(only1)[:5]:
            print("  only chroma:", c)
        for c in sorted(only2)[:5]:
            print("  only pyquda:", c)

    mode = h1["mode"]
    if mode == "uniforms":
        a = numpy.stack([r1[c] for c in common]).astype(numpy.float32)
        b = numpy.stack([r2[c] for c in common]).astype(numpy.float32)
        diff = numpy.abs(a.astype(numpy.float64) - b.astype(numpy.float64))
        n_exact = int(numpy.count_nonzero(a == b))
        print(f"uniforms: max|diff|={diff.max():.3g} bit-exact {n_exact}/{a.size}")
        ok = bool(numpy.array_equal(a, b))
        print("verdict:", "PASS (bit-exact)" if ok else "FAIL")
        return 0 if ok else 1
    else:
        a = numpy.stack([r1[c] for c in common])
        b = numpy.stack([r2[c] for c in common])
        scale = numpy.maximum(1.0, numpy.abs(a))
        rel = numpy.abs(a - b) / scale
        # 单精度 QDP-JIT 的 sin/cos/log 与 Python/Numpy 数学库可能相差一个 ulp；
        # 用 2e-6 覆盖该舍入误差，同时仍能发现明显的流或布局错误。
        tol = 2e-6 if "single" in (h1.get("precision", ""), h2.get("precision", "")) else 1e-12
        n_bad = int(numpy.count_nonzero(rel > tol))
        print(f"values: max_abs={numpy.abs(a - b).max():.3g} max_rel={rel.max():.3g} tol={tol:.0e}")
        print(f"bad elements: {n_bad}/{a.size}")
        ok = n_bad == 0
        print("verdict:", "PASS" if ok else "FAIL")
        return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
