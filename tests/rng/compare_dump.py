#!/usr/bin/env python3
"""比较 Chroma/PyQUDA v1 文本 dump，仅依赖 Python 标准库，不需要 GPU。

用法：python3 compare_dump.py <chroma.txt> <pyquda.txt>；退出码 0/1/2 为通过/失败/用法错误。
多 rank 输出须先合并为完整格点集合，可保留各 rank 相同的 JSON 头。
uniforms 按 float32 位模式比较；eta/mom 使用 |a-b|/max(1,|a|) <= tol，
double/unknown 的 tol=1e-12，显式 single 的 tol=2e-6（沿用历史口径）。
来源：25d5e347:tests/hasenbusch_operation/chroma_rng/compare_dump.py。
"""

import json
import math
from pathlib import Path
import struct
import sys


COMPONENTS = {"uniforms": 24, "eta": 24, "mom": 72}


def _json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate metadata key: {key}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"nonfinite metadata: {value}")


def _validate_header(header):
    if not isinstance(header, dict):
        raise ValueError("header must be a JSON object")
    if header.get("format") != "chroma_rng_dump_v1":
        raise ValueError("unsupported or missing format")
    if not isinstance(header.get("mode"), str) or header["mode"] not in COMPONENTS:
        raise ValueError("unsupported or missing mode")
    latt = header.get("latt")
    if (not isinstance(latt, list) or len(latt) != 4
            or any(type(n) is not int or n <= 0 for n in latt)):
        raise ValueError("latt must contain four positive integer extents")
    seed = header.get("seed")
    if type(seed) is not int or not 0 <= seed < 2**47:
        raise ValueError("seed must be a 47-bit nonnegative integer")
    if header.get("precision", "unknown") not in ("double", "single", "unknown"):
        raise ValueError("unsupported precision")


def _float32_bits(value):
    try:
        bits = struct.pack("!f", value)
    except OverflowError as exc:
        raise ValueError("nonfinite float32 conversion") from exc
    if not math.isfinite(struct.unpack("!f", bits)[0]):
        raise ValueError("nonfinite float32 conversion")
    return bits


def load_dump(path):
    """读取并验证完整 dump；重复坐标、缺分量、缺格点或非有限值均报错。"""
    header, rows = None, {}
    with Path(path).open(encoding="utf-8") as stream:
        for number, raw in enumerate(stream, 1):
            line = raw.strip()
            if not line:
                continue
            try:
                if line.startswith("#"):
                    current = json.loads(line[1:], object_pairs_hook=_json_object,
                                         parse_constant=_reject_constant)
                    _validate_header(current)
                    if header is not None and current != header:
                        raise ValueError("inconsistent repeated header")
                    header = current
                    continue
                if header is None:
                    raise ValueError("missing header before data")
                parts = line.split()
                ncomp = COMPONENTS[header["mode"]]
                if len(parts) != 4 + ncomp:
                    raise ValueError(f"expected 4 coordinates and {ncomp} components")
                coord = tuple(int(v) for v in parts[:4])
                if coord in rows:
                    raise ValueError(f"duplicate coordinate: {coord}")
                if any(not 0 <= c < n for c, n in zip(coord, header["latt"])):
                    raise ValueError(f"coordinate outside lattice: {coord}")
                if header["mode"] != "mom" and sum(coord) % 2 != 1:
                    raise ValueError(f"coordinate outside odd subset: {coord}")
                values = tuple(float(v) for v in parts[4:])
                if not all(math.isfinite(v) for v in values):
                    raise ValueError(f"nonfinite component at {coord}")
                if header["mode"] == "uniforms":
                    for value in values:
                        _float32_bits(value)
                rows[coord] = values
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{path}:{number}: {exc}") from exc
    if header is None:
        raise ValueError(f"{path}: missing header")
    volume = math.prod(header["latt"])
    expected = volume if header["mode"] == "mom" else volume // 2
    # 坐标已验证边界、奇偶与唯一性，数量相等即覆盖完整预期键集合。
    if not rows or len(rows) != expected:
        raise ValueError(f"{path}: incomplete coordinate set: got {len(rows)}, expected {expected}")
    return header, rows


def compare_dumps(left, right):
    """以左侧为数值参考；失败抛出 ValueError，通过返回简短统计。"""
    h1, r1 = load_dump(left)
    h2, r2 = load_dump(right)
    for key in ("format", "mode", "latt", "seed"):
        if h1[key] != h2[key]:
            raise ValueError(f"{key} mismatch: {h1[key]!r} vs {h2[key]!r}")
    p1, p2 = h1.get("precision", "unknown"), h2.get("precision", "unknown")
    if p1 != p2 and "unknown" not in (p1, p2):
        raise ValueError(f"precision mismatch: {p1} vs {p2}")
    # source 是生产端标识，不要求相同；其他双方声明的扩展元数据须一致。
    for key in h1.keys() & h2.keys() - {"source", "precision"}:
        if h1[key] != h2[key]:
            raise ValueError(f"{key} mismatch")
    if r1.keys() != r2.keys():
        raise ValueError("coordinate key sets differ")
    mode = h1["mode"]
    tol = 2e-6 if "single" in (p1, p2) else 1e-12
    max_abs = 0.0
    for coord in sorted(r1):
        for index, (a, b) in enumerate(zip(r1[coord], r2[coord])):
            diff = abs(a - b)
            max_abs = max(max_abs, diff)
            if mode == "uniforms":
                # 直接比较 IEEE float32 字节，+0/-0 也不能误判为位级一致。
                ok = _float32_bits(a) == _float32_bits(b)
            else:
                ok = diff / max(1.0, abs(a)) <= tol
            if not ok:
                criterion = "float32 bit mismatch" if mode == "uniforms" else f"tolerance {tol:g} exceeded"
                raise ValueError(f"{criterion} at {coord} component {index}: {a!r} vs {b!r}")
    return (f"mode={mode} sites={len(r1)} components={COMPONENTS[mode]} "
            f"max_abs={max_abs:.3g} "
            + ("bit-exact" if mode == "uniforms" else f"tol={tol:g}"))


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2:
        print(__doc__)
        return 2
    try:
        summary = compare_dumps(*args)
    except (ValueError, OSError) as exc:
        print(f"verdict: FAIL: {exc}", file=sys.stderr)
        return 1
    print(summary)
    print("verdict: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
