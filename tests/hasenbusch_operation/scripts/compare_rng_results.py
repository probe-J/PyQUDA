#!/usr/bin/env python3
"""方案 A 统计等价检验：比较两臂 run_rng_compare.py 输出的 JSON。

输出：ΔH 的两样本 KS 统计量与近似 p 值、接受率、plaquette 均值/方差对比。
无 scipy 依赖（手写 KS + 渐近 p 值）。
"""

import argparse
import json
import math
from pathlib import Path


def ks_two_sample(xs, ys):
    xs = sorted(xs)
    ys = sorted(ys)
    nx, ny = len(xs), len(ys)

    def cdf_at(sorted_arr, v):
        lo = 0
        hi = len(sorted_arr)
        while lo < hi:
            mid = (lo + hi) // 2
            if sorted_arr[mid] <= v:
                lo = mid + 1
            else:
                hi = mid
        return lo / len(sorted_arr)

    allvals = sorted(set(xs) | set(ys))
    d = 0.0
    for v in allvals:
        d = max(d, abs(cdf_at(xs, v) - cdf_at(ys, v)))
    ne = nx * ny / (nx + ny)
    # 渐近 KS 分布（Stephens 近似）
    lam = (math.sqrt(ne) + 0.12 + 0.11 / math.sqrt(ne)) * d
    if lam < 0.2:
        p = 1.0
    else:
        p = 0.0
        for k in range(1, 101):
            p += 2.0 * ((-1) ** (k - 1)) * math.exp(-2.0 * k * k * lam * lam)
        p = min(1.0, max(0.0, p))
    return d, p


def mean(v):
    return sum(v) / len(v)


def var(v):
    m = mean(v)
    return sum((x - m) ** 2 for x in v) / max(1, len(v) - 1)


def plaquette_first(record):
    # 统一取 plaquette 三分量中的第一分量用于方案 A 比较。
    value = record["plaquette"]
    if isinstance(value, (list, tuple)):
        return float(value[0])
    return float(value)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("json_a")
    ap.add_argument("json_b")
    args = ap.parse_args()

    a = json.loads(Path(args.json_a).read_text())
    b = json.loads(Path(args.json_b).read_text())
    da = [r["delta_H"] for r in a["records"]]
    db = [r["delta_H"] for r in b["records"]]
    pa = [plaquette_first(r) for r in a["records"]]
    pb = [plaquette_first(r) for r in b["records"]]
    aa = sum(1 for r in a["records"] if r["accept"]) / len(a["records"])
    ab = sum(1 for r in b["records"] if r["accept"]) / len(b["records"])

    d_ks, p_ks = ks_two_sample(da, db)
    p_ks_plaq, p_ks_plaq_value = ks_two_sample(pa, pb)
    print(f"A arm: n={len(da)} accept={aa:.3f} <dH>={mean(da):.6g} var(dH)={var(da):.6g} <plaq>={mean(pa):.8f}")
    print(f"B arm: n={len(db)} accept={ab:.3f} <dH>={mean(db):.6g} var(dH)={var(db):.6g} <plaq>={mean(pb):.8f}")
    print(f"KS(dH): D={d_ks:.4f} p≈{p_ks:.4f}")
    print(f"KS(plaquette first): D={p_ks_plaq:.4f} p≈{p_ks_plaq_value:.4f}")
    if p_ks > 0.05 and p_ks_plaq_value > 0.05:
        verdict = "NO SIGNIFICANT DIFFERENCE DETECTED (insufficient for proof)"
    else:
        verdict = "DIFFERENCE DETECTED (do not claim statistical equivalence)"
    print(f"verdict: {verdict}")
    if abs(mean(pa) - mean(pb)) > 1e-3:
        print("warning: plaquette first-component means are separated by more than 1e-3")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
