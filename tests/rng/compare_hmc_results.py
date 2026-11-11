#!/usr/bin/env python3
"""CPU 比较 Chroma 正式 XML 与 runner 的 rank0.jsonl，不读取 stdout 调试行。

CLI: --chroma XML --pyquda JSONL --output JSON --ntraj N。
Chroma 未内嵌每轨结束 RNG 时，按轨迹顺序重复 --chroma-restart XML 显式提供
restart；缺少 seed 证据不能通过。可用 --dh-atol/--plaq-atol/--rtol 明确阈值。
只核验输出数值和可观察 RNG 状态，不宣称力门禁、求解器收敛或场布局已验证。
"""

import argparse
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET


def finite_number(value, label):
    if type(value) not in (float, int) or not math.isfinite(value):
        raise ValueError("missing/non-finite number: " + label)
    return value


def check_finite(value):
    """连同可选能量字段检查 NaN/Inf，避免仅终点差值有限而误通过。"""
    if isinstance(value, dict):
        for v in value.values():
            check_finite(v)
    elif isinstance(value, list):
        for v in value:
            check_finite(v)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite JSON value")


def seed_value(value):
    if isinstance(value, list):
        if len(value) != 4 or any(type(x) is not int or not 0 <= x < 4096 for x in value) or value[3] >= 2048:
            raise ValueError("invalid four-word seed")
        value = sum(x << (12 * i) for i, x in enumerate(value))
    if type(value) is not int or not 0 <= value < 2**47:
        raise ValueError("missing/invalid 47-bit seed")
    return value


def xml_seed(node):
    if node is None:
        raise ValueError("missing Chroma RNG/Seed")
    words = [int(x) for x in " ".join(node.itertext()).split()]
    return seed_value(words)


def one_text(node, paths, label):
    values = [(e.text or "").strip() for path in paths for e in node.findall(path)]
    if len(values) != 1 or not values[0]:
        raise ValueError("missing/ambiguous Chroma " + label)
    return values[0]


def checkpoint_seed(path):
    root = ET.parse(path).getroot()
    candidates = root.findall(".//RNG/Seed")
    if len(candidates) != 1:
        raise ValueError("restart must contain exactly one RNG/Seed: " + str(path))
    return xml_seed(candidates[0])


def chroma_records(path, restarts=()):
    root = ET.parse(path).getroot()
    updates = root.findall(".//MCUpdates/elem/Update")
    if not updates:
        raise ValueError("no Chroma MCUpdates/elem/Update records")
    if restarts and len(restarts) != len(updates):
        raise ValueError("need one explicitly ordered restart per Chroma update")
    rows = []
    for i, update in enumerate(updates):
        trj = update.find("HMCTrajectory")
        if trj is None:
            raise ValueError("incomplete Chroma update: no HMCTrajectory")
        dh = float(one_text(trj, (".//DeltaH", ".//deltaH", ".//delta_H"), "delta_H"))
        accepted = one_text(trj, (".//AcceptP", ".//acceptTraj", ".//Accept", ".//accept"), "accept").lower()
        if accepted not in ("true", "false", "1", "0"):
            raise ValueError("invalid Chroma accept boolean")
        update_no = int(one_text(update, ("update_no",), "update_no"))
        # 只取明确的正式测量路径；按节点计数，空节点也不能掩盖重复测量。
        plaq_nodes = [node for path in ("Plaquette", "Observables/Plaquette", "InlineObservables/elem/Plaquette")
                      for node in update.findall(path)]
        if len(plaq_nodes) != 1:
            raise ValueError("missing/ambiguous Chroma formal plaquette")
        plaq_node = plaq_nodes[0]
        plaq = float(one_text(plaq_node, ("w_plaq",), "formal plaquette"))
        # 新版正式测量可携带自己的轨迹号，存在时必须唯一且与 Update 对齐。
        if plaq_node.findall("update_no"):
            if int(one_text(plaq_node, ("update_no",), "plaquette update_no")) != update_no:
                raise ValueError("Chroma plaquette update_no disagrees with Update")
        row = dict(trajectory=i + 1, update_no=update_no,
                   delta_H=finite_number(dh, "Chroma delta_H"),
                   plaquette=finite_number(plaq, "Chroma plaquette"),
                   accept=accepted in ("true", "1"))
        seed_nodes = update.findall("RNG/Seed")
        if len(seed_nodes) > 1:
            raise ValueError("ambiguous Chroma ending seed")
        seed = xml_seed(seed_nodes[0]) if seed_nodes else None
        if restarts:
            restart = checkpoint_seed(restarts[i])
            if seed is not None and seed != restart:
                raise ValueError("inline/restart seed disagree")
            seed = restart
        if seed is None:
            raise ValueError("missing Chroma ending seed; pass --chroma-restart in trajectory order")
        row["seed_after"] = seed
        row["seed_source"] = str(restarts[i]) if restarts else "Update/RNG/Seed"
        rows.append(row)
    if any(b["update_no"] != a["update_no"] + 1 for a, b in zip(rows, rows[1:])):
        raise ValueError("Chroma update numbers not consecutive/unique")
    # Config_info 保存组态的历史生产参数，不能当成本次输入 RNG 状态。
    initial = [node for path in ("Input/Params/MCControl/RNG/Seed", "MCControl/RNG/Seed")
               for node in root.findall(path)]
    if len(initial) > 1:
        raise ValueError("ambiguous Chroma initial seed")
    return rows, xml_seed(initial[0]) if initial else None


def pyquda_records(path):
    rows = []
    with open(path, encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"JSONL line {line_no} is not an object")
            check_finite(row)
            if row.get("type") == "failure":
                raise ValueError("PyQUDA failure record: " + str(row.get("error", row)))
            if row.get("type") != "trajectory":
                continue
            if type(row.get("trajectory")) is not int or row["trajectory"] != len(rows) + 1:
                raise ValueError("PyQUDA trajectory numbers must start at 1 and be consecutive")
            if type(row.get("accept")) is not bool:
                raise ValueError("PyQUDA accept must be a boolean")
            finite_number(row.get("delta_H"), "PyQUDA delta_H")
            plaq = row.get("plaquette")
            if isinstance(plaq, list):
                if len(plaq) != 3:
                    raise ValueError("PyQUDA plaquette array must have mean/spatial/temporal components")
                for p in plaq:
                    finite_number(p, "PyQUDA plaquette component")
                plaq = plaq[0]
            row["plaquette"] = finite_number(plaq, "PyQUDA plaquette")
            for key in ("seed_before", "seed_after_refresh", "seed_after"):
                row[key] = seed_value(row.get(key))
            if rows and row["seed_before"] != rows[-1]["seed_after"]:
                raise ValueError("PyQUDA seed continuity broken")
            rows.append(row)
    return rows


def compare(chroma, pyquda, ntraj, *, restarts=(), dh_atol=1e-7, plaq_atol=1e-12, rtol=0.0):
    """返回可序列化报告；缺失、超量、不完整、非有限值均失败，不做 zip 截断通过。"""
    if type(ntraj) is not int or ntraj <= 0:
        raise ValueError("ntraj must be a positive integer")
    for value in (dh_atol, plaq_atol, rtol):
        finite_number(value, "tolerance")
        if value < 0:
            raise ValueError("negative comparison tolerance")
    report = dict(status="failed", passed=False, ntraj=ntraj, issues=[], paired=[],
                  inputs=dict(chroma=str(chroma), pyquda=str(pyquda), restarts=[str(p) for p in restarts]),
                  tolerances=dict(dh_atol=dh_atol, plaq_atol=plaq_atol, rtol=rtol),
                  seed_scope="cross-check ending states and PyQUDA continuity; refresh has no Chroma counterpart",
                  initial_seed_checked=False)
    try:
        crows, initial = chroma_records(chroma, restarts)
        qrows = pyquda_records(pyquda)
        report["counts"] = dict(chroma=len(crows), pyquda=len(qrows))
        if len(crows) != ntraj or len(qrows) != ntraj:
            raise ValueError(f"expected exactly {ntraj} trajectories per side; got {len(crows)}/{len(qrows)}")
        if initial is not None:
            report["initial_seed_checked"] = True
            if initial != qrows[0]["seed_before"]:
                report["issues"].append("initial seed mismatch")
        for c, q in zip(crows, qrows):
            dh_difference = q["delta_H"] - c["delta_H"]
            plaq_difference = q["plaquette"] - c["plaquette"]
            finite_number(dh_difference, "delta_H difference")
            finite_number(plaq_difference, "plaquette difference")
            checks = dict(delta_H=math.isclose(c["delta_H"], q["delta_H"], rel_tol=rtol, abs_tol=dh_atol),
                          plaquette=math.isclose(c["plaquette"], q["plaquette"], rel_tol=rtol, abs_tol=plaq_atol),
                          accept=c["accept"] == q["accept"], seed_after=c["seed_after"] == q["seed_after"])
            report["paired"].append(dict(trajectory=c["trajectory"], chroma=c,
                                         pyquda={k: q[k] for k in ("delta_H", "plaquette", "accept", "seed_before", "seed_after_refresh", "seed_after")},
                                         delta_H_difference=dh_difference, plaquette_difference=plaq_difference,
                                         checks=checks, passed=all(checks.values())))
            for key, ok in checks.items():
                if not ok:
                    report["issues"].append(f"trajectory {c['trajectory']}: {key} mismatch")
    except (ValueError, OSError, ET.ParseError, OverflowError) as exc:
        report["issues"].append(str(exc))
    report["passed"] = not report["issues"] and len(report["paired"]) == ntraj
    report["status"] = "passed" if report["passed"] else "failed"
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("chroma", "pyquda", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--ntraj", required=True, type=int)
    parser.add_argument("--chroma-restart", action="append", default=[], type=Path)
    parser.add_argument("--dh-atol", type=float, default=1e-7)
    parser.add_argument("--plaq-atol", type=float, default=1e-12)
    parser.add_argument("--rtol", type=float, default=0.0)
    args = parser.parse_args(argv)
    if args.output.resolve() in {p.resolve() for p in (args.chroma, args.pyquda, *args.chroma_restart)}:
        raise ValueError("output would overwrite an input")
    report = compare(args.chroma, args.pyquda, args.ntraj, restarts=args.chroma_restart,
                     dh_atol=args.dh_atol, plaq_atol=args.plaq_atol, rtol=args.rtol)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(dict(status=report["status"], issues=report["issues"]), ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError) as exc:
        sys.exit("compare_hmc_results: " + str(exc))
