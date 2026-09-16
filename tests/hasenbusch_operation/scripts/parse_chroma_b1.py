#!/usr/bin/env python3
"""解析 Chroma HMC B1 输出 XML，并可对照 PyQUDA JSON 结果。

用法：
  python3 parse_chroma_b1.py chroma_out.xml [--pyquda-json pyquda.json] \
      [--json-out result.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET


def text_number(node: ET.Element | None) -> float | None:
    if node is None or node.text is None:
        return None
    return float(node.text)


def parse_chroma(path: Path) -> dict:
    root = ET.parse(path).getroot()
    updates = []
    for update in root.findall(".//Update"):
        hmc = update.find("./HMCTrajectory")
        if hmc is None:
            continue
        plaquettes = update.findall("./InlineObservables/elem/Plaquette")
        if not plaquettes:
            plaquettes = update.findall(".//InlineObservables//Plaquette")
        plaquette = plaquettes[-1] if plaquettes else None
        updates.append({
            "update_no": int(plaquette.findtext("update_no", "0")) if plaquette is not None else None,
            "delta_H": text_number(hmc.find("deltaH")),
            "acceptance_probability": text_number(hmc.find("AccProb")),
            "accept": hmc.findtext("AcceptP", "").lower() == "true",
            "plaquette": {
                "w": text_number(plaquette.find("w_plaq")) if plaquette is not None else None,
                "s": text_number(plaquette.find("s_plaq")) if plaquette is not None else None,
                "t": text_number(plaquette.find("t_plaq")) if plaquette is not None else None,
            },
            "H_old": {
                "KE": text_number(hmc.find("./H_old/KE_old")),
                "PE": text_number(hmc.find("./H_old/PE_old")),
            },
            "H_new": {
                "KE": text_number(hmc.find("./H_new/KE_new")),
                "PE": text_number(hmc.find("./H_new/PE_new")),
            },
            "seconds_for_trajectory": text_number(update.find("seconds_for_trajectory")),
        })
    return {
        "source": str(path),
        "n_updates": len(updates),
        "updates": updates,
    }


def attach_pyquda(result: dict, pyquda_path: Path) -> None:
    data = json.loads(pyquda_path.read_text())
    records = data.get("records", [])
    result["pyquda"] = {
        "source": str(pyquda_path),
        "run_id": data.get("run_id"),
        "ntraj": data.get("ntraj"),
        "records": records,
    }
    comparisons = []
    for chroma in result["updates"]:
        candidates = [
            rec for rec in records
            if rec.get("traj") == chroma.get("update_no")
        ]
        pyq = candidates[0] if candidates else (records[0] if records else None)
        if pyq is None:
            continue
        comparisons.append({
            "update_no": chroma.get("update_no"),
            "chroma_delta_H": chroma.get("delta_H"),
            "pyquda_delta_H": pyq.get("delta_H"),
            "delta_H_difference": (
                chroma.get("delta_H") - pyq.get("delta_H")
                if chroma.get("delta_H") is not None and pyq.get("delta_H") is not None
                else None
            ),
            "chroma_accept": chroma.get("accept"),
            "pyquda_accept": pyq.get("accept"),
            "chroma_plaquette_first": chroma.get("plaquette", {}).get("w"),
            "pyquda_plaquette_first": (pyq.get("plaquette") or [None])[0],
        })
    result["comparisons"] = comparisons


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("chroma_xml", type=Path)
    parser.add_argument("--pyquda-json", type=Path)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    result = parse_chroma(args.chroma_xml)
    if args.pyquda_json is not None:
        attach_pyquda(result, args.pyquda_json)

    text = json.dumps(result, indent=2, ensure_ascii=False)
    if args.json_out is not None:
        args.json_out.write_text(text + "\n")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
