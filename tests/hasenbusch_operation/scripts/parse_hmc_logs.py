#!/usr/bin/env python3
import csv
import math
import re
from pathlib import Path


OP_DIR = Path(__file__).resolve().parents[1]
LOG_DIR = OP_DIR / "logs"
CSV_DIR = OP_DIR / "csv"


FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def parse_meta(text: str, key: str, default: str = "") -> str:
    match = re.search(rf"{re.escape(key)} = ([^\n]+)", text)
    return match.group(1).strip() if match else default


def parse_float_list(pattern: str, text: str) -> list[float]:
    return [float(match.group(1)) for match in re.finditer(pattern, text)]


def parse_log(path: Path) -> tuple[dict, list[dict]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    run_id = parse_meta(text, "run_id", path.stem)
    lattice = parse_meta(text, "lattice")
    variant = parse_meta(text, "variant")
    cfg = parse_meta(text, "cfg")
    t_boundary = int(parse_meta(text, "t_boundary", "0"))
    trajectory_length = float(parse_meta(text, "trajectory_length", "nan"))
    stop = int(parse_meta(text, "stop", "0"))
    warm = int(parse_meta(text, "warm", "0"))
    nsteps_outer = int(parse_meta(text, "nsteps_outer", "0"))
    nsteps_inner = int(parse_meta(text, "nsteps_inner", "0"))
    hasenbusch_mu = parse_meta(text, "hasenbusch_mu")
    gauge_path = parse_meta(text, "gauge_path")
    monomial_count = int(parse_meta(text, "monomial_count", "0") or 0)
    fermion_monomial_count = int(parse_meta(text, "fermion_monomial_count", "0") or 0)
    plaquettes = parse_float_list(rf"Plaquette = \[({FLOAT})", text)
    delta_e = parse_float_list(rf"Delta_E = ({FLOAT})", text)
    accepts = [match.group(1) == "True" for match in re.finditer(r"accept\? (True|False)", text)]
    failed_reason = parse_meta(text, "Failure reason")
    status_match = re.search(r"Run status = (ok|failed)", text)
    status = status_match.group(1) if status_match else ("failed" if failed_reason else "unknown")
    if len(delta_e) < stop and status in ("ok", "unknown"):
        status = "failed"
        failed_reason = "incomplete before stop"
    if "Clover trlog has returned nan" in text or "nan" in failed_reason.lower():
        status = "failed"
        failed_reason = failed_reason or "NaN detected"
    sqrt_abs_delta_e = [math.sqrt(abs(value)) for value in delta_e]
    summary = {
        "run_id": run_id,
        "lattice": lattice,
        "variant": variant,
        "cfg": cfg,
        "t_boundary": t_boundary,
        "trajectory_length": trajectory_length,
        "status": status,
        "failed_reason": failed_reason,
        "stop": stop,
        "warm": warm,
        "nsteps_outer": nsteps_outer,
        "nsteps_inner": nsteps_inner,
        "hasenbusch_mu": hasenbusch_mu,
        "monomial_count": monomial_count,
        "fermion_monomial_count": fermion_monomial_count,
        "n_delta_e": len(delta_e),
        "accepted": sum(accepts),
        "rejected": len(accepts) - sum(accepts),
        "accept_rate": sum(accepts) / len(accepts) if accepts else 0.0,
        "delta_e_min": min(delta_e) if delta_e else math.nan,
        "delta_e_mean": sum(delta_e) / len(delta_e) if delta_e else math.nan,
        "delta_e_max_abs": max(abs(value) for value in delta_e) if delta_e else math.nan,
        "sqrt_abs_delta_e_mean": sum(sqrt_abs_delta_e) / len(sqrt_abs_delta_e) if sqrt_abs_delta_e else math.nan,
        "sqrt_abs_delta_e_max": max(sqrt_abs_delta_e) if sqrt_abs_delta_e else math.nan,
        "plaquette_initial": plaquettes[0] if plaquettes else math.nan,
        "plaquette_final": plaquettes[-1] if plaquettes else math.nan,
        "plaquette_mean": sum(plaquettes) / len(plaquettes) if plaquettes else math.nan,
        "plaquette_min": min(plaquettes) if plaquettes else math.nan,
        "plaquette_max": max(plaquettes) if plaquettes else math.nan,
        "plaquette_range": max(plaquettes) - min(plaquettes) if plaquettes else math.nan,
        "plaquette_final_minus_initial": plaquettes[-1] - plaquettes[0] if len(plaquettes) >= 2 else math.nan,
        "gauge_path": gauge_path,
        "log_file": str(path),
    }
    detail = []
    for idx, value in enumerate(delta_e, start=1):
        detail.append(
            {
                "run_id": run_id,
                "lattice": lattice,
                "variant": variant,
                "cfg": cfg,
                "t_boundary": t_boundary,
                "trajectory_length": trajectory_length,
                "trajectory": idx,
                "delta_e": value,
                "sqrt_abs_delta_e": math.sqrt(abs(value)),
                "accepted": accepts[idx - 1] if idx - 1 < len(accepts) else "",
                "plaquette_after": plaquettes[idx] if idx < len(plaquettes) else math.nan,
                "log_file": str(path),
            }
        )
    return summary, detail


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    summaries = []
    details = []
    for path in sorted(LOG_DIR.glob("*.log")):
        summary, detail = parse_log(path)
        summaries.append(summary)
        details.extend(detail)
    summaries.sort(key=lambda row: (row["cfg"], row["trajectory_length"], row["run_id"]))
    details.sort(key=lambda row: (row["cfg"], row["trajectory_length"], row["run_id"], row["trajectory"]))
    write_csv(CSV_DIR / "hmc_summary.csv", summaries)
    write_csv(CSV_DIR / "hmc_trajectory_detail.csv", details)
    print(CSV_DIR / "hmc_summary.csv")
    print(CSV_DIR / "hmc_trajectory_detail.csv")
    for row in summaries:
        print(
            f"{row['run_id']}: status={row['status']} n={row['n_delta_e']}/{row['stop']} "
            f"accept={row['accepted']}/{row['n_delta_e']} "
            f"plaquette_range={row['plaquette_range']:.8e} delta_e_mean={row['delta_e_mean']:.8e} "
            f"delta_e_max_abs={row['delta_e_max_abs']:.8e}"
        )


if __name__ == "__main__":
    main()
