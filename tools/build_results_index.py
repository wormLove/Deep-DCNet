"""
Build one consolidated index of everything under RESULT/, instead of
manually ls-ing into a dozen RESULT/<run_name>/<timestamp>/ folders.

Reads, for every finished training run:
    RESULT/<run_name>/<timestamp>/run_config.json   (what was trained)
    RESULT/<run_name>/<timestamp>/eval_log.txt       (the "final_eval" line - what it scored)
and for every finished robustness sweep:
    RESULT/point4_robustness_<dataset>[_smoke]/<timestamp>/robustness.tsv

...and writes:
    RESULT/INDEX.md         human-readable tables (open this first)
    RESULT/index_runs.csv   one row per training run
    RESULT/index_robustness.csv   one row per (sweep, model, perturbation, level)

Safe to re-run any time after new jobs finish - it only reads, never
modifies anything under RESULT/, and always rebuilds the three output
files from scratch (no stale rows left behind from a renamed/deleted run).

Usage (from the repo root, same venv you train with):
    python tools/build_results_index.py
    python tools/build_results_index.py --result-root RESULT
"""
import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


def _finished_run_dirs(result_root: Path):
    """Every RESULT/<run_name>/<timestamp>/ that has a run_config.json."""
    for run_name_dir in sorted(result_root.iterdir()):
        if not run_name_dir.is_dir() or run_name_dir.name.startswith("point4_robustness_"):
            continue
        for ts_dir in sorted(run_name_dir.iterdir()):
            cfg_path = ts_dir / "run_config.json"
            if ts_dir.is_dir() and cfg_path.exists():
                yield run_name_dir.name, ts_dir


def _parse_final_acc(eval_log_path: Path) -> Optional[float]:
    """
    Pulls the accuracy off the LAST 'final_eval' line in eval_log.txt
    (training_engines.py always logs one at the very end of a run). Falls
    back to the last line in the file at all if that exact tag is missing
    (e.g. a run that crashed before finishing writes no final_eval line,
    so this correctly surfaces as "no final accuracy" rather than a wrong
    mid-training number).
    """
    if not eval_log_path.exists():
        return None
    final_lines = [ln for ln in eval_log_path.read_text(encoding="utf-8").splitlines() if ln.startswith("final_eval")]
    if not final_lines:
        return None
    m = re.search(r"acc=([\d.]+)", final_lines[-1])
    return float(m.group(1)) if m else None


def collect_runs(result_root: Path) -> List[Dict[str, Any]]:
    rows = []
    for run_name, ts_dir in _finished_run_dirs(result_root):
        cfg = json.loads((ts_dir / "run_config.json").read_text(encoding="utf-8"))
        final_acc = _parse_final_acc(ts_dir / "eval_log.txt")
        has_ckpt = (ts_dir / "checkpoints" / "final.pth").exists()
        rows.append({
            "run_name": run_name,
            "timestamp": ts_dir.name,
            "dataset": cfg.get("dataset", ""),
            "arch": cfg.get("arch", ""),
            "head": cfg.get("head", ""),
            "hidden_dim": cfg.get("hidden_dim", cfg.get("layer_dims", "")),
            "num_train_samples": cfg.get("num_train_samples", ""),
            "final_acc": f"{final_acc:.2f}" if final_acc is not None else "MISSING",
            "has_checkpoint": "yes" if has_ckpt else "NO",
            "run_dir": str(ts_dir),
        })
    return rows


def collect_robustness(result_root: Path) -> List[Dict[str, Any]]:
    rows = []
    for sweep_dir in sorted(result_root.glob("point4_robustness_*")):
        if not sweep_dir.is_dir():
            continue
        for ts_dir in sorted(sweep_dir.iterdir()):
            tsv_path = ts_dir / "robustness.tsv"
            if not ts_dir.is_dir() or not tsv_path.exists():
                continue
            with open(tsv_path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f, delimiter="\t"):
                    row["sweep"] = sweep_dir.name
                    row["timestamp"] = ts_dir.name
                    rows.append(row)
    return rows


def write_csv(rows: List[Dict[str, Any]], path: Path, fieldnames: List[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _md_table(rows: List[Dict[str, Any]], columns: List[str]) -> str:
    if not rows:
        return "_(none found)_\n"
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(c, "")) for c in columns) + " |")
    return "\n".join(lines) + "\n"


def write_markdown(run_rows: List[Dict[str, Any]], rob_rows: List[Dict[str, Any]], path: Path) -> None:
    run_cols = ["run_name", "dataset", "arch", "head", "hidden_dim", "final_acc", "has_checkpoint", "timestamp"]
    rob_cols = ["sweep", "model", "perturbation", "level", "acc", "drop_vs_clean"]

    missing = [r["run_name"] for r in run_rows if r["final_acc"] == "MISSING" or r["has_checkpoint"] == "NO"]

    parts = ["# RESULT index\n", "Auto-generated by `tools/build_results_index.py` - do not hand-edit.\n"]
    if missing:
        parts.append(f"**Incomplete runs (missing final accuracy or checkpoint):** {', '.join(missing)}\n")
    parts.append("\n## Training runs\n")
    parts.append(_md_table(run_rows, run_cols))
    parts.append("\n## Robustness sweeps\n")
    parts.append(_md_table(rob_rows, rob_cols))
    path.write_text("\n".join(parts), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", default="RESULT", help="Path to the RESULT/ folder (default: ./RESULT)")
    args = parser.parse_args()

    result_root = Path(args.result_root)
    if not result_root.is_dir():
        raise SystemExit(f"No RESULT folder at {result_root.resolve()}")

    run_rows = collect_runs(result_root)
    rob_rows = collect_robustness(result_root)

    write_csv(run_rows, result_root / "index_runs.csv",
              ["run_name", "timestamp", "dataset", "arch", "head", "hidden_dim",
               "num_train_samples", "final_acc", "has_checkpoint", "run_dir"])
    write_csv(rob_rows, result_root / "index_robustness.csv",
              ["sweep", "timestamp", "dataset", "model", "perturbation", "level", "acc", "drop_vs_clean"])
    write_markdown(run_rows, rob_rows, result_root / "INDEX.md")

    print(f"[{len(run_rows)}] training runs -> {result_root / 'index_runs.csv'}")
    print(f"[{len(rob_rows)}] robustness rows -> {result_root / 'index_robustness.csv'}")
    print(f"Summary -> {result_root / 'INDEX.md'}")
    missing = [r["run_name"] for r in run_rows if r["final_acc"] == "MISSING" or r["has_checkpoint"] == "NO"]
    if missing:
        print(f"\n[!] {len(missing)} run(s) look incomplete (no final_eval line or no checkpoints/final.pth): {missing}")


if __name__ == "__main__":
    main()
