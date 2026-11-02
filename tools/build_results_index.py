"""
Build one consolidated index of everything under RESULT/, instead of
manually ls-ing into a dozen RESULT/<run_name>/<timestamp>/ folders.

Reads, for every finished training run:
    RESULT/<run_name>/<timestamp>/run_config.json   (what was trained: seed, setup, protocol, git commit, ...)
    RESULT/<run_name>/<timestamp>/final_eval.json   (what it scored + cost; runs from Plan 02 on)
    RESULT/<run_name>/<timestamp>/eval_log.txt      (fallback for older runs: the "final_eval" line)
and for every finished robustness sweep:
    RESULT/point4_robustness_<dataset>[_smoke]/<timestamp>/robustness.tsv

...and writes:
    RESULT/INDEX.md                human-readable tables (open this first)
    RESULT/index_runs.csv          one row per training run folder
    RESULT/index_summary.csv       one row per configuration, aggregated over seeds
    RESULT/index_robustness.csv    one row per (sweep, model, perturbation, level)

Seed summary: run names end in _s<seed> (e.g. single_layer_baseline_s33);
runs are grouped by run_name with that suffix stripped, plus dataset, setup
and protocol, and report n_seeds, mean, std (sample std; blank when n=1),
min and max of final_acc. Only the newest finished folder of each run_name
counts (re-running the same seed replaces, not adds). *_smoke runs are
left out of the summary.

Safe to re-run any time after new jobs finish - it only reads, never
modifies anything under RESULT/ except these four output files, which it
always rebuilds from scratch (no stale rows left behind from a
renamed/deleted run).

Usage (from the repo root, same venv you train with):
    python tools/build_results_index.py
    python tools/build_results_index.py --result-root My_Notes_Aishani/results/RESULT
"""
import argparse
import csv
import json
import re
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional

SEED_SUFFIX = re.compile(r"_s(\d+)$")

RUN_COLUMNS = [
    "run_name", "timestamp", "dataset", "arch", "head", "hidden_dim", "setup", "protocol", "seed",
    "num_train_samples", "final_acc", "best_acc", "best_acc_samples", "n_organizes",
    "train_wall_sec", "peak_gpu_mem_mb", "git_commit", "git_dirty", "analysis_level",
    "has_checkpoint", "run_dir",
]
SUMMARY_COLUMNS = ["config", "dataset", "setup", "protocol", "n_seeds", "seeds",
                   "mean_acc", "std_acc", "min_acc", "max_acc", "run_names"]


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
    (training_engines.py always logs one at the very end of a run). A run
    that crashed before finishing writes no final_eval line, so this
    correctly surfaces as "no final accuracy" rather than a wrong
    mid-training number.
    """
    if not eval_log_path.exists():
        return None
    final_lines = [ln for ln in eval_log_path.read_text(encoding="utf-8").splitlines() if ln.startswith("final_eval")]
    if not final_lines:
        return None
    m = re.search(r"acc=([\d.]+)", final_lines[-1])
    return float(m.group(1)) if m else None


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _fmt(v: Any, digits: int = 2) -> Any:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return v


def config_name(run_name: str) -> str:
    """run_name with a trailing _s<seed> removed."""
    return SEED_SUFFIX.sub("", run_name)


def collect_runs(result_root: Path) -> List[Dict[str, Any]]:
    rows = []
    for run_name, ts_dir in _finished_run_dirs(result_root):
        cfg = _read_json(ts_dir / "run_config.json")
        fin = _read_json(ts_dir / "final_eval.json")
        final_acc = fin.get("final_acc")
        if final_acc is None:
            final_acc = _parse_final_acc(ts_dir / "eval_log.txt")
        seed = cfg.get("seed")
        if seed is None:
            m = SEED_SUFFIX.search(run_name)
            seed = int(m.group(1)) if m else None
        has_ckpt = (ts_dir / "checkpoints" / "final.pth").exists()
        rows.append({
            "run_name": run_name,
            "timestamp": ts_dir.name,
            "dataset": cfg.get("dataset", ""),
            "arch": cfg.get("arch", ""),
            "head": cfg.get("head", ""),
            "hidden_dim": cfg.get("hidden_dim", cfg.get("layer_dims", "")),
            "setup": cfg.get("setup", ""),
            "protocol": cfg.get("protocol", cfg.get("training_mode", "")),
            "seed": _fmt(seed),
            "num_train_samples": cfg.get("num_train_samples", ""),
            "final_acc": _fmt(final_acc) if final_acc is not None else "MISSING",
            "best_acc": _fmt(fin.get("best_acc")),
            "best_acc_samples": _fmt(fin.get("best_acc_samples")),
            "n_organizes": _fmt(fin.get("n_organizes")),
            "train_wall_sec": _fmt(fin.get("train_wall_sec"), 0),
            "peak_gpu_mem_mb": _fmt(fin.get("peak_gpu_mem_mb"), 0),
            "git_commit": cfg.get("git_commit", ""),
            "git_dirty": _fmt(cfg.get("git_dirty")),
            "analysis_level": cfg.get("analysis_level", ""),
            "has_checkpoint": "yes" if has_ckpt else "NO",
            "run_dir": str(ts_dir),
        })
    return rows


def summarize(run_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Aggregate final_acc over seeds; newest finished folder per run_name; no *_smoke runs."""
    newest: Dict[str, Dict[str, Any]] = {}
    for row in run_rows:  # rows are sorted by (run_name, timestamp), so later rows are newer
        if "_smoke" in row["run_name"] or row["final_acc"] == "MISSING":
            continue
        newest[row["run_name"]] = row

    groups: Dict[tuple, List[Dict[str, Any]]] = {}
    for row in newest.values():
        key = (config_name(row["run_name"]), row["dataset"], row["setup"], row["protocol"])
        groups.setdefault(key, []).append(row)

    out = []
    for (config, dataset, setup, protocol), rows in sorted(groups.items()):
        rows = sorted(rows, key=lambda r: (r["seed"] == "", str(r["seed"]).zfill(8)))
        accs = [float(r["final_acc"]) for r in rows]
        out.append({
            "config": config,
            "dataset": dataset,
            "setup": setup,
            "protocol": protocol,
            "n_seeds": len(accs),
            "seeds": " ".join(str(r["seed"]) if r["seed"] != "" else "?" for r in rows),
            "mean_acc": f"{statistics.mean(accs):.2f}",
            "std_acc": f"{statistics.stdev(accs):.2f}" if len(accs) > 1 else "",
            "min_acc": f"{min(accs):.2f}",
            "max_acc": f"{max(accs):.2f}",
            "run_names": " ".join(r["run_name"] for r in rows),
        })
    return out


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


def _incomplete(run_rows: List[Dict[str, Any]]) -> List[str]:
    return [r["run_name"] for r in run_rows if r["final_acc"] == "MISSING" or r["has_checkpoint"] == "NO"]


def write_markdown(run_rows, summary_rows, rob_rows, path: Path) -> None:
    run_cols = ["run_name", "dataset", "arch", "head", "hidden_dim", "setup", "protocol", "seed",
                "final_acc", "best_acc", "train_wall_sec", "git_commit", "has_checkpoint", "timestamp"]
    sum_cols = ["config", "dataset", "setup", "protocol", "n_seeds", "seeds", "mean_acc", "std_acc", "min_acc", "max_acc"]
    rob_cols = ["sweep", "model", "perturbation", "level", "acc", "drop_vs_clean"]

    missing = _incomplete(run_rows)
    parts = ["# RESULT index\n", "Auto-generated by `tools/build_results_index.py` - do not hand-edit.\n"]
    if missing:
        parts.append(f"**Incomplete runs (missing final accuracy or checkpoint):** {', '.join(missing)}\n")
    parts.append("\n## Summary over seeds\n")
    parts.append("final_acc (%), newest folder per run_name, `_s<seed>` stripped to group; smoke runs excluded; "
                 "std = sample std (blank for 1 seed). Blank setup = run recorded before setups existed.\n")
    parts.append(_md_table(summary_rows, sum_cols))
    parts.append("\n## Training runs\n")
    parts.append(_md_table(run_rows, run_cols))
    parts.append("\n## Robustness sweeps\n")
    parts.append(_md_table(rob_rows, rob_cols))
    path.write_text("\n".join(parts), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--result-root", default="RESULT", help="Path to the RESULT/ folder (default: ./RESULT)")
    args = parser.parse_args()

    result_root = Path(args.result_root)
    if not result_root.is_dir():
        raise SystemExit(f"No RESULT folder at {result_root.resolve()}")

    run_rows = collect_runs(result_root)
    summary_rows = summarize(run_rows)
    rob_rows = collect_robustness(result_root)

    write_csv(run_rows, result_root / "index_runs.csv", RUN_COLUMNS)
    write_csv(summary_rows, result_root / "index_summary.csv", SUMMARY_COLUMNS)
    write_csv(rob_rows, result_root / "index_robustness.csv",
              ["sweep", "timestamp", "dataset", "model", "perturbation", "level", "acc", "drop_vs_clean"])
    write_markdown(run_rows, summary_rows, rob_rows, result_root / "INDEX.md")

    print(f"[{len(run_rows)}] training runs -> {result_root / 'index_runs.csv'}")
    print(f"[{len(summary_rows)}] configurations -> {result_root / 'index_summary.csv'}")
    print(f"[{len(rob_rows)}] robustness rows -> {result_root / 'index_robustness.csv'}")
    print(f"Summary -> {result_root / 'INDEX.md'}")
    missing = _incomplete(run_rows)
    if missing:
        print(f"\n[!] {len(missing)} run(s) look incomplete (no final_eval line or no checkpoints/final.pth): {missing}")


if __name__ == "__main__":
    main()
