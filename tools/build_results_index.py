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

Run status: final_eval.json has "status": "ok" or "failed" for runs from
Plan 02c on (a failed run also records its "error"). Failed runs show as
`failed` - not as incomplete/MISSING - and are left out of the summary.
Older runs without a status are treated as before: "ok" when they have a
final accuracy and a checkpoint, otherwise "incomplete".

Safe to re-run any time after new jobs finish - it only reads, never
modifies anything under RESULT/ except these four output files, which it
always rebuilds from scratch (no stale rows left behind from a
renamed/deleted run).

--update-doc PATH also rewrites the block between
<!-- AUTO-RESULTS:BEGIN --> and <!-- AUTO-RESULTS:END --> in PATH
(docs/results.md): one table of mean +- std over seeds per configuration
(smoke, failed and pre-setup runs left out), plus a short robustness table
when non-smoke robustness sweeps exist for those runs. Text outside the
markers is never touched.

Usage (from the repo root, same venv you train with):
    python tools/build_results_index.py
    python tools/build_results_index.py --update-doc docs/results.md
    python tools/build_results_index.py --result-root My_Notes_Aishani/results/RESULT
"""
import argparse
import csv
import json
import re
import statistics
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

SEED_SUFFIX = re.compile(r"_s(\d+)$")

RUN_COLUMNS = [
    "run_name", "timestamp", "status", "dataset", "arch", "head", "hidden_dim", "setup", "protocol", "seed",
    "num_train_samples", "final_acc", "best_acc", "best_acc_samples", "n_organizes",
    "train_wall_sec", "peak_gpu_mem_mb", "git_commit", "git_dirty", "analysis_level",
    "has_checkpoint", "error", "run_dir",
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
        failed = fin.get("status") == "failed"
        final_acc = fin.get("final_acc")
        if final_acc is None and not failed:
            final_acc = _parse_final_acc(ts_dir / "eval_log.txt")
        seed = cfg.get("seed")
        if seed is None:
            m = SEED_SUFFIX.search(run_name)
            seed = int(m.group(1)) if m else None
        has_ckpt = (ts_dir / "checkpoints" / "final.pth").exists()
        if failed:
            status = "failed"
        elif final_acc is None or not has_ckpt:
            status = "incomplete"
        else:
            status = "ok"
        if failed:
            final_acc_cell = "failed"
        else:
            final_acc_cell = _fmt(final_acc) if final_acc is not None else "MISSING"
        rows.append({
            "run_name": run_name,
            "timestamp": ts_dir.name,
            "status": status,
            "dataset": cfg.get("dataset", ""),
            "arch": cfg.get("arch", ""),
            "head": cfg.get("head", ""),
            "hidden_dim": cfg.get("hidden_dim", cfg.get("layer_dims", "")),
            "setup": cfg.get("setup", ""),
            "protocol": cfg.get("protocol", cfg.get("training_mode", "")),
            "seed": _fmt(seed),
            "num_train_samples": cfg.get("num_train_samples", ""),
            "final_acc": final_acc_cell,
            "best_acc": _fmt(fin.get("best_acc")),
            "best_acc_samples": _fmt(fin.get("best_acc_samples")),
            "n_organizes": _fmt(fin.get("n_organizes")),
            "train_wall_sec": _fmt(fin.get("train_wall_sec"), 0),
            "peak_gpu_mem_mb": _fmt(fin.get("peak_gpu_mem_mb"), 0),
            "git_commit": cfg.get("git_commit", ""),
            "git_dirty": _fmt(cfg.get("git_dirty")),
            "analysis_level": cfg.get("analysis_level", ""),
            "has_checkpoint": "yes" if has_ckpt else "NO",
            "error": fin.get("error", "") if failed else "",
            "run_dir": str(ts_dir),
        })
    return rows


def summarize(run_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Aggregate final_acc over seeds; newest finished folder per run_name; no *_smoke or failed runs."""
    newest: Dict[str, Dict[str, Any]] = {}
    for row in run_rows:  # rows are sorted by (run_name, timestamp), so later rows are newer
        if "_smoke" in row["run_name"] or row["status"] == "failed" or row["final_acc"] == "MISSING":
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
    return [r["run_name"] for r in run_rows if r["status"] == "incomplete"]


def _failed(run_rows: List[Dict[str, Any]]) -> List[str]:
    return [f"{r['run_name']} ({r['timestamp']})" for r in run_rows if r["status"] == "failed"]


def write_markdown(run_rows, summary_rows, rob_rows, path: Path) -> None:
    run_cols = ["run_name", "status", "dataset", "arch", "head", "hidden_dim", "setup", "protocol", "seed",
                "final_acc", "best_acc", "train_wall_sec", "git_commit", "has_checkpoint", "timestamp"]
    sum_cols = ["config", "dataset", "setup", "protocol", "n_seeds", "seeds", "mean_acc", "std_acc", "min_acc", "max_acc"]
    rob_cols = ["sweep", "model", "perturbation", "level", "acc", "drop_vs_clean"]

    missing = _incomplete(run_rows)
    parts = ["# RESULT index\n", "Auto-generated by `tools/build_results_index.py` - do not hand-edit.\n"]
    if missing:
        parts.append(f"**Incomplete runs (missing final accuracy or checkpoint):** {', '.join(missing)}\n")
    failed = _failed(run_rows)
    if failed:
        parts.append(f"**Failed runs (raised during training; error in index_runs.csv):** {', '.join(failed)}\n")
    parts.append("\n## Summary over seeds\n")
    parts.append("final_acc (%), newest folder per run_name, `_s<seed>` stripped to group; smoke and failed runs excluded; "
                 "std = sample std (blank for 1 seed). Blank setup = run recorded before setups existed.\n")
    parts.append(_md_table(summary_rows, sum_cols))
    parts.append("\n## Training runs\n")
    parts.append(_md_table(run_rows, run_cols))
    parts.append("\n## Robustness sweeps\n")
    parts.append(_md_table(rob_rows, rob_cols))
    path.write_text("\n".join(parts), encoding="utf-8")


# ---------------------------------------------------------------------------
# docs/results.md auto block (--update-doc)
# ---------------------------------------------------------------------------
DOC_BEGIN = "<!-- AUTO-RESULTS:BEGIN -->"
DOC_END = "<!-- AUTO-RESULTS:END -->"
DATASET_ORDER = ["mnist", "fashion_mnist", "kmnist", "cifar10", "cifar10_gray"]
DOC_ROBUSTNESS_CELLS = [
    ("clean", 0.0, "Clean"),
    ("gaussian", 0.5, "Gaussian noise \u03c3 0.5"),
    ("occlusion", 14.0, "Occlusion 14\u00d714"),
    ("shift", 4.0, "Shift \u00b14 px"),
]


def _mean_std(values: List[float]) -> str:
    if len(values) > 1:
        return f"{statistics.mean(values):.2f} \u00b1 {statistics.stdev(values):.2f}"
    return f"{statistics.mean(values):.2f}"


def doc_groups(run_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Configurations for the docs table: newest folder per run_name, without
    smoke runs, failed/incomplete-accuracy runs and old runs with no setup.
    Grouped by run_name minus _s<seed> plus dataset/arch/head/setup/protocol,
    so two different experiments never get averaged together.
    """
    newest: Dict[str, Dict[str, Any]] = {}
    for row in run_rows:  # sorted by (run_name, timestamp): later rows are newer
        if ("_smoke" in row["run_name"] or row["status"] == "failed"
                or row["final_acc"] in ("MISSING", "failed") or not row["setup"]):
            continue
        newest[row["run_name"]] = row

    groups: Dict[tuple, List[Dict[str, Any]]] = {}
    for row in newest.values():
        key = (row["dataset"], row["arch"], row["head"], row["setup"], row["protocol"], config_name(row["run_name"]))
        groups.setdefault(key, []).append(row)

    def sort_key(key):
        ds = key[0]
        return (DATASET_ORDER.index(ds) if ds in DATASET_ORDER else len(DATASET_ORDER), ds, *key[1:])

    out = []
    for key in sorted(groups, key=sort_key):
        dataset, arch, head, setup, protocol, config = key
        rows = sorted(groups[key], key=lambda r: (r["seed"] == "", str(r["seed"]).zfill(8)))
        accs = [float(r["final_acc"]) for r in rows]
        out.append({
            "dataset": dataset, "arch": arch, "head": head, "setup": setup, "protocol": protocol,
            "config": config,
            "seeds": ", ".join(str(r["seed"]) if r["seed"] != "" else "?" for r in rows),
            "n": len(rows),
            "acc": _mean_std(accs),
            "range": f"{min(accs):.2f}\u2013{max(accs):.2f}",
            "run_names": [r["run_name"] for r in rows],
        })
    return out


def _doc_robustness(groups: List[Dict[str, Any]], rob_rows: List[Dict[str, Any]]) -> str:
    """Per-dataset table: rows = the 4 headline corruptions, columns = configurations."""
    # newest non-smoke sweep per model (= run_name)
    newest_ts: Dict[str, str] = {}
    for r in rob_rows:
        if r["sweep"].endswith("_smoke"):
            continue
        if r["model"] not in newest_ts or r["timestamp"] > newest_ts[r["model"]]:
            newest_ts[r["model"]] = r["timestamp"]
    acc: Dict[tuple, float] = {}
    for r in rob_rows:
        if r["sweep"].endswith("_smoke") or newest_ts.get(r["model"]) != r["timestamp"]:
            continue
        try:
            acc[(r["model"], r["perturbation"], float(r["level"]))] = float(r["acc"])
        except (TypeError, ValueError):
            continue

    parts = []
    datasets = list(dict.fromkeys(g["dataset"] for g in groups))
    for ds in datasets:
        cols = []
        for g in groups:
            if g["dataset"] != ds:
                continue
            if not any((rn, p, lv) in acc for rn in g["run_names"] for p, lv, _ in DOC_ROBUSTNESS_CELLS):
                continue
            cols.append(g)
        if not cols:
            continue
        labels = [f"{g['arch']} + {g['head']} ({g['setup']}, {g['protocol']})" for g in cols]
        lines = [f"**{ds}** - accuracy (%) under corruption of the test images, mean \u00b1 std over seeds",
                 "",
                 "| Corruption | " + " | ".join(labels) + " |",
                 "|---|" + "---|" * len(labels)]
        for p, lv, name in DOC_ROBUSTNESS_CELLS:
            cells = []
            for g in cols:
                vals = [acc[(rn, p, lv)] for rn in g["run_names"] if (rn, p, lv) in acc]
                if not vals:
                    cells.append("")
                    continue
                cell = _mean_std(vals)
                if len(vals) != g["n"]:
                    cell += f" (n={len(vals)})"
                cells.append(cell)
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


def doc_results_block(run_rows: List[Dict[str, Any]], rob_rows: List[Dict[str, Any]], result_root: Path) -> str:
    groups = doc_groups(run_rows)
    n_runs = sum(g["n"] for g in groups)
    lines = [f"Generated {date.today().isoformat()} from `{result_root}` \u00b7 {n_runs} run{'' if n_runs == 1 else 's'}", ""]
    if not groups:
        lines.append("_No finished non-smoke runs with a setup yet._")
        return "\n".join(lines)
    lines.append("| Dataset | Architecture | Head | Setup | Protocol | Seeds | Accuracy (mean \u00b1 std) | Min\u2013max | Runs |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for g in groups:
        lines.append(f"| {g['dataset']} | {g['arch']} | {g['head']} | {g['setup']} | {g['protocol']} | "
                     f"{g['seeds']} | {g['acc']} | {g['range']} | `{g['config']}` |")
    rob = _doc_robustness(groups, rob_rows)
    if rob:
        lines += ["", "### Robustness", "", rob]
    return "\n".join(lines)


def update_doc(path: Path, block: str) -> None:
    """Replace the text between the AUTO-RESULTS markers in `path`; nothing outside them changes."""
    if not path.exists():
        raise SystemExit(f"--update-doc: {path} does not exist")
    text = path.read_text(encoding="utf-8")
    if text.count(DOC_BEGIN) != 1 or text.count(DOC_END) != 1:
        raise SystemExit(f"--update-doc: {path} must contain exactly one '{DOC_BEGIN}' and one '{DOC_END}' line")
    start = text.index(DOC_BEGIN) + len(DOC_BEGIN)
    end = text.index(DOC_END)
    if end < start:
        raise SystemExit(f"--update-doc: in {path}, '{DOC_END}' comes before '{DOC_BEGIN}'")
    path.write_text(text[:start] + "\n" + block + "\n" + text[end:], encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--result-root", default="RESULT", help="Path to the RESULT/ folder (default: ./RESULT)")
    parser.add_argument("--update-doc", default=None, metavar="PATH",
                        help="Also rewrite the AUTO-RESULTS block in this markdown file (e.g. docs/results.md)")
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
    if args.update_doc:
        update_doc(Path(args.update_doc), doc_results_block(run_rows, rob_rows, result_root))
        print(f"Results block -> {args.update_doc}")
    missing = _incomplete(run_rows)
    if missing:
        print(f"\n[!] {len(missing)} run(s) look incomplete (no final_eval line or no checkpoints/final.pth): {missing}")
    failed = [r for r in run_rows if r["status"] == "failed"]
    if failed:
        print(f"\n[!] {len(failed)} run(s) failed:")
        for r in failed:
            print(f"    {r['run_name']} ({r['timestamp']}): {r['error']}")


if __name__ == "__main__":
    main()
