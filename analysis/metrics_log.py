"""
Per-organize metrics table: RESULT/<run_name>/<timestamp>/metrics.csv.

One row per organize() call (including the final catch-up organize), written
by the training engines. Replaces the per-organize PNGs of the "full"
analysis level with a few numbers per layer that are easy to plot later.

The file is rewritten from memory on every row (it is small: ~60 rows per
full run), so a column that first appears in a later row (e.g. test_acc on
an eval organize) still gets a header, with blanks in the earlier rows.
"""
import csv
from pathlib import Path
from typing import Any, Dict, List

METRICS_FILENAME = "metrics.csv"


class MetricsLog:
    def __init__(self, run_dir):
        self.path = Path(run_dir) / METRICS_FILENAME
        self.rows: List[Dict[str, Any]] = []
        self.fieldnames: List[str] = []

    def __len__(self) -> int:
        return len(self.rows)

    def log(self, row: Dict[str, Any]) -> None:
        row = dict(row)
        self.rows.append(row)
        for key in row:
            if key not in self.fieldnames:
                self.fieldnames.append(key)
        self._write()

    def _write(self) -> None:
        with open(self.path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()
            for row in self.rows:
                writer.writerow({k: _fmt(row.get(k)) for k in self.fieldnames})


def _fmt(v: Any) -> Any:
    if v is None:
        return ""
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, float):
        return f"{v:.6g}"
    return v
