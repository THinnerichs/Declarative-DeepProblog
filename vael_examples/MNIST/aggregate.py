"""Aggregate the per-seed VAEL summaries into the numbers used in Table 1."""
import csv
import statistics
import sys
from pathlib import Path

RESULTS = Path(__file__).parent / "results"
COLUMNS = ["class_digit", "class_add", "gen_digit", "gen_add"]

rows = []
PATTERN = sys.argv[1] if len(sys.argv) > 1 else "vael_seed*/summary.csv"

for summary in sorted(RESULTS.glob(PATTERN)):
    with open(summary) as f:
        rows.append(next(csv.DictReader(f)))

print(f"{len(rows)} seeds: " + ", ".join(r["seed"] for r in rows))
for col in COLUMNS:
    values = [100 * float(r[col]) for r in rows]
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    per_seed = " ".join(f"{v:.1f}" for v in values)
    print(f"{col:12s} {mean:5.1f} +- {std:4.1f}   (seeds: {per_seed})")
print("best epochs: " + " ".join(r["best_epoch"] for r in rows))
