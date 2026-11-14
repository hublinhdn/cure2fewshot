#!/usr/bin/env python3
"""summarize_seeds.py — one row per backbone from the JSON files train_baseline.py writes.

  python training/summarize_seeds.py runs_seed expected/baseline_seeds.csv --sweep expected/retrieval_baselines.csv

Columns: the number of seeds, the seeds, mean and sample standard deviation of mAP, Rank-1 and
CMC@5 under the sweep's selection rule (best scored epoch), the same for the last epoch, the
seed 42 value, and the sweep's single seed number when --sweep is given.
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

METRICS = ["mAP", "Rank-1", "CMC@5"]


def summarize(run_dir, out_path, sweep_csv=None):
    runs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(run_dir, "*_seed*.json")))]
    runs = [r for r in runs if "selected" in r]
    sweep = pd.read_csv(sweep_csv).set_index("backbone") if sweep_csv else None
    rows = []
    for bb in sorted({r["backbone"] for r in runs}):
        rs = sorted([r for r in runs if r["backbone"] == bb], key=lambda r: r["seed"])
        row = {"backbone": bb, "n_seeds": len(rs), "seeds": ";".join(str(r["seed"]) for r in rs)}
        for m in METRICS:
            sel = np.array([r["selected"][m] for r in rs]); last = np.array([r["last"][m] for r in rs])
            row[f"{m}_mean"] = float(sel.mean()); row[f"{m}_sd"] = float(sel.std(ddof=1)) if len(rs) > 1 else np.nan
            row[f"{m}_last_mean"] = float(last.mean()); row[f"{m}_last_sd"] = float(last.std(ddof=1)) if len(rs) > 1 else np.nan
        for m in METRICS:
            s42 = [r for r in rs if r["seed"] == 42]
            row[f"{m}_seed42"] = float(s42[0]["selected"][m]) if s42 else np.nan
        for m in METRICS:
            row[f"{m}_sweep"] = float(sweep.loc[bb, m]) if (sweep is not None and bb in sweep.index) else np.nan
        row["selected_epochs"] = ";".join(str(r["selected"]["epoch"]) for r in rs)
        row["seconds_mean"] = float(np.mean([r.get("seconds", np.nan) for r in rs]))
        rows.append(row)
    d = pd.DataFrame(rows)
    d.to_csv(out_path, index=False, float_format="%.4f")
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run_dir"); ap.add_argument("out"); ap.add_argument("--sweep", default=None)
    a = ap.parse_args()
    d = summarize(a.run_dir, a.out, a.sweep)
    cols = ["backbone", "n_seeds", "mAP_mean", "mAP_sd", "mAP_last_mean", "mAP_sweep"]
    print(d[[c for c in cols if c in d.columns]].to_string(index=False))
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
