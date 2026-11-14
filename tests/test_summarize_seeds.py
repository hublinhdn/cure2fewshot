"""training/summarize_seeds.py: one CSV row per backbone from the per run JSON files."""
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "training"))
import summarize_seeds as SS  # noqa: E402


def _run(d, bb, seed, sel, last):
    json.dump({"backbone": bb, "seed": seed, "selected": {"epoch": 70, **sel}, "last": {"epoch": 80, **last},
               "seconds": 100.0, "history": []}, open(d / f"{bb}_seed{seed}.json", "w"))


def test_summary_has_mean_and_sd_for_selected_and_last(tmp_path):
    for seed, m in ((42, 0.70), (123, 0.72), (223, 0.71)):
        _run(tmp_path, "resnet18_tv", seed, {"mAP": m, "Rank-1": m - 0.1, "CMC@5": 0.9},
             {"mAP": m - 0.01, "Rank-1": m - 0.11, "CMC@5": 0.89})
    _run(tmp_path, "levit_128s_timm", 42, {"mAP": 0.1, "Rank-1": 0.05, "CMC@5": 0.12},
         {"mAP": 0.1, "Rank-1": 0.05, "CMC@5": 0.12})
    out = tmp_path / "baseline_seeds.csv"
    sweep = pd.DataFrame({"backbone": ["resnet18_tv", "levit_128s_timm"], "mAP": [0.7087, 0.0932],
                          "Rank-1": [0.5592, 0.0437], "CMC@5": [0.881, 0.123]})
    sweep.to_csv(tmp_path / "sweep.csv", index=False)
    d = SS.summarize(str(tmp_path), str(out), str(tmp_path / "sweep.csv"))
    r = d[d["backbone"] == "resnet18_tv"].iloc[0]
    assert r["n_seeds"] == 3 and r["seeds"] == "42;123;223"
    assert abs(r["mAP_mean"] - 0.71) < 1e-9 and abs(r["mAP_sd"] - 0.01) < 1e-9
    assert abs(r["mAP_last_mean"] - 0.70) < 1e-9
    assert abs(r["mAP_sweep"] - 0.7087) < 1e-9
    assert abs(r["mAP_seed42"] - 0.70) < 1e-9
    lv = d[d["backbone"] == "levit_128s_timm"].iloc[0]
    assert lv["n_seeds"] == 1 and pd.isna(lv["mAP_sd"])
    assert list(pd.read_csv(out).columns) == list(d.columns)
