"""build_cure_fewshot.py clean: duplicated photographs resolved from a frozen decision file."""
import json
import os

import pandas as pd
import pytest

import build_cure_fewshot as B


def _tiny_outputs(out):
    """Two pills, two sides each, four consumer crops per class plus one reference per class.
    Fold 0 holds pill 2, fold 1 holds pill 1. One photograph is filed under both sides of pill 1."""
    rows = []
    for pill, fold in (("1", 1), ("2", 0)):
        for side in ("bottom", "top"):
            cid = f"{pill}__{side}"
            rows.append((pill, side, "reference", f"crops/{pill}/{side}/ref/{pill}_ref_{side}.png", cid, pill))
            for k in range(4):
                rows.append((pill, side, "consumer", f"crops/{pill}/{side}/customer/{pill}_{side}_{k}.png", cid, pill))
    df = pd.DataFrame(rows, columns=["pill_id", "side", "domain", "crop_rel", "class_id", "class_id_oneside"])
    df.to_csv(os.path.join(out, "manifest_classed.csv"), index=False)
    folds = df[["pill_id", "class_id"]].drop_duplicates()
    folds["fold"] = folds["pill_id"].map({"1": 1, "2": 0})
    folds.to_csv(os.path.join(out, "folds.csv"), index=False)
    con = df[df["domain"] == "consumer"]
    pd.DataFrame({"crop_rel": con["crop_rel"], "class_id": con["class_id"],
                  "cos_m": 0.5, "leaked": False}).to_csv(os.path.join(out, "leak_list.csv"), index=False)
    pd.DataFrame(columns=["class_a", "class_b", "centroid_cos_m", "n_near_dup_images",
                          "trigger", "suggest"]).to_csv(os.path.join(out, "dedup_map.csv"), index=False)
    pd.DataFrame([{"group": 1, "keep_crop_rel": "crops/1/bottom/customer/1_bottom_0.png",
                   "drop_crop_rel": "crops/1/top/customer/1_top_0.png",
                   "keep_class": "1__bottom", "drop_class": "1__top",
                   "rule": "keep the copy under the class that sorts first",
                   "note": "same photograph filed under both sides"}]).to_csv(
        os.path.join(out, "duplicate_resolution.csv"), index=False)
    return df


@pytest.fixture
def out_dir(tmp_path, monkeypatch):
    out = str(tmp_path / "outputs")
    os.makedirs(out)
    monkeypatch.setattr(B, "OUT", out)
    monkeypatch.setattr(B, "EVAL_SAMPLE_N", 4)
    _tiny_outputs(out)
    return out


def test_clean_drops_the_recorded_copy_and_logs_it(out_dir):
    B.stage_clean()
    q = pd.read_csv(os.path.join(out_dir, "query_clean.csv"))
    tr = pd.read_csv(os.path.join(out_dir, "train_clean.csv"))
    assert "crops/1/top/customer/1_top_0.png" not in set(q["crop_rel"])
    assert "crops/1/bottom/customer/1_bottom_0.png" in set(q["crop_rel"])
    assert "crops/1/top/customer/1_top_0.png" not in set(tr["crop_rel"])
    assert len(q) == 15 and len(tr) == 9          # 16 consumers minus one; pill 1: 7 consumers + 2 refs
    log = json.load(open(os.path.join(out_dir, "removal_log.json")))
    assert log["n_duplicate_groups"] == 1
    assert log["n_duplicate_images_dropped"] == 1
    assert log["n_leaked_removed"] == 0 and log["leak_frac_before"] == 0.0   # a drop is not a leak
    assert log["duplicate_rule"] == "keep the copy under the class that sorts first"
    summary = json.load(open(os.path.join(out_dir, "cure_processed_summary.json")))
    assert summary["duplicates_dropped"] == 1
    assert summary["n_images_total"] == 19
    assert summary["query_total"] == 15
    assert summary["train_fold0_heldout"]["train_images"] == 9


def test_clean_without_resolution_file_changes_nothing(out_dir):
    os.remove(os.path.join(out_dir, "duplicate_resolution.csv"))
    B.stage_clean()
    q = pd.read_csv(os.path.join(out_dir, "query_clean.csv"))
    assert len(q) == 16
    log = json.load(open(os.path.join(out_dir, "removal_log.json")))
    assert log["n_duplicate_groups"] == 0 and log["n_duplicate_images_dropped"] == 0


def test_clean_refuses_a_drop_whose_keep_copy_is_absent(out_dir):
    p = os.path.join(out_dir, "duplicate_resolution.csv")
    d = pd.read_csv(p); d.loc[0, "keep_crop_rel"] = "crops/nowhere.png"; d.to_csv(p, index=False)
    with pytest.raises(SystemExit):
        B.stage_clean()
