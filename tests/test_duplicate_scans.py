"""cure_fitness_check.py: image level duplicate scans that do not gate on a class centroid."""
import numpy as np
import pandas as pd

import cure_fitness_check as F


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def _toy():
    """Six crops, two classes. A_c1 and B_c1 are the same photograph filed under both classes;
    B_c2 is a consumer shot almost identical to its own reference; everything else is far apart."""
    df = pd.DataFrame({
        "crop_rel": ["A_c1.png", "A_c2.png", "A_ref.png", "B_c1.png", "B_c2.png", "B_ref.png"],
        "class_id": ["A", "A", "A", "B", "B", "B"],
        "domain": ["consumer", "consumer", "reference", "consumer", "consumer", "reference"],
        "pill_id": ["A", "A", "A", "B", "B", "B"],
    })
    base = {
        "A_c1": [1, 0, 0, 0], "A_c2": [0, 0, 1, 0], "A_ref": [0, 1, 0, 0],
        "B_c1": [1, 0, 0, 0], "B_c2": [0, 0, 0, 1], "B_ref": [0.1, 0, 0, 1],
    }
    e = np.stack([_unit(base[c[:-4]]) for c in df["crop_rel"]])
    return df, {"m1": e, "m2": e.copy()}


def test_near_duplicate_scan_counts_cross_and_same_class_pairs():
    df, embs = _toy()
    summary, pairs = F.near_duplicate_scan(df, embs, thr=0.99)
    assert summary["threshold"] == 0.99
    assert summary["n_pairs"] == 2
    assert summary["n_pairs_cross_class"] == 1
    assert summary["n_pairs_same_class"] == 1
    assert summary["per_model"]["m1"]["n_pairs"] == 2
    assert summary["class_pairs"] == [{"class_a": "A", "class_b": "B", "n_pairs": 1}]
    cross = pairs[~pairs["same_class"]]
    assert list(cross[["crop_a", "crop_b"]].iloc[0]) == ["A_c1.png", "B_c1.png"]
    same = pairs[pairs["same_class"]]
    assert list(same[["crop_a", "crop_b"]].iloc[0]) == ["B_c2.png", "B_ref.png"]
    assert abs(float(cross["cos_m1"].iloc[0]) - 1.0) < 1e-6
    assert summary["cross_class_pairs"][0]["classes"] == ["A", "B"]


def test_dedup_scan_flags_class_pair_from_image_pairs_without_centroid_gate():
    df, embs = _toy()
    _, pairs = F.near_duplicate_scan(df, embs, thr=0.99)
    summary, dm = F.dedup_scan(df, embs, cen_thr=0.97, img_thr=0.99, min_pairs=1,
                               near_pairs=pairs)
    row = dm.iloc[0]
    assert (row["class_a"], row["class_b"]) == ("A", "B")
    assert row["trigger"] == "images"
    assert row["suggest"] == "resolve_images"
    assert int(row["n_near_dup_images"]) == 1
    assert summary["n_flagged_pairs"] == 1
    assert summary["n_suggest_resolve_images"] == 1
    assert summary["n_suggest_merge"] == 0


def test_dedup_scan_centroid_trigger_still_suggests_merge_when_images_agree():
    df, embs = _toy()
    # pull every consumer of B onto A's consumers so the centroids agree in both probes
    for e in embs.values():
        e[4] = e[1]
    _, pairs = F.near_duplicate_scan(df, embs, thr=0.99)
    _, dm = F.dedup_scan(df, embs, cen_thr=0.97, img_thr=0.99, min_pairs=2, near_pairs=pairs)
    row = dm.iloc[0]
    assert row["trigger"] == "both"
    assert row["suggest"] == "merge"
    assert int(row["n_near_dup_images"]) == 2


def test_dedup_scan_centroid_only_is_review():
    df, embs = _toy()
    for e in embs.values():                      # two tight clusters, no pair above 0.99
        e[0] = _unit([1, 0, 0, 0]); e[1] = _unit([1, 0.4, 0, 0])
        e[3] = _unit([1, 0.2, 0, 0]); e[4] = _unit([1, 0.6, 0, 0])
    _, pairs = F.near_duplicate_scan(df, embs, thr=0.99)
    _, dm = F.dedup_scan(df, embs, cen_thr=0.97, img_thr=0.99, min_pairs=3, near_pairs=pairs)
    assert len(dm) == 1
    assert dm.iloc[0]["trigger"] == "centroid"
    assert dm.iloc[0]["suggest"] == "review"


def test_cross_partition_scan_reports_nearest_training_image_per_query():
    df, embs = _toy()
    train = ["A_c1.png", "A_c2.png", "A_ref.png"]
    evalq = ["B_c1.png", "B_c2.png"]
    summary, top = F.cross_partition_scan(df, embs, train, evalq,
                                          thresholds=(0.985, 0.97), topk=5)
    assert summary["n_train"] == 3 and summary["n_eval"] == 2
    assert summary["class_overlap"] == 0
    m = summary["per_model"]["m1"]
    assert abs(m["max"] - 1.0) < 1e-6
    assert m["n_at_or_above"]["0.985"] == 1
    assert m["n_at_or_above"]["0.97"] == 1
    first = top.iloc[0]
    assert (first["eval_crop"], first["train_crop"]) == ("B_c1.png", "A_c1.png")
    assert (first["eval_label"], first["train_label"]) == ("B", "A")


def test_nearest_query_reference_pairs_sorted_by_best_probe():
    df, embs = _toy()
    for e in embs.values():
        e[1] = _unit([0, 1, 0.05, 0])      # A_c2 almost its own reference A_ref
    top = F.nearest_query_reference_pairs(df, embs, topk=2)
    assert list(top.columns[:3]) == ["crop_rel", "reference_crop_rel", "class_id"]
    assert top.iloc[0]["crop_rel"] == "A_c2.png"
    assert top.iloc[0]["reference_crop_rel"] == "A_ref.png"
    assert top.iloc[0]["cos_max"] > top.iloc[1]["cos_max"]


def test_drop_reference_rows_keeps_consumers_and_aligns_embeddings():
    df, embs = _toy()
    df2, embs2 = F.drop_reference_rows(df, embs, {"A"})
    assert len(df2) == 5 and "A_ref.png" not in set(df2["crop_rel"])
    assert embs2["m1"].shape[0] == 5
    i = list(df2["crop_rel"]).index("B_ref.png")
    assert np.allclose(embs2["m1"][i], embs["m1"][5])


def test_out_name_with_tag():
    assert F._out_name("fitness_report.json", "") == "fitness_report.json"
    assert F._out_name("fitness_report.json", "nosuspect") == "fitness_report_nosuspect.json"
