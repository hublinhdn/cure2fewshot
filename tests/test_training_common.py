"""training/common.py: the recipe table, the data frames and the retrieval metric, without torch."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "training"))
import common as T  # noqa: E402


def test_recipe_small_cnn():
    r = T.recipe_for("resnet18_tv")
    assert r["group"] == "small"
    assert (r["n_classes_batch"], r["n_samples"], r["accumulation_steps"]) == (16, 2, 4)
    assert (r["lr_backbone"], r["lr_head"]) == (5e-5, 5e-4)
    assert r["arcface_margin"] == 0.20 and r["w_csce"] == 0.2
    assert r["total_epochs"] == 80 and r["warmup_epochs"] == 5
    assert r["img_size"] == 224 and r["freeze_epochs"] == 2 and r["clip_grad"] == 3.0
    assert r["weight_decay"] == 5e-2


def test_recipe_medium_and_lightweight_vit_and_massive():
    m = T.recipe_for("convnext_small_timm")
    assert m["group"] == "medium" and m["total_epochs"] == 60 and m["img_size"] == 384
    assert m["freeze_epochs"] == 0 and m["arcface_margin"] == 0.35 and m["clip_grad"] == 5.0
    assert (m["n_classes_batch"], m["n_samples"], m["accumulation_steps"]) == (8, 2, 8)
    lv = T.recipe_for("levit_128s_timm")
    assert lv["group"] == "lightweight_vit" and lv["warmup_epochs"] == 10 and lv["total_epochs"] == 80
    assert lv["arcface_margin"] == 0.25 and lv["w_csce"] == 0.10 and lv["lr_head"] == 2e-4
    assert lv["img_size"] == 224 and lv["freeze_epochs"] == 2
    ev = T.recipe_for("eva02_large_patch14_448_timm")
    assert ev["group"] == "super_large" and ev["img_size"] == 392 and ev["weight_decay"] == 0.1
    assert abs(ev["lr_backbone"] - 2e-5 * 0.3) < 1e-12 and ev["freeze_epochs"] == 0
    assert ev["accumulation_steps"] == 20
    cl = T.recipe_for("convnext_large_timm")
    assert cl["freeze_epochs"] == 4 and cl["drop_path"] == 0.3
    gh = T.recipe_for("ghostnet_100_timm")
    assert (gh["lr_backbone"], gh["lr_head"]) == (2e-5, 2e-4)
    mb = T.recipe_for("mobilenet_v3_large_tv")
    assert (mb["lr_backbone"], mb["lr_head"]) == (4e-5, 4e-4)
    with pytest.raises(KeyError):
        T.recipe_for("not_a_backbone_timm")


def test_every_sweep_backbone_has_a_recipe():
    for b in T.SWEEP_BACKBONES:
        assert T.recipe_for(b)["total_epochs"] in (60, 70, 80)
    assert len(T.SWEEP_BACKBONES) == 27


def test_build_frames_maps_training_references_to_hardened_images(tmp_path):
    crops = tmp_path / "crops"; hard = tmp_path / "hard"; data = tmp_path / "data"
    for d in (crops, hard, data):
        d.mkdir()
    (hard / "1__top.png").write_bytes(b"x")
    pd.DataFrame({"crop_rel": ["data/processed/cure_crops/1/top/customer/a.png",
                               "data/processed/cure_crops/1/top/ref/1_ref_top.png",
                               "data/processed/cure_crops/3/top/ref/3_ref_top.png"],
                  "label": ["1__top", "1__top", "3__top"],
                  "domain": ["consumer", "reference", "reference"], "fold": [1, 1, 1]}).to_csv(
        data / "train_clean.csv", index=False)
    pd.DataFrame({"crop_rel": ["data/processed/cure_crops/2/top/customer/b.png"], "label": ["2__top"],
                  "fold": [0]}).to_csv(data / "query_eval_fold0.csv", index=False)
    pd.DataFrame({"crop_rel": [str(hard / "2__top.png")], "label": ["2__top"], "fold": [0]}).to_csv(
        data / "gallery.csv", index=False)
    tr, ev, n, labels = T.build_frames(str(data), str(crops), str(hard), gallery=str(data / "gallery.csv"))
    assert n == 3 and labels == ["1__top", "2__top", "3__top"]
    assert tr.loc[tr["is_ref"] == 1, "image_path"].tolist() == [
        str(hard / "1__top.png"), str(crops / "data/processed/cure_crops/3/top/ref/3_ref_top.png")]
    assert tr["label_idx"].tolist() == [0, 0, 2]
    assert ev["is_ref"].tolist() == [1, 0] and ev["label_idx"].tolist() == [1, 1]


def test_retrieval_metrics_rank_based():
    gf = np.eye(3, dtype=np.float32); gl = np.array([0, 1, 2])
    qf = np.array([[1, 0, 0], [0, 0.9, 0.1], [0.1, 0, 0.9], [0, 1, 0]], dtype=np.float32)
    ql = np.array([0, 2, 0, 5])          # query 1: correct at rank 2; query 2: rank 2; query 3: class absent
    m = T.retrieval_metrics(qf, ql, gf, gl)
    assert m["n_query"] == 4 and m["n_query_without_gallery"] == 1
    assert abs(m["mAP"] - (1 + 0.5 + 0.5 + 0) / 4) < 1e-9        # an absent class is a miss at every rank
    assert abs(m["Rank-1"] - 0.25) < 1e-9 and abs(m["CMC@5"] - 0.75) < 1e-9
