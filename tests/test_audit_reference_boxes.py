"""audit_reference_boxes.py: a reference box must contain the tablet found on the raw photograph."""
import csv
import os

import numpy as np
import pandas as pd
from PIL import Image

import audit_reference_boxes as AB


def _raw(tmp_path, name, h=400, w=400, box=(100, 120, 260, 300), field=(200, 200, 200), pill=(80, 90, 200)):
    arr = np.full((h, w, 3), field, np.uint8)
    y0, x0, y1, x1 = box[1], box[0], box[3], box[2]
    Y, X = np.ogrid[:h, :w]
    cy, cx = (y0 + y1) / 2, (x0 + x1) / 2
    disc = ((Y - cy) / ((y1 - y0) / 2)) ** 2 + ((X - cx) / ((x1 - x0) / 2)) ** 2 <= 1
    arr[disc] = pill
    p = tmp_path / name
    Image.fromarray(arr).save(p)
    return str(p), disc


def test_tablet_bbox_finds_the_object_on_a_uniform_field(tmp_path):
    p, disc = _raw(tmp_path, "r.png")
    arr = np.asarray(Image.open(p).convert("RGB"))
    x0, y0, x1, y1 = AB.tablet_bbox(arr)
    ys, xs = np.nonzero(disc)
    assert abs(x0 - xs.min()) <= 2 and abs(y0 - ys.min()) <= 2 and abs(x1 - (xs.max() + 1)) <= 2 and abs(y1 - (ys.max() + 1)) <= 2


def test_box_with_margin_adds_fifteen_percent_per_side_and_clips():
    assert AB.box_with_margin((100, 120, 260, 300), 400, 400, 0.15) == (76, 93, 284, 327)
    assert AB.box_with_margin((0, 0, 100, 100), 120, 120, 0.15) == (0, 0, 115, 115)


def test_audit_flags_only_boxes_that_cut_the_tablet(tmp_path):
    good, _ = _raw(tmp_path, "good.png")
    bad, _ = _raw(tmp_path, "bad.png")
    rows = [{"crop_rel": "c/good.png", "src_rel": "good.png", "pillId": "1", "side": "top", "kind": "ref",
             "method": "classical", "x0": "76", "y0": "93", "x1": "284", "y1": "327"},
            {"crop_rel": "c/bad.png", "src_rel": "bad.png", "pillId": "2", "side": "top", "kind": "ref",
             "method": "classical", "x0": "76", "y0": "93", "x1": "180", "y1": "327"}]
    report = AB.audit(rows, str(tmp_path), min_cut_px=4)
    assert [r["class_id"] for r in report if r["cut_px"] > 4] == ["2__top"]
    bad_row = [r for r in report if r["class_id"] == "2__top"][0]
    assert bad_row["new_box"] == AB.box_with_margin(tuple(bad_row["tablet_bbox"]), 400, 400, 0.15)


def test_apply_rewrites_the_manifest_and_appends_the_corrections(tmp_path):
    man = tmp_path / "manifest.csv"
    cols = ["crop_rel", "src_rel", "pillId", "side", "kind", "method", "orig_w", "orig_h", "x0", "y0", "x1", "y1",
            "margin", "img_size", "mask_frac", "sam_score"]
    pd.DataFrame([["c/a.png", "a.jpg", "2", "top", "ref", "classical", 400, 400, 76, 93, 180, 327, 0.15, 384, 0.3, 0.9],
                  ["c/b.png", "b.jpg", "3", "top", "customer", "sam", 400, 400, 1, 2, 3, 4, 0.15, 384, 0.3, 0.9]],
                 columns=cols).to_csv(man, index=False)
    corr = tmp_path / "box_corrections.csv"
    corr.write_text("crop_rel,old_x0,old_y0,old_x1,old_y1,old_method,new_x0,new_y0,new_x1,new_y1,new_method,source\n")
    report = [{"class_id": "2__top", "crop_rel": "c/a.png", "box": [76, 93, 180, 327], "new_box": (76, 93, 284, 327),
               "cut_px": 104, "method": "classical"}]
    n = AB.apply(report, str(man), str(corr), source="test", min_cut_px=4)
    m = pd.read_csv(man)
    assert n == 1 and int(m.loc[0, "x1"]) == 284 and m.loc[0, "method"] == "audit" and int(m.loc[1, "x1"]) == 3
    c = list(csv.DictReader(open(corr)))
    assert len(c) == 1 and c[0]["new_x1"] == "284" and c[0]["old_method"] == "classical" and c[0]["source"] == "test"
