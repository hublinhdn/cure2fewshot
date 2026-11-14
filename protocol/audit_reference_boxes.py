#!/usr/bin/env python3
"""audit_reference_boxes.py — mỗi box reference trong manifest phải chứa trọn viên thuốc trên ảnh gốc.

Vì sao có script này. Box của 330 reference được sinh bởi một luật cổ điển trên nền tổng hợp đồng
nhất (|xám - trung vị biên| > 22, thành phần liên thông lớn nhất, lề 15 %). Với viên nhạt màu và
viên nang hai màu, luật đó lấy một phần viên, nên box cắt viên và crop thiếu một phần (KNOWN_ISSUES.md
mục 12). Script này tìm viên trên ảnh gốc bằng khoảng cách màu RGB tới trung vị vành biên, ngưỡng
thấp hơn và tính trên cả ba kênh, rồi so với box đã ghi.

  audit  : đọc manifest, in mỗi reference mà box không chứa viên, ghi JSON báo cáo kèm box đề nghị
  apply  : ghi box đề nghị vào manifest (method = audit) và nối vào protocol/frozen/box_corrections.csv

  CURE_RAW_ROOT=/data python protocol/audit_reference_boxes.py audit --out outputs/reference_box_audit.json
  CURE_RAW_ROOT=/data python protocol/audit_reference_boxes.py apply --report outputs/reference_box_audit.json

Sau `apply`, xóa crop của các reference đổi box rồi chạy lại `make_crops.py build` và `verify`.
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
from PIL import Image
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
MANIFEST = os.environ.get("CURE_MANIFEST", os.path.join(HERE, "frozen", "cure_crops_manifest.curated.csv"))
CORRECTIONS = os.path.join(HERE, "frozen", "box_corrections.csv")
RAW_ROOT = os.environ.get("CURE_RAW_ROOT")
THR = 18            # tổng |RGB - nền| trên ba kênh; luật cũ dùng 22 trên một kênh xám
MARGIN = 0.15       # lề mỗi bên, theo tỷ lệ kích thước viên (quy ước của manifest)
MIN_CUT_PX = 4      # bỏ qua lệch dưới mức này (răng cưa biên)


def tablet_bbox(arr, thr=THR):
    """(x0, y0, x1, y1) bao quanh thành phần liên thông lớn nhất khác màu nền; nền = trung vị
    vành biên rộng 1 % ảnh. Mở hình thái hai bước để bỏ chấm lạc, lấp lỗ bên trong viên."""
    a = arr.astype(np.int16)
    H, W = a.shape[:2]
    w = max(8, H // 100)
    ring = np.concatenate([a[:w].reshape(-1, 3), a[-w:].reshape(-1, 3),
                           a[:, :w].reshape(-1, 3), a[:, -w:].reshape(-1, 3)])
    bg = np.median(ring, axis=0)
    m = np.abs(a - bg).sum(2) > thr
    m = ndimage.binary_opening(m, iterations=2)
    lab, n = ndimage.label(m)
    if n == 0:
        return None
    sizes = ndimage.sum_labels(m, lab, index=np.arange(1, n + 1))
    big = ndimage.binary_fill_holes(lab == (int(np.argmax(sizes)) + 1))
    ys, xs = np.nonzero(big)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def box_with_margin(bbox, W, H, margin=MARGIN):
    x0, y0, x1, y1 = bbox
    mw, mh = int(round(margin * (x1 - x0))), int(round(margin * (y1 - y0)))
    return (max(0, x0 - mw), max(0, y0 - mh), min(W, x1 + mw), min(H, y1 + mh))


def audit(rows, raw_root, min_cut_px=MIN_CUT_PX, thr=THR):
    out = []
    for r in rows:
        if r["kind"] != "ref":
            continue
        cid = f"{r['pillId']}__{r['side']}"
        arr = np.asarray(Image.open(os.path.join(raw_root, r["src_rel"])).convert("RGB"))
        H, W = arr.shape[:2]
        box = [int(r[k]) for k in ("x0", "y0", "x1", "y1")]
        tb = tablet_bbox(arr, thr)
        if tb is None:
            out.append({"class_id": cid, "crop_rel": r["crop_rel"], "method": r["method"], "box": box,
                        "tablet_bbox": None, "cut_px": 0, "new_box": None, "note": "no object found"})
            continue
        cut = max(box[0] - tb[0], 0) + max(box[1] - tb[1], 0) + max(tb[2] - box[2], 0) + max(tb[3] - box[3], 0)
        out.append({"class_id": cid, "crop_rel": r["crop_rel"], "method": r["method"], "box": box,
                    "tablet_bbox": list(tb), "cut_px": int(cut),
                    "new_box": box_with_margin(tb, W, H) if cut > min_cut_px else None})
    return out


def apply(report, manifest_path, corrections_path, source, min_cut_px=MIN_CUT_PX):
    fix = {r["crop_rel"]: r for r in report if r["cut_px"] > min_cut_px and r.get("new_box")}
    with open(manifest_path, newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        rows, cols = list(rd), rd.fieldnames
    n = 0
    with open(corrections_path, "a", newline="", encoding="utf-8") as cf:
        w = csv.writer(cf)
        for r in rows:
            fx = fix.get(r["crop_rel"])
            if not fx:
                continue
            nb = fx["new_box"]
            w.writerow([r["crop_rel"], r["x0"], r["y0"], r["x1"], r["y1"], r["method"],
                        nb[0], nb[1], nb[2], nb[3], "audit", source])
            r["x0"], r["y0"], r["x1"], r["y1"] = (str(v) for v in nb)
            r["method"] = "audit"
            n += 1
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=cols)
        wr.writeheader()
        wr.writerows(rows)
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=["audit", "apply"])
    ap.add_argument("--out", default=None, help="audit: JSON báo cáo")
    ap.add_argument("--report", default=None, help="apply: JSON báo cáo của audit")
    ap.add_argument("--source", default="reference box audit", help="apply: ghi vào cột source")
    args = ap.parse_args()
    if not RAW_ROOT:
        raise SystemExit("[box audit] thiếu CURE_RAW_ROOT")
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if args.stage == "audit":
        rep = audit(rows, RAW_ROOT)
        flagged = [r for r in rep if r["cut_px"] > MIN_CUT_PX]
        print(f"[box audit] {len(rep)} reference; box không chứa viên (lệch > {MIN_CUT_PX} px): {len(flagged)}")
        for r in sorted(flagged, key=lambda r: -r["cut_px"]):
            print(f"  {r['class_id']:14s} {r['method']:9s} box {r['box']} viên {r['tablet_bbox']} lệch {r['cut_px']} px -> {r['new_box']}")
        if args.out:
            with open(args.out, "w") as f:
                json.dump(rep, f, indent=1)
            print(f"-> {args.out}")
    else:
        rep = json.load(open(args.report))
        n = apply(rep, MANIFEST, CORRECTIONS, args.source)
        print(f"[box audit] {n} box đã sửa trong {MANIFEST}; nối vào {CORRECTIONS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
