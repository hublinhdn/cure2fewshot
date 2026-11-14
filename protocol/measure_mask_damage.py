#!/usr/bin/env python3
"""measure_mask_damage.py — do mức mặt nạ của Phase F cắt vào thân viên thuốc.

Phase F thay nền của ảnh reference và tuyên bố KHÔNG đụng tới viên thuốc. Script này kiểm
tuyên bố đó bằng số, cho bất kỳ thư mục ảnh hardened nào (bản v1 chưa có cổng, bản v2 có cổng).

Cách đo, không cần biết mặt nạ mà script hardening đã dùng:

  keep      = pixel giống nhau giữa ảnh gốc và ảnh hardened (lệch <= 2 mức ở cả ba kênh).
              Đó chính là vùng mà bước hardening giữ lại, tức phần nó coi là viên thuốc.
  solidity  = diện tích thành phần liên thông lớn nhất của keep, chia diện tích bao lồi của
              nó. Viên thuốc là hình lồi, nên mặt nạ cắt vào thân viên làm solidity tụt.
              Phải lấy thành phần lớn nhất trước: pixel nền lạc trùng màu xám letterbox làm
              bao lồi phình ra và cho số sai rất nặng.
  exposure  = tỷ lệ pixel thân viên (vùng keep) có độ xám nằm trong dải mà luật ngưỡng theo
              độ sáng coi là nền, tức |gray - gray_nen| <= 22, với gray_nen là trung vị vành
              biên 6 px của ảnh gốc sau khi loại vành letterbox xám 127. Đây là phơi nhiễm
              của cơ chế, đo trên ảnh gốc: viên càng nhạt màu thì càng dễ bị coi là nền.

  Với --masks DIR (mặt nạ độc lập từ reference_masks.py, một PNG 0/255 mỗi lớp) còn đo thêm,
  mỗi lớp, so với mặt nạ đó: fg_removed_frac = phần thân viên (theo mặt nạ) mà hardening đã thay;
  fg_removed_core_frac = cùng số nhưng sau khi gọt mặt nạ vào BAND_PX pixel, để tách bất đồng ở
  biên giữa hai cơ chế khỏi vết cắt thật; bg_kept_frac = phần vùng giữ lại nằm ngoài mặt nạ.
  Solidity không nhìn thấy một vết cắt thẳng (viên bị cắt vẫn lồi); số này thì thấy.

Dùng:
  CURE_CROPS_ROOT=/path CURE_MANIFEST=protocol/frozen/cure_crops_manifest.curated.csv \
    python protocol/measure_mask_damage.py --hardened outputs/hardened_refs --tag v1 \
      --out outputs/mask_damage_v1.json [--masks outputs_v2/reference_masks]

Mã thoát: 0 nếu đo xong, 2 nếu thiếu đầu vào.
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
MANIFEST = os.environ.get(
    "CURE_MANIFEST", os.path.join(HERE, "frozen", "cure_crops_manifest.curated.csv"))
CROPS_ROOT = os.environ.get("CURE_CROPS_ROOT")

KEEP_TOL = 2        # lệch tối đa mỗi kênh để coi là "pixel được giữ nguyên"
BAND = 22           # dải độ xám mà luật ngưỡng theo độ sáng coi là nền
BORDER = 6          # bề rộng vành biên dùng để ước lượng độ xám nền
PAD_GREY = 127      # xám letterbox, loại khỏi ước lượng nền
PAD_TOL = 3
MIN_KEEP = 200      # dưới mức này coi như không đo được
BAND_PX = 3         # gọt mặt nạ độc lập vào chừng này pixel trước khi đo "lõi"


def _disk(r):
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (y * y + x * x) <= r * r


def preservation_stats(keep, mask, band=BAND_PX):
    """keep: vùng hardening giữ nguyên (bool); mask: mặt nạ foreground độc lập (bool).
    Trả tỷ lệ thân viên bị thay (toàn bộ và sau khi gọt biên band px) và tỷ lệ vùng giữ nằm ngoài
    mặt nạ. Mọi tỷ lệ tính trên số pixel, chia cho mẫu số của chính nó."""
    keep = np.asarray(keep, bool)
    fg = np.asarray(mask, bool)
    core = ndimage.binary_erosion(fg, structure=_disk(band)) if band > 0 else fg
    fg_px, core_px, keep_px = int(fg.sum()), int(core.sum()), int(keep.sum())
    removed = int((fg & ~keep).sum())
    removed_core = int((core & ~keep).sum())
    bg_kept = int((keep & ~fg).sum())
    return {"fg_px": fg_px, "keep_px": keep_px, "fg_removed_px": removed,
            "fg_removed_frac": removed / fg_px if fg_px else 0.0,
            "fg_core_px": core_px, "fg_removed_core_frac": removed_core / core_px if core_px else 0.0,
            "bg_kept_px": bg_kept, "bg_kept_frac": bg_kept / keep_px if keep_px else 0.0,
            "boundary_band_px": int(band)}


def _hull_area(ys, xs):
    """Diện tích bao lồi của một tập điểm, qua các điểm cực trị theo hàng và theo cột."""
    cand = []
    for v in np.unique(ys):
        sel = xs[ys == v]
        cand += [(int(sel.min()), int(v)), (int(sel.max()), int(v))]
    for v in np.unique(xs):
        sel = ys[xs == v]
        cand += [(int(v), int(sel.min())), (int(v), int(sel.max()))]
    pts = sorted(set(cand))
    if len(pts) < 3:
        return 0.0

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    poly = lower[:-1] + upper[:-1]
    a = 0.0
    for i in range(len(poly)):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % len(poly)]
        a += x1 * y2 - x2 * y1
    return abs(a) / 2.0


def measure_one(orig_path, hard_path, mask_path=None):
    o = np.asarray(Image.open(orig_path).convert("RGB"), dtype=np.int16)
    h = np.asarray(Image.open(hard_path).convert("RGB"), dtype=np.int16)
    if o.shape != h.shape:
        return None
    keep = np.abs(o - h).max(axis=2) <= KEEP_TOL
    n_keep = int(keep.sum())
    if n_keep < MIN_KEEP:
        return None
    lab, n = ndimage.label(keep, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]))
    sizes = ndimage.sum_labels(keep, lab, index=np.arange(1, n + 1))
    big = int(np.argmax(sizes)) + 1
    cc = lab == big
    n_cc = int(cc.sum())
    ys, xs = np.nonzero(cc)
    ha = _hull_area(ys, xs)
    solidity = float(n_cc / ha) if ha > 0 else 0.0

    g = o.mean(axis=2)
    ring = np.concatenate([g[:BORDER, :].ravel(), g[-BORDER:, :].ravel(),
                           g[:, :BORDER].ravel(), g[:, -BORDER:].ravel()])
    ring = ring[np.abs(ring - PAD_GREY) > PAD_TOL]
    bg = float(np.median(ring)) if ring.size else float(PAD_GREY)
    pill = float(np.median(g[keep]))
    exposure = float((np.abs(g[keep] - bg) <= BAND).mean())
    out = {"solidity": solidity, "exposure": exposure,
           "grey_gap": abs(pill - bg), "keep_px": n_keep,
           "keep_px_largest_component": n_cc,
           "stray_px": n_keep - n_cc}
    if mask_path and os.path.exists(mask_path):
        m = np.asarray(Image.open(mask_path).convert("L")) > 127
        if m.shape == keep.shape:
            out.update(preservation_stats(keep, m))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--hardened", required=True, help="thu muc anh hardened, ten file <class_id>.png")
    ap.add_argument("--out", required=True, help="file JSON ket qua")
    ap.add_argument("--tag", default="", help="nhan ghi vao JSON, vi du v1 hoac v2")
    ap.add_argument("--masks", default=None,
                    help="thu muc mat na doc lap <class_id>.png (reference_masks.py), neu co")
    args = ap.parse_args()

    if not CROPS_ROOT:
        raise SystemExit("[mask damage] thieu CURE_CROPS_ROOT (goc cua crop_rel)")
    if not os.path.isdir(args.hardened):
        raise SystemExit(f"[mask damage] khong thay thu muc hardened: {args.hardened}")

    rows = [r for r in csv.DictReader(open(MANIFEST, encoding="utf-8")) if r["kind"] == "ref"]
    print(f"[mask damage] manifest = {MANIFEST} ({len(rows)} reference)", flush=True)
    print(f"[mask damage] hardened = {args.hardened}", flush=True)

    per = {}
    skipped = []
    for i, r in enumerate(rows, 1):
        cid = f"{r['pillId']}__{r['side']}"
        hp = os.path.join(args.hardened, cid + ".png")
        op = r["crop_rel"] if os.path.isabs(r["crop_rel"]) else os.path.join(CROPS_ROOT, r["crop_rel"])
        if not (os.path.exists(hp) and os.path.exists(op)):
            skipped.append(cid)
            continue
        mp = os.path.join(args.masks, cid + ".png") if args.masks else None
        m = measure_one(op, hp, mp)
        if m is None:
            skipped.append(cid)
            continue
        per[cid] = m
        if i % 100 == 0:
            print(f"[mask damage] {i}/{len(rows)}", flush=True)

    if not per:
        raise SystemExit("[mask damage] khong do duoc anh nao")
    sol = np.array([v["solidity"] for v in per.values()])
    exp = np.array([v["exposure"] for v in per.values()])
    stray = np.array([v["stray_px"] for v in per.values()])
    worst_sol = min(per.items(), key=lambda kv: kv[1]["solidity"])
    worst_exp = max(per.items(), key=lambda kv: kv[1]["exposure"])
    stats = {
        "tag": args.tag,
        "n_measured": len(per),
        "n_skipped": len(skipped),
        "solidity_min": float(sol.min()),
        "solidity_p05": float(np.percentile(sol, 5)),
        "solidity_median": float(np.median(sol)),
        "n_solidity_below_098": int((sol < 0.98).sum()),
        "n_solidity_below_095": int((sol < 0.95).sum()),
        "n_solidity_below_090": int((sol < 0.90).sum()),
        "exposure_median": float(np.median(exp)),
        "exposure_p95": float(np.percentile(exp, 95)),
        "exposure_max": float(exp.max()),
        "n_exposure_above_005": int((exp > 0.05).sum()),
        "n_exposure_above_020": int((exp > 0.20).sum()),
        "n_exposure_above_040": int((exp > 0.40).sum()),
        "worst_solidity_class": worst_sol[0],
        "worst_exposure_class": worst_exp[0],
        "stray_px_median": float(np.median(stray)),
        "keep_tolerance": KEEP_TOL,
        "background_band": BAND,
    }
    with_mask = {c: v for c, v in per.items() if "fg_removed_frac" in v}
    if with_mask:
        fr = np.array([v["fg_removed_frac"] for v in with_mask.values()])
        fc = np.array([v["fg_removed_core_frac"] for v in with_mask.values()])
        bk = np.array([v["bg_kept_frac"] for v in with_mask.values()])
        worst = max(with_mask.items(), key=lambda kv: kv[1]["fg_removed_core_frac"])
        stats.update({
            "n_with_independent_mask": len(with_mask),
            "boundary_band_px": BAND_PX,
            "fg_removed_median": float(np.median(fr)), "fg_removed_p95": float(np.percentile(fr, 95)),
            "fg_removed_max": float(fr.max()),
            "fg_removed_core_median": float(np.median(fc)),
            "fg_removed_core_p95": float(np.percentile(fc, 95)),
            "fg_removed_core_max": float(fc.max()),
            "n_fg_removed_core_above_001": int((fc > 0.01).sum()),
            "n_fg_removed_core_above_005": int((fc > 0.05).sum()),
            "n_fg_removed_core_above_020": int((fc > 0.20).sum()),
            "worst_fg_removed_core_class": worst[0],
            "bg_kept_median": float(np.median(bk)), "bg_kept_p95": float(np.percentile(bk, 95)),
            "bg_kept_max": float(bk.max()),
        })
    with open(args.out, "w") as f:
        json.dump({"stats": stats, "per_class": per}, f, indent=2, sort_keys=True)
    print(json.dumps(stats, indent=2))
    print(f"[mask damage] xong -> {args.out}", flush=True)
    if skipped:
        print(f"[mask damage] bo qua {len(skipped)} anh: {skipped[:5]}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
