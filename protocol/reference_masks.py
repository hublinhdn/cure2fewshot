#!/usr/bin/env python3
"""reference_masks.py — mặt nạ foreground ĐỘC LẬP cho 388 reference gốc, để kiểm cổng toàn vẹn.

Cổng solidity của Phase F không bắt được vết cắt thẳng: một viên bị cắt mất một lát vẫn lồi.
Script này tạo mặt nạ bằng một cơ chế khác hẳn luật ngưỡng màu của Phase F, mô hình phân đoạn SAM
(cùng trọng số vit_b đã sinh các bbox đóng băng, SHA256 ghi trong expected/cure_crops_meta.json),
trên ảnh reference GỐC. `measure_mask_damage.py --masks` rồi so vùng Phase F giữ lại với mặt nạ này
và báo tỷ lệ pixel thân viên bị thay (xem KNOWN_ISSUES.md mục 10).

Hai chế độ:
  masks   chạy SAM, ghi <out>/reference_masks/<class_id>.png (0/255) và reference_masks.csv
  sheet   chọn mẫu 40 lớp (5 suspect, 10 solidity thấp nhất, 25 rải đều theo solidity) và in
          contact sheet gốc | mặt nạ SAM | ảnh hardened để người kiểm bằng mắt; ghi mẫu
          reference_mask_review.csv (class_id, solidity, suspect, verdict, note) để điền.

Biến môi trường: CURE_CROPS_ROOT, CURE_MANIFEST (mặc định frozen/), CURE_SAM_CKPT (mặc định
weights/sam/sam_vit_b.pth cạnh repo), PHASE_F_V2_OUT (mặc định outputs_v2).
"""
import argparse
import csv
import hashlib
import json
import os
import sys

import numpy as np
from PIL import Image
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MANIFEST = os.environ.get("CURE_MANIFEST", os.path.join(HERE, "frozen", "cure_crops_manifest.curated.csv"))
CROPS_ROOT = os.environ.get("CURE_CROPS_ROOT")
SAM_CKPT = os.environ.get("CURE_SAM_CKPT", os.path.join(ROOT, "weights", "sam", "sam_vit_b.pth"))
OUT = os.environ.get("PHASE_F_V2_OUT", os.path.join(ROOT, "outputs_v2"))
PAD_GREY, PAD_TOL = 127, 3
AREA_LO, AREA_HI = 0.05, 0.90
REVIEW_N, REVIEW_LOWEST = 40, 10
TILE = 192


BORDER_PX = 2


def touches_border(mask, valid, px=BORDER_PX):
    """True nếu mặt nạ có pixel trong vành `px` pixel sát biên của vùng hợp lệ (ngoài letterbox).
    Nền tổng hợp của reference chạm biên crop; viên thuốc thì không, vì mọi bbox mang lề 15 %."""
    ys, xs = np.where(valid)
    if len(ys) == 0:
        return False
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    ring = np.zeros_like(valid)
    ring[y0:y1 + 1, x0:x1 + 1] = True
    ring[y0 + px:y1 + 1 - px, x0 + px:x1 + 1 - px] = False
    return bool((mask & ring).any())


def pick_mask(masks, scores, lo=AREA_LO, hi=AREA_HI, valid=None):
    """Chỉ số mặt nạ có điểm cao nhất trong số mặt nạ có diện tích trong [lo, hi] của ảnh và,
    khi có `valid`, không chạm biên vùng hợp lệ; None nếu không có. Mặt nạ phủ gần hết ảnh hay
    chạm biên là nền, mặt nạ tí hon là chữ in hoặc bóng."""
    fr = masks.reshape(len(masks), -1).mean(axis=1)
    ok = [i for i in range(len(masks)) if lo <= fr[i] <= hi
          and (valid is None or not touches_border(masks[i].astype(bool), valid))]
    if not ok:
        return None
    return int(max(ok, key=lambda i: scores[i]))


def largest_component(mask):
    """Thành phần liên thông lớn nhất (4 liên thông), lỗ bên trong được lấp."""
    lab, n = ndimage.label(mask)
    if n == 0:
        return mask.astype(bool)
    sizes = ndimage.sum_labels(mask, lab, index=np.arange(1, n + 1))
    big = lab == (int(np.argmax(sizes)) + 1)
    return ndimage.binary_fill_holes(big)


def select_review_sample(solidity_by_cid, suspects, n=REVIEW_N, n_lowest=REVIEW_LOWEST):
    """Mẫu kiểm tay, tất định: mọi suspect; rồi n_lowest lớp solidity thấp nhất; rồi phần còn lại
    rải đều theo thứ hạng solidity của các lớp còn lại."""
    order = sorted(solidity_by_cid, key=lambda c: (solidity_by_cid[c], c))
    picked = [c for c in sorted(suspects) if c in solidity_by_cid]
    for c in order:
        if len(picked) >= min(n, len(order)):
            break
        if c not in picked and len([p for p in picked if p not in suspects]) < n_lowest:
            picked.append(c)
    rest = [c for c in order if c not in picked]
    need = min(n, len(order)) - len(picked)
    if need > 0 and rest:
        idx = np.linspace(0, len(rest) - 1, need).round().astype(int)
        seen = set()
        for i in idx:
            j = int(i)
            while rest[j] in seen or rest[j] in picked:
                j = (j + 1) % len(rest)
            seen.add(rest[j]); picked.append(rest[j])
    return picked[:n]


def classes_by_damage(damage_json, worst=None, above=None):
    """Lớp xếp theo fg_removed_core_frac giảm dần từ một mask_damage JSON; `worst` lấy n đầu,
    `above` chỉ giữ lớp vượt ngưỡng. Lớp không có số đo bị bỏ qua."""
    per = json.load(open(damage_json))["per_class"]
    rows = sorted(((v["fg_removed_core_frac"], c) for c, v in per.items() if "fg_removed_core_frac" in v),
                  key=lambda t: (-t[0], t[1]))
    if above is not None:
        rows = [t for t in rows if t[0] > above]
    if worst is not None:
        rows = rows[:worst]
    return [c for _, c in rows]


def _refs():
    rows = [r for r in csv.DictReader(open(MANIFEST, encoding="utf-8")) if r["kind"] == "ref"]
    rows.sort(key=lambda r: r["crop_rel"])
    return rows


def _orig_path(r):
    c = r["crop_rel"]
    return c if os.path.isabs(c) else os.path.join(CROPS_ROOT, c)


def stage_masks(out_dir):
    import torch
    from segment_anything import sam_model_registry, SamPredictor
    if not CROPS_ROOT:
        raise SystemExit("[ref masks] thieu CURE_CROPS_ROOT")
    if not os.path.exists(SAM_CKPT):
        raise SystemExit(f"[ref masks] khong thay SAM checkpoint: {SAM_CKPT} (CURE_SAM_CKPT)")
    sha = hashlib.sha256(open(SAM_CKPT, "rb").read()).hexdigest()
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    # CURE_SAM_DEVICE=cpu ép chạy CPU (chậm, chừng nửa giờ cho 388 ảnh); mặc định cuda nếu có,
    # và tự rơi về CPU khi GPU hết bộ nhớ vì bị tiến trình khác chiếm.
    dev = os.environ.get("CURE_SAM_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
    sam = sam_model_registry["vit_b"](checkpoint=SAM_CKPT).to(dev).eval()
    pred = SamPredictor(sam)
    mask_dir = os.path.join(out_dir, "reference_masks")
    os.makedirs(mask_dir, exist_ok=True)
    refs = _refs()
    print(f"[ref masks] SAM vit_b sha256 {sha[:12]} trên {dev}; {len(refs)} reference", flush=True)
    rows, n_none = [], 0
    for i, r in enumerate(refs, 1):
        cid = f"{r['pillId']}__{r['side']}"
        arr = np.asarray(Image.open(_orig_path(r)).convert("RGB"))
        H, W = arr.shape[:2]
        pad = (np.abs(arr.astype(np.int16) - PAD_GREY).sum(2) <= PAD_TOL * 3)
        valid = largest_component(~pad) if (~pad).any() else np.ones((H, W), bool)
        ys, xs = np.where(valid)
        y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
        # Prompt: tâm ảnh dương (viên nằm giữa crop); điểm âm là nền tổng hợp: bốn góc của vùng
        # hợp lệ (bộ A), hoặc 12 điểm rải quanh vành trong của vùng hợp lệ (bộ B), bộ C = B kèm
        # box của vùng hợp lệ. Ba bộ x ba mặt nạ = chín ứng viên; loại ứng viên chạm biên hay lệch
        # dải diện tích, rồi lấy điểm cao nhất. Viên nhạt màu trên nền sáng là ca khó nhất.
        inset = 6
        xa, xb, ya, yb = x0 + inset, x1 - inset, y0 + inset, y1 - inset
        xm, ym = (xa + xb) / 2, (ya + yb) / 2
        corners = [[xa, ya], [xb, ya], [xa, yb], [xb, yb]]
        ring = corners + [[xm, ya], [xm, yb], [xa, ym], [xb, ym],
                          [(xa + xm) / 2, ya], [(xm + xb) / 2, yb], [xa, (ya + ym) / 2], [xb, (ym + yb) / 2]]
        centre = [[W / 2, H / 2]]
        try:
            pred.set_image(arr)
        except torch.OutOfMemoryError:
            if dev == "cpu":
                raise
            print("[ref masks] GPU hết bộ nhớ (tiến trình khác đang chiếm): chuyển sang CPU", flush=True)
            dev = "cpu"
            sam = sam.to(dev); pred = SamPredictor(sam)
            torch.cuda.empty_cache()
            pred.set_image(arr)
        cands, cscores = [], []
        for pts, box in ((centre + corners, None), (centre + ring, None),
                         (centre + ring, np.array([x0, y0, x1, y1]))):
            lbl = np.array([1] + [0] * (len(pts) - 1))
            mk, sc, _ = pred.predict(point_coords=np.array(pts, dtype=float), point_labels=lbl,
                                     box=box, multimask_output=True)
            cands.append(mk); cscores.append(sc)
        masks = np.concatenate(cands); scores = np.concatenate(cscores)
        masks = np.stack([largest_component(mk & valid) for mk in masks])
        k = pick_mask(masks, scores, valid=valid)
        if k is None:
            k = int(np.argmax(scores)); n_none += 1
        m = masks[k]
        Image.fromarray((m * 255).astype(np.uint8)).save(os.path.join(mask_dir, f"{cid}.png"))
        rows.append({"class_id": cid, "area_frac": round(float(m.mean()), 6),
                     "sam_score": round(float(scores[k]), 6), "candidate": k,
                     "touches_border": int(touches_border(m, valid)),
                     "in_area_band": int(AREA_LO <= float(m.mean()) <= AREA_HI)})
        if i % 100 == 0:
            print(f"[ref masks] {i}/{len(refs)}", flush=True)
    with open(os.path.join(out_dir, "reference_masks.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    meta = {"sam_type": "vit_b", "sam_ckpt_sha256": sha, "n_masks": len(rows),
            "n_outside_area_band": n_none, "area_band": [AREA_LO, AREA_HI],
            "prompt": ("centre point positive; negatives at the four corners of the non letterbox region "
                       "(set A) or at twelve points around its inner rim (set B), set C = B plus the "
                       "region's box; multimask: nine candidates"),
            "selection": ("largest 4 connected component inside the non letterbox region, holes filled; "
                          "candidates touching the region's border or outside the area band rejected; "
                          "best SAM score among the rest"),
            "border_px": BORDER_PX, "n_border_or_band_fallback": n_none}
    with open(os.path.join(out_dir, "reference_masks_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[ref masks] xong {len(rows)} mặt nạ -> {mask_dir}; không ứng viên hợp lệ (lấy điểm cao nhất): {n_none}", flush=True)


def _overlay(img, mask, colour):
    ov = np.asarray(img).copy()
    ov[mask] = (0.45 * ov[mask] + 0.55 * np.array(colour)).astype(np.uint8)
    return Image.fromarray(ov)


def stage_sheet(out_dir, hardened_dir, log_path, classes=None, damage=None, name="reference_mask_review"):
    """Mỗi hàng một lớp: ảnh gốc | vùng Phase F giữ lại (đỏ) | mặt nạ SAM (xanh) | ảnh hardened.
    Mặc định mẫu 40 lớp theo solidity; `classes` cho danh sách tường minh. `damage` (mask_damage
    JSON) để in số foreground bị mất lên đầu hàng."""
    from PIL import ImageDraw
    log = json.load(open(log_path))["per_class"]
    sol = {c: v["solidity"] for c, v in log.items()}
    sus = [c for c, v in log.items() if v.get("suspect")]
    sample = list(classes) if classes else select_review_sample(sol, sus)
    dmg = json.load(open(damage))["per_class"] if damage else {}
    refs = {f"{r['pillId']}__{r['side']}": r for r in _refs()}
    mask_dir = os.path.join(out_dir, "reference_masks")
    per_sheet = 10
    for s in range(0, len(sample), per_sheet):
        chunk = sample[s:s + per_sheet]
        sheet = Image.new("RGB", (4 * TILE + 12, len(chunk) * (TILE + 18)), "white")
        d = ImageDraw.Draw(sheet)
        for row, cid in enumerate(chunk):
            y = row * (TILE + 18)
            o_full = np.asarray(Image.open(_orig_path(refs[cid])).convert("RGB")).astype(np.int16)
            h_full = np.asarray(Image.open(os.path.join(hardened_dir, f"{cid}.png")).convert("RGB")).astype(np.int16)
            keep = Image.fromarray(((np.abs(o_full - h_full).max(axis=2) <= 2) * 255).astype(np.uint8))
            o = Image.fromarray(o_full.astype(np.uint8)).resize((TILE, TILE))
            keep_m = np.asarray(keep.resize((TILE, TILE))) > 127
            sam_m = np.asarray(Image.open(os.path.join(mask_dir, f"{cid}.png")).resize((TILE, TILE))) > 127
            h = Image.fromarray(h_full.astype(np.uint8)).resize((TILE, TILE))
            sheet.paste(o, (0, y + 18))
            sheet.paste(_overlay(o, keep_m, (220, 0, 0)), (TILE + 4, y + 18))
            sheet.paste(_overlay(o, sam_m, (0, 200, 0)), (2 * TILE + 8, y + 18))
            sheet.paste(h, (3 * TILE + 12, y + 18))
            extra = (f"  core removed {dmg[cid]['fg_removed_core_frac']:.2f}  bg kept {dmg[cid]['bg_kept_frac']:.2f}"
                     if cid in dmg and "fg_removed_core_frac" in dmg[cid] else "")
            d.text((2, y + 3), f"{cid}  solidity {sol[cid]:.3f}{'  SUSPECT' if cid in sus else ''}{extra}"
                   f"   | original | kept by Phase F (red) | SAM mask (green) | hardened", fill="black")
        p = os.path.join(out_dir, f"{name}_sheet_{s // per_sheet + 1}.png")
        sheet.save(p)
        print(f"[ref masks] sheet -> {p}", flush=True)
    tmpl = os.path.join(out_dir, f"{name}.csv")
    with open(tmpl, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class_id", "solidity_v2", "suspect", "fg_removed_core_frac", "bg_kept_frac",
                    "use", "note"])
        for cid in sample:
            dm = dmg.get(cid, {})
            w.writerow([cid, f"{sol[cid]:.4f}", int(cid in sus),
                        f"{dm.get('fg_removed_core_frac', float('nan')):.4f}",
                        f"{dm.get('bg_kept_frac', float('nan')):.4f}", "", ""])
    print(f"[ref masks] mẫu {len(sample)} lớp -> {tmpl}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=["masks", "sheet"])
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--hardened", default=None, help="sheet: thư mục ảnh hardened v2")
    ap.add_argument("--log", default=None, help="sheet: hardened_v2_log.json")
    ap.add_argument("--damage", default=None, help="sheet: mask_damage JSON để ghi số lên hàng")
    ap.add_argument("--worst", type=int, default=None, help="sheet: n lớp mất lõi nhiều nhất theo --damage")
    ap.add_argument("--above", type=float, default=None, help="sheet: lớp có core removed vượt mức này")
    ap.add_argument("--classes", default=None, help="sheet: danh sách lớp, phẩy ngăn cách")
    ap.add_argument("--name", default="reference_mask_review", help="sheet: tên file đầu ra")
    args = ap.parse_args()
    if args.stage == "masks":
        stage_masks(args.out)
    else:
        classes = args.classes.split(",") if args.classes else None
        if classes is None and args.damage and (args.worst is not None or args.above is not None):
            classes = classes_by_damage(args.damage, args.worst, args.above)
        stage_sheet(args.out, args.hardened or os.path.join(args.out, "hardened_refs_v2"),
                    args.log or os.path.join(args.out, "hardened_v2_log.json"), classes, args.damage, args.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
