"""
harden_refs.py — Phase F: làm cứng domain gap bằng cách thay nền đồng nhất của reference
bằng nền THẬT crop từ ảnh raw consumer (vùng ngoài bbox viên thuốc), triệt shortcut
"nền đồng nhất = reference" (domain AUC frozen đo được ~1.0).

Cách làm, tất định seed 42:
  1. Với mỗi crop reference 384: ước lượng màu nền từ vành biên (loại vành letterbox xám 127),
     mask nền = pixel gần màu nền; foreground = thành phần liên thông lớn nhất còn lại,
     lấp lỗ + nở 2px giữ mép viên.
  2. Donor: chọn 1 ảnh raw consumer của PILL KHÁC, cắt cửa sổ vuông ở góc xa bbox viên nhất
     (nền bàn/tay thật), resize 384. Nếu không có góc sạch -> nhiễu Gauss (fallback, có đếm).
  3. Ghép: nền donor + pixel viên giữ nguyên -> outputs/hardened_refs/<class_id>.png
  4. Xuất outputs/manifest_hardened.csv: bản sao manifest curated, crop_rel của reference
     trỏ (đường dẫn tuyệt đối) sang ảnh hardened. Chạy lại fitness trên manifest này:
       CURE_MANIFEST=outputs/manifest_hardened.csv CURE_FITNESS_OUT=outputs/hardened \
       CURE_EMB_REUSE=outputs/embeddings python cure_fitness_check.py all
  5. Xuất gallery_fewshot_K1_hardened.csv + lưới audit hardened_audit_grid.png để kiểm mắt.
"""
import os
import json

import numpy as np
import pandas as pd
from PIL import Image

THIS = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("CURE_FITNESS_OUT", os.path.join(THIS, "outputs"))
HARD_DIR = os.path.join(OUT, "hardened_refs")
PILL_PROJ = os.environ.get(
    "PILL_PROJ", os.path.join(os.path.dirname(THIS), "data"))
# PILL_PROJ = goc du lieu: thu muc chua data/raw/CURE_dataset va data/processed/cure_crops.
# Dat bang bien moi truong PILL_PROJ, xem REPRODUCE.md.
# crops có thể nằm ngoài project (workspace server độc lập); raw/donor luôn theo PILL_PROJ
CROPS_ROOT = os.environ.get("CURE_CROPS_ROOT", PILL_PROJ)
MANIFEST = os.environ.get(
    "CURE_MANIFEST", os.path.join(PILL_PROJ, "data", "processed", "cure_crops_manifest.curated.csv"))
SEED = 42
BG_TOL = 42          # khoảng cách màu coi là nền đồng nhất
PAD_TOL = 12         # khoảng cách tới xám letterbox (127,127,127)
SIZE = 384


def _otsu(vals, lo=12, hi=400):
    """Ngưỡng Otsu trên phân bố khoảng cách màu (sum abs RGB, 0..765), kẹp [lo, hi]."""
    hist, edges = np.histogram(vals, bins=256, range=(0, 765))
    total = hist.sum()
    if total == 0:
        return lo
    w0 = np.cumsum(hist)
    w1 = total - w0
    mids = (edges[:-1] + edges[1:]) / 2
    s0 = np.cumsum(hist * mids)
    mu0 = np.divide(s0, w0, out=np.zeros_like(s0), where=w0 > 0)
    mu1 = np.divide(s0[-1] - s0, w1, out=np.zeros_like(s0), where=w1 > 0)
    var = w0 * w1 * (mu0 - mu1) ** 2
    return float(np.clip(mids[int(np.argmax(var))], lo, hi))


def build_fg_mask(arr):
    """arr HxWx3 uint8 -> mask foreground (viên thuốc). Trả (mask, fg_frac, thr).
    Nền reference đồng nhất theo cấu trúc -> mask theo khoảng cách tới màu nền, ngưỡng Otsu
    thích nghi; nếu tỉ lệ foreground vô lý (viên bị ăn hoặc dính cả nền) thì retry theo
    thang ngưỡng. Viên chiếm ~1/1.3^2 ~ 0.6 khung (crop margin 0.15) nên fg hợp lý ~[0.2, 0.85]."""
    from scipy import ndimage
    a = arr.astype(np.int16)
    pad = (np.abs(a - 127).sum(2) <= PAD_TOL * 3)
    w = 8
    ring = np.zeros(arr.shape[:2], bool)
    ring[:w], ring[-w:], ring[:, :w], ring[:, -w:] = True, True, True, True
    ring_px = a[ring & ~pad]
    if len(ring_px) < 50:                       # biên toàn letterbox -> lấy vành trong hơn
        ring2 = np.zeros(arr.shape[:2], bool)
        ring2[w:3 * w], ring2[-3 * w:-w], ring2[:, w:3 * w], ring2[:, -3 * w:-w] = (True,) * 4
        ring_px = a[ring2 & ~pad]
    if len(ring_px) < 50:                       # cùng lắm: toàn bộ vành 24px trừ letterbox
        border = np.zeros(arr.shape[:2], bool)
        border[:24], border[-24:], border[:, :24], border[:, -24:] = (True,) * 4
        ring_px = a[border & ~pad]
    bg_color = np.median(ring_px, axis=0) if len(ring_px) else np.array([255, 255, 255])
    dist = np.abs(a - bg_color).sum(2)

    def try_thr(thr):
        """Trả (mask, frac, fill). fill = độ đặc mask trong bbox của nó — viên thuốc là hình
        lồi nên mask đúng có fill cao (~0.78 ellipse); mask 'ăn viên' lỗ chỗ có fill thấp."""
        fg0 = (dist > thr) & ~pad
        lab, n = ndimage.label(fg0)
        if n == 0:
            return None, 0.0, 0.0
        largest = np.argmax(np.bincount(lab.ravel())[1:]) + 1
        m = ndimage.binary_fill_holes(lab == largest)
        m = ndimage.binary_dilation(m, iterations=2)
        ys, xs = np.where(m)
        fill = float(m.sum() / max((ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1), 1))
        return m, float(m.mean()), fill

    ring_dist = dist[ring & ~pad]
    candidates = [_otsu(dist[~pad].ravel()),
                  (float(np.quantile(ring_dist, 0.999)) * 1.5 if len(ring_dist) else 90.0),
                  90.0, 60.0, 36.0, 24.0, 16.0]
    for thr in candidates:
        m, frac, fill = try_thr(thr)
        if m is not None and 0.20 <= frac <= 0.85 and fill >= 0.55:
            return m, frac, thr
    # fallback an toàn: ellipse nội tiếp phủ vùng viên (crop = bbox + margin 0.15 -> viên
    # chiếm ~77% mỗi chiều). Giữ nguyên viên + quầng nền gốc mỏng, không bao giờ phá viên.
    H, W = arr.shape[:2]
    inner = ~pad
    ys, xs = np.where(inner)
    cy, cx = (ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2
    ry, rx = (ys.max() - ys.min()) * 0.42, (xs.max() - xs.min()) * 0.42
    Y, X = np.ogrid[:H, :W]
    m = (((Y - cy) / max(ry, 1)) ** 2 + ((X - cx) / max(rx, 1)) ** 2) <= 1.0
    return m, float(m.mean()), -1.0   # thr = -1 đánh dấu ellipse fallback


def donor_patch(row, rng):
    """Cắt cửa sổ nền vuông từ ảnh raw consumer donor, tránh bbox viên. None nếu không có góc sạch."""
    src = os.path.join(PILL_PROJ, row["src_rel"])
    if not os.path.exists(src):
        return None
    W, H = int(row["orig_w"]), int(row["orig_h"])
    x0, y0, x1, y1 = int(row["x0"]), int(row["y0"]), int(row["x1"]), int(row["y1"])
    s = min(W, H) // 3
    corners = [(0, 0), (W - s, 0), (0, H - s), (W - s, H - s)]
    ok = [(cx, cy) for cx, cy in corners
          if cx + s <= x0 or cx >= x1 or cy + s <= y0 or cy >= y1]
    if not ok:
        return None
    cx, cy = ok[int(rng.integers(len(ok)))]
    img = Image.open(src).convert("RGB").crop((cx, cy, cx + s, cy + s)).resize(
        (SIZE, SIZE), Image.LANCZOS)
    return np.asarray(img)


def main():
    os.makedirs(HARD_DIR, exist_ok=True)
    m = pd.read_csv(MANIFEST)
    refs = m[m["kind"] == "ref"].sort_values("crop_rel").reset_index(drop=True)
    cons = m[m["kind"] == "customer"].sort_values("crop_rel").reset_index(drop=True)
    rng = np.random.default_rng(SEED)
    stats = {"n_refs": len(refs), "noise_fallback": 0, "ellipse_fallback": 0, "fg_frac": []}
    hard_path = {}
    for i, r in refs.iterrows():
        cid = f"{r['pillId']}__{r['side']}"
        arr = np.asarray(Image.open(os.path.join(CROPS_ROOT, r["crop_rel"])).convert("RGB"))
        fg, frac, thr = build_fg_mask(arr)
        if thr < 0:
            stats["ellipse_fallback"] += 1
        stats["fg_frac"].append(frac)
        # donor của pill khác, chọn tất định
        patch = None
        for _ in range(6):
            d = cons.iloc[int(rng.integers(len(cons)))]
            if str(d["pillId"]) != str(r["pillId"]):
                patch = donor_patch(d, rng)
                if patch is not None:
                    break
        if patch is None:
            patch = rng.normal(127, 40, arr.shape).clip(0, 255).astype(np.uint8)
            stats["noise_fallback"] += 1
        out_arr = patch.copy()
        out_arr[fg] = arr[fg]
        p = os.path.join(HARD_DIR, f"{cid}.png")
        Image.fromarray(out_arr).save(p)
        hard_path[r["crop_rel"]] = p
        if (i + 1) % 100 == 0:
            print(f"[harden] {i+1}/{len(refs)}", flush=True)

    # manifest hardened: crop_rel của ref -> đường dẫn TUYỆT ĐỐI ảnh hardened
    m2 = m.copy()
    m2["crop_rel"] = m2["crop_rel"].map(lambda c: hard_path.get(c, c))
    m2.to_csv(os.path.join(OUT, "manifest_hardened.csv"), index=False)

    # gallery K=1 hardened (khớp lựa chọn của gallery_fewshot_K1_clean nếu có, else K1 gốc)
    base_p = os.path.join(OUT, "gallery_fewshot_K1_clean.csv")
    if not os.path.exists(base_p):
        base_p = os.path.join(OUT, "gallery_fewshot_K1.csv")
    g = pd.read_csv(base_p)
    g["crop_rel"] = g["crop_rel"].map(lambda c: hard_path.get(c, c))
    g.to_csv(os.path.join(OUT, "gallery_fewshot_K1_hardened.csv"), index=False)

    # lưới audit 4x6: hàng gốc / hàng hardened xen kẽ
    pick = refs.iloc[np.linspace(0, len(refs) - 1, 12, dtype=int)]
    grid = Image.new("RGB", (6 * 192, 4 * 192), (32, 32, 32))
    for k, (_, r) in enumerate(pick.iterrows()):
        cid = f"{r['pillId']}__{r['side']}"
        o = Image.open(os.path.join(CROPS_ROOT, r["crop_rel"])).resize((192, 192))
        h = Image.open(os.path.join(HARD_DIR, f"{cid}.png")).resize((192, 192))
        col, rowk = k % 6, (k // 6) * 2
        grid.paste(o, (col * 192, rowk * 192))
        grid.paste(h, (col * 192, (rowk + 1) * 192))
    grid.save(os.path.join(OUT, "hardened_audit_grid.png"))

    fg = np.array(stats.pop("fg_frac"))
    stats.update({"fg_frac_p05": float(np.quantile(fg, .05)), "fg_frac_median": float(np.median(fg)),
                  "fg_frac_p95": float(np.quantile(fg, .95)),
                  "n_suspect_masks": int(((fg < 0.15) | (fg > 0.95)).sum())})
    with open(os.path.join(OUT, "hardened_log.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[harden] xong {stats['n_refs']} ref, fallback noise {stats['noise_fallback']}, "
          f"mask nghi vấn {stats['n_suspect_masks']} -> kiểm outputs/hardened_audit_grid.png", flush=True)


if __name__ == "__main__":
    main()
