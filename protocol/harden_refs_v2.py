"""Phase F v2: làm cứng nền reference, mask KHÔNG cắt vào thân viên thuốc.

Vì sao có v2. Bản v1 (`cure-to-fewshot/harden_refs.py`) cắt vào thân viên ở một phần đáng kể
bộ dữ liệu, xem `docs/phase_f_mask_defect.md`. Hai nguyên nhân trong code v1:

  1. Thang ngưỡng thử **Otsu trước**, rồi 90, 60, 36, 24, 16, và **nhận ngưỡng đầu tiên** thoả
     điều kiện. Nhưng nền reference của CURE là nền tổng hợp **đồng nhất tuyệt đối**: đo trên
     385 ảnh, trung vị của q99 khoảng cách màu trong vành nền là 0.0, và 352/385 ảnh có q999
     dưới 12 (dưới 4 mức mỗi kênh). Với nền như vậy thì ngưỡng ĐÚNG là ngưỡng NHỎ. Otsu chia
     phân bố làm hai cụm và với viên màu nhạt nó cắt vào giữa thân viên.
  2. Cổng chất lượng là `fill >= 0.55`, tức độ đặc của mask trong bbox của chính nó. Viên bị
     cắt mất nửa vẫn còn fill khoảng 0.6 nên vẫn qua cổng. Và `n_suspect_masks` chỉ kiểm
     `fg_frac` ngoài dải [0.15, 0.95], hoàn toàn không bắt được vết khoét.

v2 sửa đúng ba chỗ đó:

  1. Thang ngưỡng **tăng dần** từ 9, và Otsu chỉ là phương án cuối.
  2. Cổng chất lượng đổi sang **solidity** (diện tích chia diện tích bao lồi). Viên thuốc là
     hình lồi nên mask đúng có solidity gần 1; một vết khoét làm nó tụt ngay. Ngưỡng 0.97.
  3. Có bước **sửa mask**: đóng hình thái để hàn vết khoét nhỏ, và nếu vẫn không đạt thì lấy
     luôn **bao lồi** của mask, hợp lý vì viên thuốc lồi. Ảnh nào vẫn không đạt thì bị đánh dấu
     `suspect` và ghi vào log, chứ không im lặng cho qua như v1.

GIỮ NGUYÊN so với v1: hàm chọn donor, seed 42, và **thứ tự rút số ngẫu nhiên**. Nhờ vậy mỗi
reference nhận đúng miếng nền donor như v1, và khác biệt duy nhất giữa v1 và v2 là cái mask.
Đây là thay đổi có kiểm soát, so sánh được.

KHÔNG ghi đè gì của v1: v2 ghi vào `outputs_v2/`, bản v1 ghi vào thư mục riêng của nó, nên
chạy được cả hai biến thể cạnh nhau để so sánh mà không cái nào đụng cái nào.

Chạy:  python protocol/harden_refs_v2.py
"""
import json
import os

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def _need(var, what):
    v = os.environ.get(var)
    if not v:
        raise SystemExit(f"[phase F v2] thieu bien moi truong {var} ({what})")
    return v


# Duong dan la THAM SO, khong hard code layout may nao ca.
#   CURE_CROPS_ROOT : goc chua cac crop 384, tuc thu muc ma `crop_rel` tinh tuong doi tu do
#   CURE_RAW_ROOT   : goc chua anh raw dung lam donor, tuc thu muc cua `src_rel`
#   CURE_GALLERY    : file gallery K=1 lam co so de ghi lai duong dan reference
#   CURE_MANIFEST   : manifest bbox dong bang (mac dinh: frozen/ canh script)
#   PHASE_F_V2_OUT  : thu muc ghi ket qua (mac dinh: <repo>/outputs_v2)
CROPS_ROOT = _need("CURE_CROPS_ROOT", "goc cua crop_rel")
RAW_ROOT = _need("CURE_RAW_ROOT", "goc cua src_rel, anh raw lam donor")
GALLERY = _need("CURE_GALLERY", "gallery K=1 lam co so")
MANIFEST = os.environ.get("CURE_MANIFEST",
                          os.path.join(HERE, "frozen", "cure_crops_manifest.curated.csv"))

OUT = os.environ.get("PHASE_F_V2_OUT", os.path.join(ROOT, "outputs_v2"))
HARD_DIR = os.path.join(OUT, "hardened_refs_v2")

SEED = 42
PAD_TOL = 12          # khoảng cách tới xám letterbox (127,127,127)
SIZE = 384
SOLIDITY_OK = 0.97    # cổng chất lượng mới
FRAC_LO, FRAC_HI = 0.20, 0.85
CLOSE_R = 5           # bán kính đóng hình thái để hàn vết khoét nhỏ


def _otsu(vals, lo=12, hi=400):
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


def _disk(r):
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (y * y + x * x) <= r * r


def _hull_mask(mask):
    """Bao lồi của mask, rasterise. Viên thuốc lồi nên đây là mask 'không thể khoét'."""
    from scipy.spatial import ConvexHull, QhullError
    ys, xs = np.nonzero(mask)
    if len(ys) < 3:
        return mask.copy()
    pts = np.stack([xs, ys], axis=1).astype(float)
    try:
        hull = ConvexHull(pts)
    except (QhullError, ValueError):
        return mask.copy()
    v = pts[hull.vertices]                       # thứ tự ngược chiều kim đồng hồ
    H, W = mask.shape
    Y, X = np.mgrid[:H, :W]
    inside = np.ones((H, W), bool)
    for i in range(len(v)):
        x1, y1 = v[i]
        x2, y2 = v[(i + 1) % len(v)]
        # dấu của tích có hướng: điểm trong đa giác lồi luôn cùng một phía mọi cạnh
        inside &= ((x2 - x1) * (Y - y1) - (y2 - y1) * (X - x1)) >= -1e-9
    return inside


def _solidity(mask):
    a = float(mask.sum())
    if a < 1:
        return 0.0
    h = float(_hull_mask(mask).sum())
    return a / h if h > 0 else 0.0


def _bg_color(a, pad):
    """Màu nền, lấy từ vành biên, loại vành letterbox. Giữ y logic v1."""
    w = 8
    ring = np.zeros(a.shape[:2], bool)
    ring[:w], ring[-w:], ring[:, :w], ring[:, -w:] = True, True, True, True
    px = a[ring & ~pad]
    if len(px) < 50:
        ring2 = np.zeros(a.shape[:2], bool)
        ring2[w:3 * w], ring2[-3 * w:-w], ring2[:, w:3 * w], ring2[:, -3 * w:-w] = (True,) * 4
        px = a[ring2 & ~pad]
    if len(px) < 50:
        b = np.zeros(a.shape[:2], bool)
        b[:24], b[-24:], b[:, :24], b[:, -24:] = (True,) * 4
        px = a[b & ~pad]
    return np.median(px, axis=0) if len(px) else np.array([255.0, 255.0, 255.0])


def build_fg_mask_v2(arr):
    """arr HxWx3 -> (mask, info). info ghi lại mọi quyết định để log kiểm được."""
    a = arr.astype(np.int16)
    pad = (np.abs(a - 127).sum(2) <= PAD_TOL * 3)
    dist = np.abs(a - _bg_color(a, pad)).sum(2)

    def make(thr):
        fg0 = (dist > thr) & ~pad
        lab, n = ndimage.label(fg0)
        if n == 0:
            return None
        largest = np.argmax(np.bincount(lab.ravel())[1:]) + 1
        m = ndimage.binary_fill_holes(lab == largest)
        m = ndimage.binary_closing(m, structure=_disk(CLOSE_R))   # hàn vết khoét nhỏ
        m = ndimage.binary_fill_holes(m)
        return ndimage.binary_dilation(m, iterations=2)

    # ngưỡng TANG DAN: nền tổng hợp đồng nhất nên ngưỡng nhỏ mới đúng. Otsu chi la cuoi cung.
    cands = [9, 12, 18, 24, 36, 60, 90, _otsu(dist[~pad].ravel())]
    tried = []
    for thr in cands:
        m = make(thr)
        if m is None:
            continue
        frac = float(m.mean())
        if not (FRAC_LO <= frac <= FRAC_HI):
            tried.append((thr, frac, None))
            continue
        sol = _solidity(m)
        tried.append((thr, frac, sol))
        if sol >= SOLIDITY_OK:
            return m, {"thr": float(thr), "frac": frac, "solidity": sol,
                       "repair": "none", "suspect": False}

    # không ngưỡng nào sạch: lấy cái solidity cao nhất trong dải frac hợp lý, rồi lấy bao lồi
    ok = [(thr, frac, sol) for thr, frac, sol in tried if sol is not None]
    if ok:
        thr, frac, sol = max(ok, key=lambda t: t[2])
        m = make(thr)
        hm = _hull_mask(m)
        hfrac = float(hm.mean())
        if hfrac <= 0.90:
            return hm, {"thr": float(thr), "frac": hfrac, "solidity": _solidity(hm),
                        "repair": "convex_hull", "suspect": False,
                        "solidity_before_repair": sol}
        return m, {"thr": float(thr), "frac": frac, "solidity": sol,
                   "repair": "none", "suspect": True,
                   "note": "bao loi phu qua rong, giu mask goc"}

    # cuối cùng: ellipse nội tiếp, không bao giờ phá viên
    H, W = arr.shape[:2]
    ys, xs = np.where(~pad)
    cy, cx = (ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2
    ry, rx = (ys.max() - ys.min()) * 0.42, (xs.max() - xs.min()) * 0.42
    Y, X = np.ogrid[:H, :W]
    m = (((Y - cy) / max(ry, 1)) ** 2 + ((X - cx) / max(rx, 1)) ** 2) <= 1.0
    return m, {"thr": -1.0, "frac": float(m.mean()), "solidity": _solidity(m),
               "repair": "ellipse", "suspect": True}


def donor_patch(row, rng):
    """Y NGUYEN v1: cắt cửa sổ nền vuông từ ảnh raw consumer donor, tránh bbox viên."""
    src = os.path.join(RAW_ROOT, row["src_rel"])
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
    rng = np.random.default_rng(SEED)      # thứ tự rút số y v1 -> donor trùng v1

    per = {}
    stats = {"n_refs": len(refs), "noise_fallback": 0, "ellipse_fallback": 0,
             "hull_repair": 0, "suspect": 0}
    hard_path = {}
    for i, r in refs.iterrows():
        cid = f"{r['pillId']}__{r['side']}"
        arr = np.asarray(Image.open(os.path.join(CROPS_ROOT, r["crop_rel"])).convert("RGB"))
        fg, info = build_fg_mask_v2(arr)
        if info["repair"] == "ellipse":
            stats["ellipse_fallback"] += 1
        if info["repair"] == "convex_hull":
            stats["hull_repair"] += 1
        if info["suspect"]:
            stats["suspect"] += 1
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
        per[cid] = info
        if (i + 1) % 100 == 0:
            print(f"[harden v2] {i+1}/{len(refs)}", flush=True)

    m2 = m.copy()
    m2["crop_rel"] = m2["crop_rel"].map(lambda c: hard_path.get(c, c))
    m2.to_csv(os.path.join(OUT, "manifest_hardened_v2.csv"), index=False)

    g = pd.read_csv(GALLERY)
    g["crop_rel"] = g["crop_rel"].map(lambda c: hard_path.get(c, c))
    g.to_csv(os.path.join(OUT, "gallery_fewshot_K1_hardened_v2.csv"), index=False)

    pick = refs.iloc[np.linspace(0, len(refs) - 1, 12, dtype=int)]
    grid = Image.new("RGB", (6 * 192, 4 * 192), (32, 32, 32))
    for k, (_, r) in enumerate(pick.iterrows()):
        cid = f"{r['pillId']}__{r['side']}"
        o = Image.open(os.path.join(CROPS_ROOT, r["crop_rel"])).resize((192, 192))
        h = Image.open(os.path.join(HARD_DIR, f"{cid}.png")).resize((192, 192))
        col, rowk = k % 6, (k // 6) * 2
        grid.paste(o, (col * 192, rowk * 192))
        grid.paste(h, (col * 192, (rowk + 1) * 192))
    grid.save(os.path.join(OUT, "hardened_audit_grid_v2.png"))

    sol = np.array([v["solidity"] for v in per.values()])
    frac = np.array([v["frac"] for v in per.values()])
    stats.update({
        "solidity_min": float(sol.min()), "solidity_p05": float(np.quantile(sol, .05)),
        "solidity_median": float(np.median(sol)),
        "n_solidity_below_097": int((sol < 0.97).sum()),
        "n_solidity_below_095": int((sol < 0.95).sum()),
        "fg_frac_p05": float(np.quantile(frac, .05)),
        "fg_frac_median": float(np.median(frac)),
        "fg_frac_p95": float(np.quantile(frac, .95)),
    })
    with open(os.path.join(OUT, "hardened_v2_log.json"), "w") as f:
        json.dump({"stats": stats, "per_class": per}, f, indent=2)
    print(f"[harden v2] xong {stats['n_refs']} ref | hull repair {stats['hull_repair']} | "
          f"ellipse {stats['ellipse_fallback']} | suspect {stats['suspect']} | "
          f"solidity min {stats['solidity_min']:.3f} median {stats['solidity_median']:.3f}",
          flush=True)


if __name__ == "__main__":
    main()
