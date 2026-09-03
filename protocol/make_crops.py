"""
make_crops.py — Phase A: sinh lại toàn bộ crop 384 từ manifest bbox đóng băng.

Đây là bước ĐẦU TIÊN của protocol và là bước duy nhất cần đến ảnh gốc. Mọi script còn lại
(cure_fitness_check.py, build_cure_fewshot.py, harden_refs_*.py) đọc các crop do bước này sinh ra.

Vì sao repo phát hành bbox chứ không phát hành crop: cắt theo một bbox đã ghi là phép biến đổi
ảnh thuần, nên người đọc nào có bộ ảnh gốc cũng dựng lại được đúng từng crop mà không phải cài
hay chạy mô hình phân đoạn. Nhờ vậy mô hình phân đoạn không nằm trong tập phụ thuộc của protocol,
và repo này không phát hành lại một pixel nào của bộ ảnh gốc.

Ba chế độ:

  inventory  Chỉ kiểm ảnh gốc, không ghi gì: mọi `src_rel` phải tồn tại và đúng `orig_w`x`orig_h`.
             Chạy cái này TRƯỚC. Bộ ảnh không mang số phiên bản nên đây là cách xác lập tính
             toàn vẹn: một bản lưu trữ khác đi âm thầm sẽ làm mọi bbox đã ghi trở nên vô nghĩa.
             Lưu ý: chỉ kiểm 8811 file mà protocol thực sự dùng, không kiểm cả 8971 file trong
             bản phân phối.

  build      Cắt và ghi những crop còn thiếu vào `CURE_CROPS_ROOT/<crop_rel>`. Bỏ qua file đã có.

  verify     Cắt lại trong bộ nhớ rồi so TỪNG PIXEL với crop đã có trên đĩa. Đây là bước kiểm
             tính tất định của Phase A.

  all        inventory, rồi build, rồi verify.

Biến môi trường (xem REPRODUCE.md):
  CURE_MANIFEST     manifest bbox đóng băng. Mặc định: protocol/frozen/cure_crops_manifest.curated.csv
  CURE_RAW_ROOT     gốc chứa `src_rel`, tức thư mục có data/raw/CURE_dataset/
  CURE_CROPS_ROOT   gốc chứa `crop_rel`, nơi ghi crop. Mặc định: bằng CURE_RAW_ROOT

Chạy:
  python make_crops.py inventory
  python make_crops.py build
  python make_crops.py verify

CẢNH BÁO PHIÊN BẢN: letterbox có resample, nên crop trùng tới từng pixel đòi ĐÚNG phiên bản
Pillow đã ghi trong requirements.txt (12.0.0). Phiên bản khác cho ra cùng bbox và cùng nội dung
nhưng có thể lệch vài pixel ở biên đệm; khi đó `verify` sẽ báo lệch. Việc đó không ảnh hưởng
splits hay điểm rubric, vốn chỉ phụ thuộc manifest và embedding.
"""
import os
import csv
import sys
import argparse
from collections import Counter

THIS = os.path.dirname(os.path.abspath(__file__))
FROZEN = os.path.join(THIS, "frozen")

MANIFEST = os.environ.get(
    "CURE_MANIFEST", os.path.join(FROZEN, "cure_crops_manifest.curated.csv"))
RAW_ROOT = os.environ.get("CURE_RAW_ROOT")
CROPS_ROOT = os.environ.get("CURE_CROPS_ROOT", RAW_ROOT)

PAD_FILL = (127, 127, 127)   # xám letterbox, y hệt bản đã sinh ra các số trong expected/
LOG_EVERY = 1000


def _need_roots():
    if not RAW_ROOT:
        raise SystemExit(
            "[crops] thiếu CURE_RAW_ROOT — thư mục chứa data/raw/CURE_dataset/. Xem REPRODUCE.md")
    if not os.path.isdir(RAW_ROOT):
        raise SystemExit(f"[crops] CURE_RAW_ROOT không phải thư mục: {RAW_ROOT}")


def _rows():
    if not os.path.exists(MANIFEST):
        raise SystemExit(f"[crops] không thấy manifest: {MANIFEST} (đặt CURE_MANIFEST)")
    with open(MANIFEST, newline="") as f:
        rows = list(csv.DictReader(f))
    need = {"crop_rel", "src_rel", "orig_w", "orig_h", "x0", "y0", "x1", "y1", "img_size"}
    missing = need - set(rows[0].keys())
    if missing:
        raise SystemExit(f"[crops] manifest thiếu cột {sorted(missing)}")
    return rows


def _letterbox(img, size):
    """Thumbnail LANCZOS rồi đệm xám 127 cho vuông. Giữ y bản đã sinh ra expected/."""
    from PIL import Image, ImageOps
    img = img.copy()
    img.thumbnail((size, size), Image.LANCZOS)
    dw, dh = size - img.size[0], size - img.size[1]
    return ImageOps.expand(
        img, (dw // 2, dh // 2, dw - dw // 2, dh - dh // 2), fill=PAD_FILL)


def _crop_one(row):
    """Mở ảnh gốc, kiểm kích thước, cắt theo bbox, letterbox. Trả PIL.Image RGB."""
    from PIL import Image
    src = os.path.join(RAW_ROOT, row["src_rel"])
    img = Image.open(src).convert("RGB")
    exp = (int(row["orig_w"]), int(row["orig_h"]))
    if img.size != exp:
        raise ValueError(f"kích thước {img.size}, manifest ghi {exp}")
    box = (int(row["x0"]), int(row["y0"]), int(row["x1"]), int(row["y1"]))
    return _letterbox(img.crop(box), int(row["img_size"])).convert("RGB")


# --------------------------------------------------------------------------- inventory
def stage_inventory(rows):
    from PIL import Image
    print(f"[crops] inventory: {len(rows)} file ảnh gốc mà protocol dùng", flush=True)
    tally, bad = Counter(), []
    for i, r in enumerate(rows):
        src = os.path.join(RAW_ROOT, r["src_rel"])
        if not os.path.exists(src):
            tally["missing"] += 1
            bad.append(("missing", r["src_rel"], ""))
        else:
            try:
                with Image.open(src) as im:
                    got = im.size
                exp = (int(r["orig_w"]), int(r["orig_h"]))
                if got != exp:
                    tally["size_mismatch"] += 1
                    bad.append(("size", r["src_rel"], f"{got} != {exp}"))
                else:
                    tally["ok"] += 1
            except Exception as e:                       # ảnh hỏng, không đọc được
                tally["unreadable"] += 1
                bad.append(("unreadable", r["src_rel"], str(e)[:80]))
        if (i + 1) % LOG_EVERY == 0:
            print(f"[crops] inventory {i+1}/{len(rows)}", flush=True)
    print(f"[crops] inventory xong: {dict(tally)}", flush=True)
    if bad:
        for kind, rel, extra in bad[:20]:
            print(f"[crops]   {kind}: {rel} {extra}", flush=True)
        if len(bad) > 20:
            print(f"[crops]   ... còn {len(bad)-20} dòng nữa", flush=True)
        raise SystemExit(
            "[crops] INVENTORY FAIL — bản ảnh gốc khác bản mà manifest được dựng trên đó. "
            "Tải lại bộ ảnh từ tác giả gốc trước khi đi tiếp.")
    print("[crops] INVENTORY PASS", flush=True)


# ------------------------------------------------------------------------------- build
def stage_build(rows):
    print(f"[crops] build: {len(rows)} crop -> {CROPS_ROOT}", flush=True)
    tally, bad = Counter(), []
    for i, r in enumerate(rows):
        dst = os.path.join(CROPS_ROOT, r["crop_rel"])
        if os.path.exists(dst):
            tally["exists"] += 1
        else:
            try:
                crop = _crop_one(r)
            except Exception as e:
                tally["failed"] += 1
                bad.append((r["src_rel"], str(e)[:80]))
                continue
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            crop.save(dst, format="PNG", optimize=False)
            tally["built"] += 1
        if (i + 1) % LOG_EVERY == 0:
            print(f"[crops] build {i+1}/{len(rows)}", flush=True)
    print(f"[crops] build xong: {dict(tally)}", flush=True)
    if bad:
        for rel, err in bad[:20]:
            print(f"[crops]   fail: {rel} {err}", flush=True)
        raise SystemExit("[crops] BUILD FAIL — chạy `inventory` để biết ảnh gốc thiếu gì.")
    print("[crops] BUILD PASS", flush=True)


# ------------------------------------------------------------------------------ verify
def stage_verify(rows):
    import numpy as np
    from PIL import Image
    print(f"[crops] verify: so từng pixel {len(rows)} crop", flush=True)
    tally, bad = Counter(), []
    for i, r in enumerate(rows):
        dst = os.path.join(CROPS_ROOT, r["crop_rel"])
        if not os.path.exists(dst):
            tally["absent"] += 1
            bad.append((r["crop_rel"], "chưa có crop, chạy `build` trước"))
        else:
            try:
                want = np.asarray(_crop_one(r))
            except Exception as e:
                tally["failed"] += 1
                bad.append((r["crop_rel"], str(e)[:80]))
                continue
            got = np.asarray(Image.open(dst).convert("RGB"))
            if got.shape != want.shape:
                tally["shape_differs"] += 1
                bad.append((r["crop_rel"], f"{got.shape} != {want.shape}"))
            elif np.array_equal(got, want):
                tally["identical"] += 1
            else:
                n = int((got != want).any(axis=2).sum())
                tally["pixels_differ"] += 1
                bad.append((r["crop_rel"], f"{n} pixel lệch"))
        if (i + 1) % LOG_EVERY == 0:
            print(f"[crops] verify {i+1}/{len(rows)}", flush=True)
    print(f"[crops] verify xong: {dict(tally)}", flush=True)
    if bad:
        for rel, why in bad[:20]:
            print(f"[crops]   {rel}: {why}", flush=True)
        if len(bad) > 20:
            print(f"[crops]   ... còn {len(bad)-20} dòng nữa", flush=True)
        if tally["pixels_differ"] and not (tally["absent"] or tally["failed"]
                                          or tally["shape_differs"]):
            print("[crops] Lệch chỉ ở mức pixel: gần như chắc chắn do khác phiên bản Pillow. "
                  "Ghim pillow==12.0.0 rồi chạy lại. Splits và điểm rubric KHÔNG bị ảnh hưởng.",
                  flush=True)
        raise SystemExit("[crops] VERIFY FAIL")
    print("[crops] VERIFY PASS — Phase A tái lập trùng tới từng pixel", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Phase A: sinh lại crop 384 từ manifest bbox")
    ap.add_argument("stage", choices=["inventory", "build", "verify", "all"])
    args = ap.parse_args()
    _need_roots()
    rows = _rows()
    print(f"[crops] manifest = {MANIFEST} ({len(rows)} dòng)", flush=True)
    print(f"[crops] raw root = {RAW_ROOT}", flush=True)
    print(f"[crops] crops root = {CROPS_ROOT}", flush=True)
    if args.stage in ("inventory", "all"):
        stage_inventory(rows)
    if args.stage in ("build", "all"):
        stage_build(rows)
    if args.stage in ("verify", "all"):
        stage_verify(rows)


if __name__ == "__main__":
    sys.exit(main())
