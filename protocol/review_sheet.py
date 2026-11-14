#!/usr/bin/env python3
"""review_sheet.py — in contact sheet từ một CSV cặp ảnh để soi tay.

  python protocol/review_sheet.py outputs/nearest_pairs_for_review.csv --crops-root $CURE_CROPS_ROOT \\
      --out outputs/nearest_pairs_sheet.png

CSV cần các cột crop_rel, other_crop_rel, label, other_label, cos_max và tùy chọn kind (đúng như
`cure_fitness_check.py partitions` ghi). Mỗi hàng một cặp: ảnh trái, ảnh phải, dòng chữ ghi nhãn
hai bên, cosine và loại cặp. Script không quyết định gì; nó chỉ bày ra để người quyết định.
"""
import argparse
import os

import pandas as pd
from PIL import Image, ImageDraw

GAP = 8
HEADER = 16


def _open(rel, root, tile):
    p = rel if os.path.isabs(str(rel)) else os.path.join(root, str(rel))
    return Image.open(p).convert("RGB").resize((tile, tile))


def build(csv_path, crops_root, out_path, tile=160):
    d = pd.read_csv(csv_path)
    sheet = Image.new("RGB", (2 * tile + GAP, len(d) * (tile + HEADER)), "white")
    draw = ImageDraw.Draw(sheet)
    for i, r in d.reset_index(drop=True).iterrows():
        y = i * (tile + HEADER)
        kind = r.get("kind", "")
        same = "same label" if str(r["label"]) == str(r["other_label"]) else "DIFFERENT labels"
        draw.text((2, y + 2), f"{i + 1}. {os.path.basename(str(r['crop_rel']))} [{r['label']}] vs "
                              f"{os.path.basename(str(r['other_crop_rel']))} [{r['other_label']}]  "
                              f"cos {float(r['cos_max']):.3f}  {same}  {kind}", fill="black")
        sheet.paste(_open(r["crop_rel"], crops_root, tile), (0, y + HEADER))
        sheet.paste(_open(r["other_crop_rel"], crops_root, tile), (tile + GAP, y + HEADER))
    sheet.save(out_path)
    print(f"[sheet] {len(d)} cặp -> {out_path}")
    return len(d)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("csv")
    ap.add_argument("--crops-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tile", type=int, default=160)
    a = ap.parse_args()
    build(a.csv, a.crops_root, a.out, a.tile)


if __name__ == "__main__":
    main()
