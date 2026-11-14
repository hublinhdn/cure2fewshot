"""
make_duplicate_resolution.py — từ source_duplicates.json (make_crops.py duplicates) sinh file quyết
định duplicate_resolution.csv mà build_cure_fewshot.py clean đọc.

Quy tắc, cố định và ghi vào từng dòng: trong mỗi nhóm ảnh trùng byte, GIỮ bản nằm dưới lớp xếp
trước theo thứ tự canonical của bước gộp nhãn (số viên tăng dần, rồi bottom trước top), BỎ các
bản còn lại. Với viên 81 hai mặt không phân biệt được bằng ảnh (hai reference của nó có cosine
0.9996 ở cả hai probe, cao nhất trong 196 viên), nên mọi quy tắc đều là quy ước; quy ước này là
quy ước đã có sẵn trong mã, tất định và kiểm được. Bản đóng băng nằm ở protocol/frozen/.

  python protocol/make_duplicate_resolution.py outputs/source_duplicates.json protocol/frozen/duplicate_resolution.csv
"""
import csv
import json
import sys

RULE = "keep the copy under the class that sorts first"


def _class_key(cid):
    pill, _, side = str(cid).partition("__")
    return (0, int(pill), side) if pill.isdigit() else (1, pill, side)


def build(scan_path, out_path):
    with open(scan_path) as f:
        scan = json.load(f)
    rows = []
    for k, g in enumerate(scan["groups"], 1):
        members = sorted(zip(g["classes"], g["crops"], g["files"]),
                         key=lambda t: (_class_key(t[0]), t[1]))
        keep_c, keep_crop, keep_src = members[0]
        note = ("same photograph filed under two classes" if len(set(g["classes"])) > 1
                else "same photograph filed twice under one class")
        for drop_c, drop_crop, drop_src in members[1:]:
            rows.append({"group": k, "keep_crop_rel": keep_crop, "drop_crop_rel": drop_crop,
                         "keep_class": keep_c, "drop_class": drop_c, "rule": RULE, "note": note})
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["group", "keep_crop_rel", "drop_crop_rel", "keep_class",
                                          "drop_class", "rule", "note"])
        w.writeheader()
        w.writerows(rows)
    print(f"[dup] {len(scan['groups'])} nhóm -> {len(rows)} dòng bỏ -> {out_path}")
    return len(rows)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    build(sys.argv[1], sys.argv[2])
