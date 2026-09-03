"""
build_cure_fewshot.py — chuyen bo anh thuoc thanh benchmark truy hoi few shot theo quy uoc ePillID.

Thực thi các phase của protocol, theo đúng thứ tự mô tả trong bài (xem REPRODUCE.md):

  stage splits (không cần embedding):
    Phase A  manifest_raw.csv                 (tu manifest bbox dong bang, xem protocol/frozen/)
    Phase B  manifest_classed.csv             (class_id = pillId__side; ghi cả quy ước one sided)
    Phase C  gallery_full.csv, query_full.csv (gallery = reference, query = consumer)
    Phase C' folds.csv, train_fold0.csv       (5 fold theo PILL, fold 0 held out — theo dung protocol danh gia cua ePillID:
             train = consumer fold 1-4 + reference của các pill train; gallery eval = TOÀN BỘ reference;
             query eval = consumer fold 0)
    Phase D  gallery_fewshot_K1.csv           (K=1 là cấu trúc tự nhiên của CURE: mỗi unit đúng 1 ref;
             lớp không có ref bị loại và ghi log. K=2 bất khả thi — đã bỏ theo ghi chú hiệu chỉnh.)

  stage clean (cần outputs/leak_list.csv + dedup_map.csv từ cure_fitness_check.py controls):
    Phase E  query_clean.csv                  (loại ảnh consumer là ảnh gốc của reference tổng hợp)
             manifest_dedup.csv               (cột class_id_merged sau khi gộp các cặp suggest=merge)
             query_eval_fold0.csv             (query fold 0 sau làm sạch)
             removal_log.json
    Phase G  eval_sample_1280.csv             (fixed sample đọc predictor: consumer sạch của fold 1-4,
             round robin theo lớp, seed 42, không shuffle ngoài RNG cố định — mirror ePillID Table A1)

Chạy:  python build_cure_fewshot.py splits
       python build_cure_fewshot.py clean
Đầu ra: ./outputs/*.csv
"""
import os
import json
import argparse

import cure_fitness_check as F

OUT = F.OUT
SEED = 42
N_FOLDS = 5
EVAL_SAMPLE_N = 1280


def _save(df, name, cols=None):
    p = os.path.join(OUT, name)
    (df[cols] if cols else df).to_csv(p, index=False)
    print(f"[build] {name}: {len(df)} dòng", flush=True)
    return p


# ------------------------------------------------------------------ stage splits
def stage_splits():
    import numpy as np
    os.makedirs(OUT, exist_ok=True)
    df = F.load_manifest()

    # Phase A + B
    _save(df, "manifest_raw.csv", ["pill_id", "side", "domain", "crop_rel"])
    df["class_id_oneside"] = df["pill_id"]
    _save(df, "manifest_classed.csv",
          ["pill_id", "side", "domain", "crop_rel", "class_id", "class_id_oneside"])
    print(f"[build] N_classes two sided = {df['class_id'].nunique()}, "
          f"one sided = {df['pill_id'].nunique()}", flush=True)

    # Phase C
    gal = df[df["domain"] == "reference"]
    qry = df[df["domain"] == "consumer"]
    _save(gal, "gallery_full.csv", ["crop_rel", "class_id"])
    _save(qry, "query_full.csv", ["crop_rel", "class_id"])

    # Phase D — K=1: giữ đúng 1 ref mỗi class (CURE vốn K=1; chọn tất định nếu dư), loại lớp không ref
    k1 = gal.sort_values("crop_rel").groupby("class_id", sort=True).head(1)
    no_ref = sorted(set(df["class_id"]) - set(k1["class_id"]))
    _save(k1, "gallery_fewshot_K1.csv", ["crop_rel", "class_id"])
    with open(os.path.join(OUT, "phaseD_log.json"), "w") as f:
        json.dump({"K": 1, "n_gallery": len(k1),
                   "classes_dropped_no_ref": no_ref,
                   "note": "K=2 bất khả thi: CURE chỉ có 1 reference mỗi (pill, side)"}, f,
                  indent=2, ensure_ascii=False)
    print(f"[build] K=1 gallery: {len(k1)} lớp; loại vì không có ref: {no_ref}", flush=True)

    # Phase C' — folds theo PILL (hai mặt của cùng viên đi cùng fold, tránh rò thông tin giữa
    # train và eval qua mặt còn lại), fold 0 held out làm query classes — theo protocol ePillID
    pills = sorted(df["pill_id"].unique(), key=lambda p: (len(p), p))
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(pills))
    # gán tất định: pill theo thứ tự đã permute, chia đều round robin
    fold_of = {pills[int(i)]: k % N_FOLDS for k, i in enumerate(order)}
    df["fold"] = df["pill_id"].map(fold_of)
    folds_df = df[["pill_id", "class_id", "fold"]].drop_duplicates().sort_values(
        ["fold", "pill_id", "class_id"])
    _save(folds_df, "folds.csv")

    # train gồm consumer fold!=0 + reference của pill train (dung cong thuc cua protocol ePillID)
    train = df[df["fold"] != 0]
    _save(train, "train_fold0.csv", ["crop_rel", "class_id", "domain", "fold"])
    q0 = df[(df["fold"] == 0) & (df["domain"] == "consumer")]
    _save(q0, "query_eval_fold0_preclean.csv", ["crop_rel", "class_id", "fold"])
    print(f"[build] fold 0 held out: {q0['class_id'].nunique()} lớp query, "
          f"train {train['class_id'].nunique()} lớp", flush=True)


# ------------------------------------------------------------------ stage clean
def _class_key(cid):
    """Sắp lớp theo số pill (20__top < 191__top), để lớp canonical sau merge là pill số nhỏ."""
    pill, _, side = str(cid).partition("__")
    return (0, int(pill), side) if pill.isdigit() else (1, pill, side)


def stage_clean():
    import numpy as np
    import pandas as pd
    need = ["leak_list.csv", "dedup_map.csv", "manifest_classed.csv", "folds.csv"]
    for n in need:
        if not os.path.exists(os.path.join(OUT, n)):
            raise SystemExit(f"[build] thiếu {n} — chạy `build_cure_fewshot.py splits` "
                             f"và `cure_fitness_check.py controls` trước")
    df = pd.read_csv(os.path.join(OUT, "manifest_classed.csv"))
    folds = pd.read_csv(os.path.join(OUT, "folds.csv"))[["class_id", "fold"]].drop_duplicates()
    df = df.merge(folds, on="class_id", how="left")
    leak = pd.read_csv(os.path.join(OUT, "leak_list.csv"))
    dedup = pd.read_csv(os.path.join(OUT, "dedup_map.csv"))

    # E.1 — de leak: loại ảnh consumer vượt ngưỡng cosine với reference cùng lớp
    leaked_set = set(leak.loc[leak["leaked"], "crop_rel"])
    df = df[~((df["domain"] == "consumer") & df["crop_rel"].isin(leaked_set))].copy()

    # E.2 — dedup: union find trên cặp suggest=merge (tự động) + manual_merges.csv (quyết định tay)
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            lo, hi = sorted([ra, rb], key=_class_key)
            parent[hi] = lo

    auto = dedup[dedup["suggest"] == "merge"] if len(dedup) else dedup
    for _, r in auto.iterrows():
        union(r["class_a"], r["class_b"])
    man_p = os.path.join(OUT, "manual_merges.csv")
    manual = pd.read_csv(man_p) if os.path.exists(man_p) else pd.DataFrame(columns=["class_a", "class_b"])
    for _, r in manual.iterrows():
        union(r["class_a"], r["class_b"])
    df["class_id_merged"] = df["class_id"].map(find)
    # fold sau merge = fold của lớp canonical (lớp gộp phải nằm trọn trong một fold)
    canon_fold = dict(zip(folds["class_id"], folds["fold"]))
    df["fold"] = df["class_id_merged"].map(canon_fold)
    _save(df, "manifest_dedup.csv",
          ["pill_id", "side", "domain", "crop_rel", "class_id", "class_id_merged", "fold"])

    # -------- các split cuối cùng (nhãn = class_id_merged) --------
    gal = df[df["domain"] == "reference"]
    qry = df[df["domain"] == "consumer"]
    _save(gal.rename(columns={"class_id_merged": "label"}), "gallery_eval_clean.csv",
          ["crop_rel", "label", "fold"])
    _save(qry.rename(columns={"class_id_merged": "label"}), "query_clean.csv",
          ["crop_rel", "label", "fold"])
    # Phase D thật sự có tác dụng sau merge: lớp gộp có 2 ref -> giữ 1 (tất định theo tên file)
    k1 = gal.sort_values("crop_rel").groupby("class_id_merged", sort=True).head(1)
    _save(k1.rename(columns={"class_id_merged": "label"}), "gallery_fewshot_K1_clean.csv",
          ["crop_rel", "label", "fold"])
    train = df[df["fold"] != 0]
    _save(train.rename(columns={"class_id_merged": "label"}), "train_clean.csv",
          ["crop_rel", "label", "domain", "fold"])
    q0 = qry[qry["fold"] == 0]
    _save(q0.rename(columns={"class_id_merged": "label"}), "query_eval_fold0.csv",
          ["crop_rel", "label", "fold"])

    # Phase G — eval sample 1280: consumer sạch fold 1-4, round robin theo lớp, seed 42
    pool = qry[qry["fold"] != 0].sort_values("crop_rel")
    rng = np.random.default_rng(SEED)
    per_class = {c: g["crop_rel"].tolist() for c, g in pool.groupby("class_id_merged", sort=True)}
    for c in per_class:
        per_class[c] = list(rng.permutation(per_class[c]))
    picked, classes = [], sorted(per_class)
    r = 0
    while len(picked) < min(EVAL_SAMPLE_N, len(pool)):
        for c in classes:
            if r < len(per_class[c]):
                picked.append(per_class[c][r])
                if len(picked) == EVAL_SAMPLE_N:
                    break
        r += 1
    es = pool[pool["crop_rel"].isin(picked)]
    _save(es.rename(columns={"class_id_merged": "label"}), "eval_sample_1280.csv",
          ["crop_rel", "label", "fold"])

    n_qry_before = int((pd.read_csv(os.path.join(OUT, "manifest_classed.csv"))["domain"]
                        == "consumer").sum())
    log = {
        "leak_threshold": float(F.LEAK_THR),
        "n_query_before": n_qry_before,
        "n_leaked_removed": int(n_qry_before - len(qry)),
        "leak_frac_before": float((n_qry_before - len(qry)) / max(n_qry_before, 1)),
        "leak_frac_after": 0.0,   # theo cấu trúc: mọi ảnh vượt ngưỡng đã bị loại
        "n_dedup_pairs_flagged": int(len(dedup)),
        "n_dedup_pairs_merged": int(len(auto) + len(manual)),
        "n_dedup_pairs_merged_manual": int(len(manual)),
        "n_classes_before_merge": int(df["class_id"].nunique()),
        "n_classes_after_merge": int(df["class_id_merged"].nunique()),
        "eval_sample_n": int(len(es)),
        "query_eval_fold0_n": int(len(q0)),
    }
    with open(os.path.join(OUT, "removal_log.json"), "w") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)

    # -------- metadata bộ processed (tính lại sau làm sạch + down sample K=1) --------
    per_cls = qry.groupby("class_id_merged").size()
    gal_cls = set(k1["class_id_merged"])
    summary = {
        "dataset": "CURE processed (crops 384, side as class, merged, K=1)",
        "n_images_total": int(len(df)),
        "n_classes": int(df["class_id_merged"].nunique()),
        "n_classes_with_gallery": int(len(gal_cls)),
        "classes_without_gallery": sorted(set(df["class_id_merged"]) - gal_cls,
                                          key=_class_key),
        "gallery_K1": int(len(k1)),
        "query_total": int(len(qry)),
        "query_per_class": {"min": int(per_cls.min()), "median": float(per_cls.median()),
                            "max": int(per_cls.max())},
        "train_fold0_heldout": {
            "train_images": int(len(train)),
            "train_consumer": int((train["domain"] == "consumer").sum()),
            "train_reference": int((train["domain"] == "reference").sum()),
            "train_classes": int(train["class_id_merged"].nunique()),
            "query_eval_images": int(len(q0)),
            "query_eval_classes": int(q0["class_id_merged"].nunique()),
        },
        "eval_sample": int(len(es)),
        "merged_pairs": {"auto": int(len(auto)), "manual": int(len(manual))},
        "leaked_removed": log["n_leaked_removed"],
    }
    with open(os.path.join(OUT, "cure_processed_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[build] clean xong: loại {log['n_leaked_removed']} ảnh leak, gộp "
          f"{log['n_dedup_pairs_merged']} cặp lớp ({len(manual)} tay), "
          f"{log['n_classes_before_merge']} -> {log['n_classes_after_merge']} lớp, "
          f"gallery K=1 = {len(k1)}", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Dựng splits CURE theo quy ước ePillID")
    ap.add_argument("stage", choices=["splits", "clean", "all"])
    args = ap.parse_args()
    if args.stage in ("splits", "all"):
        stage_splits()
    if args.stage in ("clean", "all"):
        stage_clean()


if __name__ == "__main__":
    main()
