"""
cure_fitness_check.py — 4 controls do do phu hop cua bo du lieu sau chuyen doi, tren FROZEN embeddings.

Control 1  low shot profile      : thống kê reference/consumer mỗi appearance class (không cần embedding).
Control 2  headroom + domain gap : cross domain top1/mAP (consumer -> reference K=1) và within domain
                                   top1 (consumer -> 1 consumer pseudo ref, seed 42). gap = within - cross.
Control 3  domain separability   : AUC logistic regression phân biệt domain (ref vs consumer), 5 fold CV.
Control 4  leakage               : cosine giữa mỗi ảnh consumer và reference CÙNG lớp; vượt ngưỡng
                                   (mặc định 0.985) coi là ảnh gốc của reference tổng hợp -> phải loại.

Ngoài 4 control còn có dedup scan (nhiễu nhãn chéo lớp): các cặp lớp có centroid consumer cosine cao
kèm nhiều cặp ảnh gần trùng chéo lớp.

Du lieu vao: manifest bbox dong bang (protocol/frozen/cure_crops_manifest.curated.csv) + cac crop
PNG 384 sinh lai tu manifest do. Xem REPRODUCE.md.
KHÔNG chạy trên ảnh raw: ảnh raw 2448/2976 vuông với viên thuốc rất nhỏ, embedding sẽ bị nền chi phối.

Backbone frozen: resnet50 + vit_base_patch16_224 (timm, pretrained, num_classes=0). Báo cáo từng model
và trung bình. Mọi phép đo tất định (seed 42, không shuffle ngoài RNG cố định).

Chạy:
  python cure_fitness_check.py extract              # trích embedding (cache .npy, chạy 1 lần)
  python cure_fitness_check.py controls             # control 1-4 + dedup scan -> outputs/fitness_report.json
  python cure_fitness_check.py all                  # cả hai

Toàn bộ đầu ra nằm trong ./outputs (embeddings/, fitness_report.json, leak_list.csv, dedup_map.csv).
Module top là stdlib + pandas/numpy; torch/timm chỉ import trong stage extract.
"""
import os
import sys
import json
import argparse

THIS = os.path.dirname(os.path.abspath(__file__))
PILL_PROJ = os.environ.get(
    "PILL_PROJ", os.path.join(os.path.dirname(THIS), "data"))
# PILL_PROJ = goc du lieu: thu muc chua data/raw/CURE_dataset va data/processed/cure_crops.
# Dat bang bien moi truong PILL_PROJ, xem REPRODUCE.md.
MANIFEST = os.environ.get(
    "CURE_MANIFEST", os.path.join(PILL_PROJ, "data", "processed", "cure_crops_manifest.curated.csv"))
OUT = os.environ.get("CURE_FITNESS_OUT", os.path.join(THIS, "outputs"))
EMB_DIR = os.path.join(OUT, "embeddings")
EMB_REUSE = os.environ.get("CURE_EMB_REUSE")   # dir embeddings của run gốc để tái dùng theo crop_rel

MODELS = ["resnet50", "vit_base_patch16_224"]
IMG_SIZE = 224          # crop PNG 384 -> resize 224 (input chuẩn của cả hai backbone frozen)
BATCH = 64
SEED = 42
LEAK_THR = float(os.environ.get("LEAK_COSINE", "0.985"))
DEDUP_CENTROID_THR = float(os.environ.get("DEDUP_CENTROID", "0.97"))
DEDUP_IMG_THR = 0.99    # ảnh chéo lớp cosine >= mức này tính là 1 cặp gần trùng
DEDUP_MIN_PAIRS = 3     # cặp lớp bị gắn cờ khi có >= số cặp ảnh gần trùng này (kèm centroid cao)


# ----------------------------------------------------------------------------- manifest
def load_manifest(path=MANIFEST):
    """Đọc manifest crop curated -> DataFrame chuẩn hóa:
    class_id = pillId__side (two sided, quy ước ePillID), domain in {reference, consumer},
    abs_path = đường dẫn tuyệt đối tới crop PNG."""
    import pandas as pd
    if not os.path.exists(path):
        raise SystemExit(f"[fitness] không thấy manifest: {path} (set CURE_MANIFEST / PILL_PROJ)")
    m = pd.read_csv(path)
    need = {"crop_rel", "pillId", "side", "kind"}
    missing = need - set(m.columns)
    if missing:
        raise SystemExit(f"[fitness] manifest thiếu cột {missing}")
    df = pd.DataFrame({
        "pill_id": m["pillId"].astype(str),
        "side": m["side"].astype(str).str.lower(),
        "domain": m["kind"].map(lambda k: "reference" if str(k) == "ref" else "consumer"),
        "crop_rel": m["crop_rel"],
        # crop_rel tuyệt đối (ảnh hardened) giữ nguyên; tương đối thì ghép CROPS_ROOT
        # (env CURE_CROPS_ROOT — cho workspace server độc lập; mặc định = PILL_PROJ)
        "abs_path": m["crop_rel"].map(lambda r: r if os.path.isabs(r)
                                      else os.path.join(
                                          os.environ.get("CURE_CROPS_ROOT", PILL_PROJ), r)),
    })
    df["class_id"] = df["pill_id"] + "__" + df["side"]
    df = df.sort_values("crop_rel").reset_index(drop=True)   # thứ tự tất định cho mọi stage
    return df


def check_crops_exist(df, sample=200):
    """Kiểm nhanh crop PNG đã được rebuild chưa (kiểm đều toàn dải, không chỉ đầu danh sách)."""
    idx = range(0, len(df), max(1, len(df) // sample))
    miss = [df["abs_path"].iloc[i] for i in idx if not os.path.exists(df["abs_path"].iloc[i])]
    return miss


# ----------------------------------------------------------------------------- embeddings
def emb_path(model_name):
    return os.path.join(EMB_DIR, f"{model_name}.npy")


def extract(df, model_names=MODELS, device=None, batch=BATCH):
    """Trích embedding frozen cho mọi crop, cache ra outputs/embeddings/<model>.npy
    (hàng i tương ứng df hàng i; df đã sort theo crop_rel). Kèm index CSV để đối chiếu."""
    import numpy as np
    import torch
    import timm
    from PIL import Image

    os.makedirs(EMB_DIR, exist_ok=True)
    if device is None:
        device = ("cuda" if torch.cuda.is_available()
                  else "mps" if torch.backends.mps.is_available() else "cpu")
    dev = torch.device(device)
    df[["crop_rel", "class_id", "domain"]].to_csv(os.path.join(EMB_DIR, "index.csv"), index=False)

    for name in model_names:
        out = emb_path(name)
        if os.path.exists(out):
            arr = np.load(out, mmap_mode="r")
            if arr.shape[0] == len(df):
                print(f"[extract] {name}: cache sẵn ({arr.shape}) — bỏ qua", flush=True)
                continue
            print(f"[extract] {name}: cache lệch ({arr.shape[0]} vs {len(df)}) — trích lại", flush=True)
        # tái dùng embedding của run gốc cho các crop_rel trùng (CURE_EMB_REUSE)
        reuse_emb, reuse_row = None, {}
        if EMB_REUSE and os.path.exists(os.path.join(EMB_REUSE, f"{name}.npy")):
            import pandas as pd
            ridx = pd.read_csv(os.path.join(EMB_REUSE, "index.csv"))
            reuse_row = {c: i for i, c in enumerate(ridx["crop_rel"])}
            reuse_emb = np.load(os.path.join(EMB_REUSE, f"{name}.npy"))
        todo = [i for i, c in enumerate(df["crop_rel"]) if c not in reuse_row]
        print(f"[extract] {name}: tính mới {len(todo)}/{len(df)} "
              f"(tái dùng {len(df) - len(todo)})", flush=True)

        model = timm.create_model(name, pretrained=True, num_classes=0).eval().to(dev)
        cfg = timm.data.resolve_model_data_config(model)
        tfm = timm.data.create_transform(**cfg, is_training=False)
        emb = None
        if reuse_emb is not None:
            emb = np.zeros((len(df), reuse_emb.shape[1]), np.float32)
            for i, c in enumerate(df["crop_rel"]):
                if c in reuse_row:
                    emb[i] = reuse_emb[reuse_row[c]]
        with torch.inference_mode():
            buf, buf_idx = [], []
            iterate = todo if reuse_emb is not None else range(len(df))
            for k, i in enumerate(iterate):
                img = Image.open(df["abs_path"].iloc[i]).convert("RGB")
                buf.append(tfm(img)); buf_idx.append(i)
                if len(buf) == batch or k == len(iterate) - 1:
                    x = torch.stack(buf).to(dev)
                    f = torch.nn.functional.normalize(model(x).float(), dim=1).cpu().numpy()
                    if emb is None:
                        emb = np.zeros((len(df), f.shape[1]), np.float32)
                    emb[buf_idx] = f
                    buf, buf_idx = [], []
                    if (k + 1) % (batch * 20) < batch:
                        print(f"[extract] {name}: {k+1}/{len(iterate)}", flush=True)
        np.save(out, emb)
        print(f"[extract] {name}: xong {emb.shape} -> {out}", flush=True)
        del model
        if dev.type == "mps":
            torch.mps.empty_cache()


def load_embeddings(df, model_names=MODELS):
    import numpy as np
    embs = {}
    for name in model_names:
        p = emb_path(name)
        if not os.path.exists(p):
            raise SystemExit(f"[fitness] thiếu embedding {p} — chạy stage extract trước")
        e = np.load(p)
        if e.shape[0] != len(df):
            raise SystemExit(f"[fitness] {name}: embedding {e.shape[0]} hàng vs manifest {len(df)}")
        embs[name] = e
    return embs


# ----------------------------------------------------------------------------- control 1
def control1_lowshot(df):
    import numpy as np
    g = df.groupby(["class_id", "domain"]).size().unstack(fill_value=0)
    for col in ("reference", "consumer"):
        if col not in g:
            g[col] = 0
    refs, cons = g["reference"], g["consumer"]
    return {
        "n_images": int(len(df)),
        "n_pill_ids": int(df["pill_id"].nunique()),
        "n_classes_two_sided": int(df["class_id"].nunique()),
        "n_classes_one_sided": int(df["pill_id"].nunique()),
        "n_reference": int((df["domain"] == "reference").sum()),
        "n_consumer": int((df["domain"] == "consumer").sum()),
        "ref_per_class": {"min": int(refs.min()), "median": float(refs.median()),
                          "max": int(refs.max())},
        "consumer_per_class": {"min": int(cons.min()), "median": float(cons.median()),
                               "max": int(cons.max())},
        "classes_without_ref": sorted(g.index[refs == 0].tolist()),
        "classes_without_consumer": sorted(g.index[cons == 0].tolist()),
    }


# ----------------------------------------------------------------------------- retrieval core
def _class_codes(series):
    import numpy as np
    cats = sorted(series.unique())
    lut = {c: i for i, c in enumerate(cats)}
    return np.array([lut[v] for v in series]), cats


def _retrieval(q_emb, q_lab, g_emb, g_lab, block=1024):
    """top1 / MRR-mAP / recall@5 cho truy hồi cosine (embedding đã L2-norm).
    Gallery mỗi lớp >= 1 ảnh; điểm lớp = max ảnh trong lớp. Trả thêm rank từng query."""
    import numpy as np
    n_g_classes = len(np.unique(g_lab))
    top1 = np.zeros(len(q_emb), bool)
    rr = np.zeros(len(q_emb))
    rec5 = np.zeros(len(q_emb), bool)
    valid = np.isin(q_lab, g_lab)
    for s in range(0, len(q_emb), block):
        e = min(s + block, len(q_emb))
        S = q_emb[s:e] @ g_emb.T                      # (b, n_gallery)
        # điểm theo lớp: max similarity trong lớp
        order = np.argsort(g_lab)
        gl_sorted = g_lab[order]
        S_sorted = S[:, order]
        bounds = np.searchsorted(gl_sorted, np.unique(gl_sorted))
        cls_scores = np.maximum.reduceat(S_sorted, bounds, axis=1)   # (b, n_classes)
        cls_ids = np.unique(gl_sorted)
        rank_order = np.argsort(-cls_scores, axis=1)
        for bi, qi in enumerate(range(s, e)):
            if not valid[qi]:
                rr[qi] = 0.0
                continue
            ranked = cls_ids[rank_order[bi]]
            pos = int(np.where(ranked == q_lab[qi])[0][0]) + 1
            top1[qi] = pos == 1
            rec5[qi] = pos <= 5
            rr[qi] = 1.0 / pos
    n = int(valid.sum())
    return {"top1": float(top1[valid].mean()), "mAP": float(rr[valid].mean()),
            "recall@5": float(rec5[valid].mean()), "n_query": n,
            "n_gallery_classes": int(n_g_classes)}, rr


def control2_headroom(df, embs, seed=SEED):
    """Cross domain (consumer -> reference K=1) vs within domain (consumer -> 1 consumer pseudo ref).
    Trả per model + mean; gap = within top1 - cross top1."""
    import numpy as np
    lab, _ = _class_codes(df["class_id"])
    is_ref = (df["domain"] == "reference").to_numpy()
    res = {"per_model": {}, "mean": {}}
    rng = np.random.default_rng(seed)
    # pseudo gallery within domain: 1 ảnh consumer mỗi lớp, chọn tất định theo seed
    pseudo_idx = []
    dfc = df[~df["domain"].eq("reference")]
    for cid, grp in dfc.groupby("class_id", sort=True):
        rows = grp.index.to_numpy()
        pseudo_idx.append(int(rng.choice(rows)))
    pseudo_mask = np.zeros(len(df), bool)
    pseudo_mask[pseudo_idx] = True

    for name, e in embs.items():
        cross, _ = _retrieval(e[~is_ref], lab[~is_ref], e[is_ref], lab[is_ref])
        within, _ = _retrieval(e[~is_ref & ~pseudo_mask], lab[~is_ref & ~pseudo_mask],
                               e[pseudo_mask], lab[pseudo_mask])
        res["per_model"][name] = {
            "cross_domain": cross, "within_domain": within,
            "domain_gap_top1": float(within["top1"] - cross["top1"]),
        }
    for key in ("top1", "mAP", "recall@5"):
        res["mean"][f"cross_{key}"] = float(np.mean(
            [res["per_model"][m]["cross_domain"][key] for m in embs]))
        res["mean"][f"within_{key}"] = float(np.mean(
            [res["per_model"][m]["within_domain"][key] for m in embs]))
    res["mean"]["domain_gap_top1"] = float(np.mean(
        [res["per_model"][m]["domain_gap_top1"] for m in embs]))
    return res


# ----------------------------------------------------------------------------- control 3
def control3_domain_sep(df, embs, seed=SEED):
    """AUC phân biệt reference vs consumer bằng logistic regression, 5 fold stratified CV.
    AUC cao = nền/hình thái reference tách bạch -> model có thể ăn gian bằng shortcut domain."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    y = (df["domain"] == "reference").astype(int).to_numpy()
    out = {"per_model": {}, "mean_auc": None}
    aucs_all = []
    for name, e in embs.items():
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        scores = np.zeros(len(y))
        for tr, te in skf.split(e, y):
            clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
            clf.fit(e[tr], y[tr])
            scores[te] = clf.decision_function(e[te])
        auc = float(roc_auc_score(y, scores))
        out["per_model"][name] = {"auc": auc}
        aucs_all.append(auc)
    out["mean_auc"] = float(np.mean(aucs_all))
    return out


# ----------------------------------------------------------------------------- control 4
def control4_leakage(df, embs, thr=LEAK_THR):
    """Với mỗi ảnh consumer: cosine tới reference CÙNG lớp (K=1). Vượt ngưỡng -> ảnh gốc mà
    reference tổng hợp sinh ra từ đó -> phải loại khỏi query. Cờ leak = union giữa các model."""
    import numpy as np
    is_ref = (df["domain"] == "reference").to_numpy()
    lab, _ = _class_codes(df["class_id"])
    ref_of = {}                                   # class code -> index reference (K=1)
    for i in np.where(is_ref)[0]:
        ref_of.setdefault(lab[i], []).append(i)
    con_idx = np.where(~is_ref)[0]
    per_model = {}
    leak_union = np.zeros(len(con_idx), bool)
    sim_by_model = {}
    for name, e in embs.items():
        sims = np.full(len(con_idx), np.nan)
        for j, qi in enumerate(con_idx):
            refs = ref_of.get(lab[qi])
            if refs:
                sims[j] = float(np.max(e[refs] @ e[qi]))
        leaked = sims >= thr
        per_model[name] = {
            "threshold": thr,
            "leak_frac": float(np.nanmean(leaked.astype(float))),
            "n_leaked": int(np.nansum(leaked)),
            "same_class_cos_p50": float(np.nanmedian(sims)),
            "same_class_cos_p95": float(np.nanquantile(sims, 0.95)),
            "same_class_cos_max": float(np.nanmax(sims)),
        }
        leak_union |= np.nan_to_num(leaked, nan=False)
        sim_by_model[name] = sims
    import pandas as pd
    leak_df = pd.DataFrame({
        "crop_rel": df["crop_rel"].to_numpy()[con_idx],
        "class_id": df["class_id"].to_numpy()[con_idx],
        **{f"cos_{m}": sim_by_model[m] for m in embs},
        "leaked": leak_union,
    })
    return {"per_model": per_model,
            "union_leak_frac": float(leak_union.mean()),
            "union_n_leaked": int(leak_union.sum()),
            "n_query_checked": int(len(con_idx))}, leak_df


# ----------------------------------------------------------------------------- dedup scan
def dedup_scan(df, embs, cen_thr=DEDUP_CENTROID_THR, img_thr=DEDUP_IMG_THR,
               min_pairs=DEDUP_MIN_PAIRS):
    """Nhiễu nhãn chéo lớp: cặp class_id có centroid consumer cosine >= cen_thr ở MỌI model,
    kèm số cặp ảnh chéo lớp cosine >= img_thr. Đề xuất merge khi đủ cả hai điều kiện."""
    import numpy as np
    import pandas as pd
    dfc = df[df["domain"] == "consumer"]
    cids = sorted(dfc["class_id"].unique())
    rows_of = {c: g.index.to_numpy() for c, g in dfc.groupby("class_id")}
    cen = {}
    for name, e in embs.items():
        C = np.stack([e[rows_of[c]].mean(0) for c in cids])
        C /= np.linalg.norm(C, axis=1, keepdims=True)
        cen[name] = C
    sims = {name: C @ C.T for name, C in cen.items()}
    pairs = []
    for i in range(len(cids)):
        for j in range(i + 1, len(cids)):
            if all(sims[m][i, j] >= cen_thr for m in embs):
                # đếm cặp ảnh gần trùng chéo lớp (model đầu tiên là đủ cho việc đếm)
                m0 = next(iter(embs))
                S = embs[m0][rows_of[cids[i]]] @ embs[m0][rows_of[cids[j]]].T
                n_dup = int((S >= img_thr).sum())
                pairs.append({
                    "class_a": cids[i], "class_b": cids[j],
                    **{f"centroid_cos_{m}": float(sims[m][i, j]) for m in embs},
                    "n_near_dup_images": n_dup,
                    "suggest": "merge" if n_dup >= min_pairs else "review",
                })
    dedup_df = pd.DataFrame(pairs, columns=(
        ["class_a", "class_b"] + [f"centroid_cos_{m}" for m in embs]
        + ["n_near_dup_images", "suggest"]))
    n_cls = len(cids)
    return {"n_flagged_pairs": len(pairs),
            "n_suggest_merge": int((dedup_df["suggest"] == "merge").sum()) if len(pairs) else 0,
            "flagged_class_frac": float(len({p["class_a"] for p in pairs}
                                            | {p["class_b"] for p in pairs}) / n_cls),
            "centroid_threshold": cen_thr, "image_threshold": img_thr}, dedup_df


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("stage", choices=["extract", "controls", "all"])
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--device", default=None)
    ap.add_argument("--leak-thr", type=float, default=LEAK_THR)
    args = ap.parse_args()
    models = args.models.split(",")

    os.makedirs(OUT, exist_ok=True)
    df = load_manifest()
    miss = check_crops_exist(df)
    if miss:
        raise SystemExit(f"[fitness] {len(miss)} crop chưa tồn tại (mẫu kiểm đều), ví dụ:\n  "
                         + "\n  ".join(miss[:5])
                         + "\n-> chạy rebuild_cure_crops.py với manifest curated trước.")
    print(f"[fitness] manifest: {len(df)} crops, {df['class_id'].nunique()} class, "
          f"{(df['domain'] == 'reference').sum()} ref / {(df['domain'] == 'consumer').sum()} consumer",
          flush=True)

    if args.stage in ("extract", "all"):
        extract(df, models, device=args.device)
    if args.stage in ("controls", "all"):
        embs = load_embeddings(df, models)
        report = {"config": {"manifest": MANIFEST, "models": models, "img_size": IMG_SIZE,
                             "seed": SEED, "leak_threshold": args.leak_thr,
                             "dedup_centroid_threshold": DEDUP_CENTROID_THR}}
        print("[fitness] control 1 (low shot profile)...", flush=True)
        report["control1_lowshot"] = control1_lowshot(df)
        print("[fitness] control 2 (headroom + domain gap)...", flush=True)
        report["control2_headroom"] = control2_headroom(df, embs)
        print("[fitness] control 3 (domain separability)...", flush=True)
        report["control3_domain_sep"] = control3_domain_sep(df, embs)
        print("[fitness] control 4 (leakage)...", flush=True)
        report["control4_leakage"], leak_df = control4_leakage(df, embs, args.leak_thr)
        leak_df.to_csv(os.path.join(OUT, "leak_list.csv"), index=False)
        print("[fitness] dedup scan (nhiễu nhãn chéo lớp)...", flush=True)
        report["dedup_scan"], dedup_df = dedup_scan(df, embs)
        dedup_df.to_csv(os.path.join(OUT, "dedup_map.csv"), index=False)
        with open(os.path.join(OUT, "fitness_report.json"), "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"[fitness] xong -> {os.path.join(OUT, 'fitness_report.json')}", flush=True)
        c2 = report["control2_headroom"]["mean"]
        print(f"  cross top1={c2['cross_top1']:.3f}  mAP={c2['cross_mAP']:.3f}  "
              f"gap={c2['domain_gap_top1']:.3f}  domainAUC={report['control3_domain_sep']['mean_auc']:.3f}  "
              f"leak={report['control4_leakage']['union_leak_frac']:.4f}", flush=True)


if __name__ == "__main__":
    main()
