"""
cure_fitness_check.py — 4 controls do do phu hop cua bo du lieu sau chuyen doi, tren FROZEN embeddings.

Control 1  low shot profile      : thống kê reference/consumer mỗi appearance class (không cần embedding).
Control 2  headroom + domain gap : cross domain top1/mAP (consumer -> reference K=1) và within domain
                                   top1 (consumer -> 1 consumer pseudo ref, seed 42). gap = within - cross.
Control 3  domain separability   : AUC logistic regression phân biệt domain (ref vs consumer), 5 fold CV.
Control 4  leakage               : cosine giữa mỗi ảnh consumer và reference CÙNG lớp; vượt ngưỡng
                                   (mặc định 0.985) coi là ảnh gốc của reference tổng hợp -> phải loại.

Ngoài 4 control còn có hai phép quét trùng lặp, cả hai KHÔNG gác cổng bằng centroid lớp:
  near_duplicate_scan : mọi cặp ảnh (mọi domain, mọi lớp) có cosine >= 0.99 ở bất kỳ probe nào.
                        Bản 1.0 chỉ đếm cặp ảnh SAU khi centroid hai lớp đã vượt 0.97, nên bỏ sót
                        viên 81: cùng một ảnh xếp dưới hai mặt, trong khi phần ảnh còn lại của mỗi
                        mặt khác nhau đủ để centroid không qua cổng (KNOWN_ISSUES.md mục 9).
  dedup_scan          : cặp lớp bị gắn cờ khi centroid consumer >= 0.97 ở mọi probe (như 1.0) HOẶC
                        khi có >= 3 cặp ảnh gần trùng chéo lớp theo phép quét trên.
Stage `partitions` (chạy sau build_cure_fewshot.py clean): mỗi query đánh giá so với MỌI ảnh huấn
luyện, và 20 cặp query/reference cùng lớp gần ngưỡng rò rỉ nhất, để soi tay.

Du lieu vao: manifest bbox dong bang (protocol/frozen/cure_crops_manifest.curated.csv) + cac crop
PNG 384 sinh lai tu manifest do. Xem REPRODUCE.md.
KHÔNG chạy trên ảnh raw: ảnh raw 2448/2976 vuông với viên thuốc rất nhỏ, embedding sẽ bị nền chi phối.

Backbone frozen: resnet50 + vit_base_patch16_224 (timm, pretrained, num_classes=0). Báo cáo từng model
và trung bình. Mọi phép đo tất định (seed 42, không shuffle ngoài RNG cố định).

Chạy:
  python cure_fitness_check.py extract              # trích embedding (cache .npy, chạy 1 lần)
  python cure_fitness_check.py controls             # control 1-4 + hai phép quét -> outputs/fitness_report.json
  python cure_fitness_check.py all                  # cả hai
  python cure_fitness_check.py partitions           # sau stage clean: eval so với train, cặp gần ngưỡng
  python cure_fitness_check.py controls --tag nosuspect --drop-refs 52__top,...
                                                    # controls không có một số reference (ghi *_nosuspect)

Toàn bộ đầu ra nằm trong ./outputs (embeddings/, fitness_report.json, leak_list.csv, dedup_map.csv,
near_duplicates.csv, cross_partition.json, nearest_pairs_for_review.csv).
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
DEDUP_IMG_THR = 0.99    # hai ảnh cosine >= mức này ở bất kỳ probe nào là 1 cặp gần trùng
DEDUP_MIN_PAIRS = 3     # cặp lớp bị gắn cờ khi có >= số cặp ảnh gần trùng chéo lớp này
XPART_THRESHOLDS = (0.985, 0.97, 0.95, 0.90)   # stage partitions: mốc đếm query gần ảnh train
REVIEW_TOPK = 20        # stage partitions: số cặp gần nhất xuất ra để soi tay


def _out_name(base, tag):
    """fitness_report.json + tag 'nosuspect' -> fitness_report_nosuspect.json."""
    if not tag:
        return base
    stem, ext = os.path.splitext(base)
    return f"{stem}_{tag}{ext}"


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
    # Probe đóng băng phải cho cùng một embedding trên mọi máy. Trên GPU Ampere trở lên, PyTorch
    # mặc định tính tích chập fp32 bằng TF32 (cudnn.allow_tf32=True): đo ngày 23/09/2026 trên
    # RTX 3080, embedding ResNet50 lệch tới 7e-3 so với máy tham chiếu và 138/8423 query đổi ảnh
    # láng giềng gần nhất (số query đúng top1 chỉ đổi 5, vì đổi cả hai chiều). Tắt TF32 thì lệch
    # còn 3e-6 và cả 8423 quyết định truy hồi trùng máy tham chiếu. Vô hại trên CPU và MPS.
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    print(f"[extract] device = {dev.type}; TF32 tắt (fp32 đúng nghĩa) để khớp máy tham chiếu", flush=True)
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


# ----------------------------------------------------------------------------- duplicate scans
def _pairs_at_or_above(e, thr, block=1024):
    """Mọi cặp (i < j) có e[i]·e[j] >= thr (embedding đã L2-norm). Trả dict {(i, j): cos}."""
    import numpy as np
    out = {}
    for s in range(0, len(e), block):
        S = e[s:s + block] @ e.T
        for r in range(S.shape[0]):
            i = s + r
            js = np.where(S[r, i + 1:] >= thr)[0] + i + 1
            for j in js:
                out[(i, int(j))] = float(S[r, j])
    return out


def near_duplicate_scan(df, embs, thr=DEDUP_IMG_THR):
    """Mọi cặp ảnh có cosine >= thr ở BẤT KỲ probe nào, không phân biệt domain, không gác cổng.
    Trả (tóm tắt, DataFrame cặp) — cặp chéo lớp là nhiễu nhãn mức ảnh, cặp cùng lớp là ảnh chụp
    lặp. Sắp theo cosine lớn nhất giảm dần rồi theo tên crop để tất định."""
    import pandas as pd
    per = {m: _pairs_at_or_above(e, thr) for m, e in embs.items()}
    keys = sorted(set().union(*[set(p) for p in per.values()]))
    cls = df["class_id"].to_numpy()
    dom = df["domain"].to_numpy()
    rel = df["crop_rel"].to_numpy()
    cos_cols = [f"cos_{m}" for m in embs]
    rows = []
    for i, j in keys:
        row = {"crop_a": rel[i], "crop_b": rel[j], "class_a": cls[i], "class_b": cls[j],
               "domain_a": dom[i], "domain_b": dom[j]}
        for m, e in embs.items():
            row[f"cos_{m}"] = per[m].get((i, j), float(e[i] @ e[j]))
        row["same_class"] = bool(cls[i] == cls[j])
        rows.append(row)
    cols = ["crop_a", "crop_b", "class_a", "class_b", "domain_a", "domain_b"] + cos_cols + ["same_class"]
    pairs = pd.DataFrame(rows, columns=cols)
    pairs["cos_max"] = pairs[cos_cols].max(axis=1) if len(pairs) else pd.Series(dtype=float)
    if len(pairs):
        pairs = pairs.sort_values(["cos_max", "crop_a", "crop_b"],
                                  ascending=[False, True, True]).reset_index(drop=True)
    cross = pairs[~pairs["same_class"].astype(bool)] if len(pairs) else pairs
    by_cp = {}
    for _, r in cross.iterrows():
        k = tuple(sorted((r["class_a"], r["class_b"])))
        by_cp[k] = by_cp.get(k, 0) + 1
    summary = {
        "threshold": float(thr),
        "n_pairs": int(len(pairs)),
        "n_pairs_cross_class": int(len(cross)),
        "n_pairs_same_class": int(len(pairs) - len(cross)),
        "per_model": {m: {"n_pairs": len(per[m]),
                          "n_pairs_cross_class": int(sum(1 for (i, j) in per[m] if cls[i] != cls[j]))}
                      for m in embs},
        "class_pairs": [{"class_a": a, "class_b": b, "n_pairs": n} for (a, b), n in sorted(by_cp.items())],
        "n_classes_in_cross_class_pairs": len({c for k in by_cp for c in k}),
        "cross_class_pairs": [
            {"crops": [os.path.basename(r["crop_a"]), os.path.basename(r["crop_b"])],
             "classes": [r["class_a"], r["class_b"]], "domains": [r["domain_a"], r["domain_b"]],
             **{c: round(float(r[c]), 6) for c in cos_cols}}
            for _, r in cross.iterrows()],
    }
    return summary, pairs


def dedup_scan(df, embs, cen_thr=DEDUP_CENTROID_THR, img_thr=DEDUP_IMG_THR,
               min_pairs=DEDUP_MIN_PAIRS, near_pairs=None):
    """Nhiễu nhãn chéo lớp, hai cò độc lập:
      centroid : centroid consumer của hai lớp cosine >= cen_thr ở MỌI probe (cò của bản 1.0);
      images   : >= min_pairs cặp ảnh chéo lớp cosine >= img_thr ở bất kỳ probe nào (mới từ 1.1).
    suggest = merge khi cả hai cò cùng kêu (hai lớp là một); resolve_images khi chỉ cò ảnh kêu
    (hai lớp khác nhau nhưng dùng chung ảnh, xử lý ở mức ảnh, xem duplicate_resolution.csv);
    review khi chỉ cò centroid kêu."""
    import numpy as np
    import pandas as pd
    if near_pairs is None:
        _, near_pairs = near_duplicate_scan(df, embs, img_thr)
    n_img = {}
    if len(near_pairs):
        for _, r in near_pairs[~near_pairs["same_class"].astype(bool)].iterrows():
            k = tuple(sorted((r["class_a"], r["class_b"])))
            n_img[k] = n_img.get(k, 0) + 1
    dfc = df[df["domain"] == "consumer"]
    cids = sorted(dfc["class_id"].unique())
    rows_of = {c: g.index.to_numpy() for c, g in dfc.groupby("class_id")}
    cen = {}
    for name, e in embs.items():
        C = np.stack([e[rows_of[c]].mean(0) for c in cids])
        C /= np.linalg.norm(C, axis=1, keepdims=True)
        cen[name] = C
    sims = {name: C @ C.T for name, C in cen.items()}
    pos = {c: i for i, c in enumerate(cids)}
    cand = {(cids[i], cids[j]) for i in range(len(cids)) for j in range(i + 1, len(cids))
            if all(sims[m][i, j] >= cen_thr for m in embs)}
    cand |= set(n_img)
    pairs = []
    for ca, cb in sorted(cand):
        cen_ok = (ca in pos and cb in pos
                  and all(sims[m][pos[ca], pos[cb]] >= cen_thr for m in embs))
        n_dup = int(n_img.get((ca, cb), 0))
        img_ok = n_dup >= min_pairs
        if not (cen_ok or img_ok):
            continue
        trigger = "both" if (cen_ok and img_ok) else ("centroid" if cen_ok else "images")
        suggest = "merge" if (cen_ok and img_ok) else ("resolve_images" if img_ok else "review")
        pairs.append({
            "class_a": ca, "class_b": cb,
            **{f"centroid_cos_{m}": (float(sims[m][pos[ca], pos[cb]])
                                     if (ca in pos and cb in pos) else float("nan")) for m in embs},
            "n_near_dup_images": n_dup, "trigger": trigger, "suggest": suggest,
        })
    dedup_df = pd.DataFrame(pairs, columns=(
        ["class_a", "class_b"] + [f"centroid_cos_{m}" for m in embs]
        + ["n_near_dup_images", "trigger", "suggest"]))
    n_cls = len(cids)
    return {"n_flagged_pairs": len(pairs),
            "n_suggest_merge": int(sum(p["suggest"] == "merge" for p in pairs)),
            "n_suggest_resolve_images": int(sum(p["suggest"] == "resolve_images" for p in pairs)),
            "n_trigger_centroid": int(sum(p["trigger"] in ("centroid", "both") for p in pairs)),
            "n_trigger_images": int(sum(p["trigger"] in ("images", "both") for p in pairs)),
            "flagged_class_frac": float(len({p["class_a"] for p in pairs}
                                            | {p["class_b"] for p in pairs}) / n_cls),
            "centroid_threshold": cen_thr, "image_threshold": img_thr, "min_pairs": min_pairs}, dedup_df


# ----------------------------------------------------------------------------- partitions
def _resolve_rels(df, rels):
    """crop_rel của file split -> chỉ số hàng trong df. Reference của bản hardened nằm trong
    manifest dưới đường dẫn tuyệt đối, nên khi không khớp tên thì tìm theo lớp (K=1)."""
    pos = {c: i for i, c in enumerate(df["crop_rel"])}
    ref_of = {}
    for i, (dom, cid) in enumerate(zip(df["domain"], df["class_id"])):
        if dom == "reference":
            ref_of.setdefault(cid, i)
    out = []
    for r in rels:
        if r in pos:
            out.append(pos[r])
            continue
        parts = str(r).replace("\\", "/").split("/")
        cid = None
        if "cure_crops" in parts:
            k = parts.index("cure_crops")
            if k + 2 < len(parts):
                cid = f"{parts[k + 1]}__{parts[k + 2]}"
        if cid in ref_of:
            out.append(ref_of[cid])
        else:
            raise SystemExit(f"[fitness] partitions: không tìm thấy {r} trong manifest")
    return out


def cross_partition_scan(df, embs, train_rels, eval_rels, train_labels=None, eval_labels=None,
                         thresholds=XPART_THRESHOLDS, topk=REVIEW_TOPK, block=256):
    """Mỗi query đánh giá so với MỌI ảnh huấn luyện (consumer lẫn reference). Báo cosine lớn
    nhất, số query vượt từng mốc, và topk cặp gần nhất để soi tay. Trùng lặp chéo partition
    là rò rỉ theo nghĩa đen; cặp gần nhất mà là hai viên khác nhau thì là độ khó thật."""
    import numpy as np
    import pandas as pd
    ti = _resolve_rels(df, train_rels)
    ei = _resolve_rels(df, eval_rels)
    cls = df["class_id"].to_numpy()
    tl = list(train_labels) if train_labels is not None else [cls[i] for i in ti]
    el = list(eval_labels) if eval_labels is not None else [cls[i] for i in ei]
    rel = df["crop_rel"].to_numpy()
    per_model, cand = {}, {}
    for m, e in embs.items():
        A, B = e[ei], e[ti]
        best = np.zeros(len(ei)); arg = np.zeros(len(ei), int)
        for s in range(0, len(ei), block):
            S = A[s:s + block] @ B.T
            best[s:s + block] = S.max(1); arg[s:s + block] = S.argmax(1)
        per_model[m] = {"max": float(best.max()), "median": float(np.median(best)),
                        "n_at_or_above": {str(t): int((best >= t).sum()) for t in thresholds}}
        for q in np.argsort(-best)[:topk]:
            cand.setdefault((int(q), int(arg[q])), {})[m] = float(best[q])
    rows = []
    for (q, t), cs in cand.items():
        full = {m: float(embs[m][ei[q]] @ embs[m][ti[t]]) for m in embs}
        rows.append({"eval_crop": rel[ei[q]], "train_crop": rel[ti[t]],
                     "eval_label": el[q], "train_label": tl[t],
                     **{f"cos_{m}": full[m] for m in embs}, "cos_max": max(full.values())})
    top = pd.DataFrame(rows, columns=["eval_crop", "train_crop", "eval_label", "train_label"]
                       + [f"cos_{m}" for m in embs] + ["cos_max"])
    if len(top):
        top = top.sort_values(["cos_max", "eval_crop"], ascending=[False, True]).head(topk).reset_index(drop=True)
    summary = {"n_train": len(ti), "n_eval": len(ei), "class_overlap": len(set(tl) & set(el)),
               "thresholds": [float(t) for t in thresholds], "per_model": per_model}
    return summary, top


def nearest_query_reference_pairs(df, embs, topk=REVIEW_TOPK):
    """topk cặp (ảnh consumer, reference cùng lớp) có cosine lớn nhất ở probe tốt nhất: đây là
    dải sát ngưỡng rò rỉ của control 4, để soi tay thay vì tin vào ngưỡng."""
    import numpy as np
    import pandas as pd
    is_ref = (df["domain"] == "reference").to_numpy()
    cls = df["class_id"].to_numpy()
    rel = df["crop_rel"].to_numpy()
    refs = {}
    for i in np.flatnonzero(is_ref):
        refs.setdefault(cls[i], []).append(i)
    rows = []
    for qi in np.flatnonzero(~is_ref):
        rs = refs.get(cls[qi])
        if not rs:
            continue
        row = {"crop_rel": rel[qi], "reference_crop_rel": None, "class_id": cls[qi]}
        best_ref, best_val = None, -2.0
        for m, e in embs.items():
            sims = e[rs] @ e[qi]
            k = int(np.argmax(sims))
            row[f"cos_{m}"] = float(sims[k])
            if sims[k] > best_val:
                best_val, best_ref = float(sims[k]), rel[rs[k]]
        row["reference_crop_rel"] = best_ref
        row["cos_max"] = best_val
        rows.append(row)
    top = pd.DataFrame(rows, columns=["crop_rel", "reference_crop_rel", "class_id"]
                       + [f"cos_{m}" for m in embs] + ["cos_max"])
    if len(top):
        top = top.sort_values(["cos_max", "crop_rel"], ascending=[False, True]).head(topk).reset_index(drop=True)
    return top


def drop_reference_rows(df, embs, classes):
    """Bỏ hàng reference của các lớp đã cho (consumer giữ nguyên), embedding cắt theo."""
    classes = set(classes)
    keep = ~((df["domain"] == "reference") & df["class_id"].isin(classes)).to_numpy()
    df2 = df[keep].reset_index(drop=True)
    return df2, {m: e[keep] for m, e in embs.items()}


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("stage", choices=["extract", "controls", "all", "partitions"])
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--device", default=None)
    ap.add_argument("--leak-thr", type=float, default=LEAK_THR)
    ap.add_argument("--tag", default="", help="hậu tố tên file đầu ra của controls/partitions")
    ap.add_argument("--drop-refs", default="",
                    help="controls: bỏ reference của các lớp này (phẩy ngăn cách) trước khi đo")
    ap.add_argument("--splits-dir", default=None,
                    help="partitions: thư mục chứa train_clean.csv và query_eval_fold0.csv (mặc định OUT)")
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
        dropped = [c for c in args.drop_refs.split(",") if c]
        if dropped:
            n0 = len(df)
            df, embs = drop_reference_rows(df, embs, dropped)
            print(f"[fitness] bỏ {n0 - len(df)} reference của {len(dropped)} lớp: {dropped}", flush=True)
        report = {"config": {"manifest": MANIFEST, "models": models, "img_size": IMG_SIZE,
                             "seed": SEED, "leak_threshold": args.leak_thr,
                             "dedup_centroid_threshold": DEDUP_CENTROID_THR,
                             "near_duplicate_threshold": DEDUP_IMG_THR,
                             "dropped_reference_classes": dropped}}
        print("[fitness] control 1 (low shot profile)...", flush=True)
        report["control1_lowshot"] = control1_lowshot(df)
        print("[fitness] control 2 (headroom + domain gap)...", flush=True)
        report["control2_headroom"] = control2_headroom(df, embs)
        print("[fitness] control 3 (domain separability)...", flush=True)
        report["control3_domain_sep"] = control3_domain_sep(df, embs)
        print("[fitness] control 4 (leakage)...", flush=True)
        report["control4_leakage"], leak_df = control4_leakage(df, embs, args.leak_thr)
        leak_df.to_csv(os.path.join(OUT, _out_name("leak_list.csv", args.tag)), index=False)
        print("[fitness] near duplicate scan (mọi cặp ảnh, không gác cổng)...", flush=True)
        report["near_duplicate_scan"], near_df = near_duplicate_scan(df, embs)
        near_df.to_csv(os.path.join(OUT, _out_name("near_duplicates.csv", args.tag)), index=False)
        print("[fitness] dedup scan (nhiễu nhãn chéo lớp, cò centroid và cò ảnh)...", flush=True)
        report["dedup_scan"], dedup_df = dedup_scan(df, embs, near_pairs=near_df)
        dedup_df.to_csv(os.path.join(OUT, _out_name("dedup_map.csv", args.tag)), index=False)
        rep_p = os.path.join(OUT, _out_name("fitness_report.json", args.tag))
        with open(rep_p, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"[fitness] xong -> {rep_p}", flush=True)
        nd = report["near_duplicate_scan"]
        print(f"  cặp ảnh >= {nd['threshold']}: {nd['n_pairs']} (chéo lớp {nd['n_pairs_cross_class']}, "
              f"cùng lớp {nd['n_pairs_same_class']}); cặp lớp gắn cờ: "
              f"{report['dedup_scan']['n_flagged_pairs']}", flush=True)
        c2 = report["control2_headroom"]["mean"]
        print(f"  cross top1={c2['cross_top1']:.3f}  mAP={c2['cross_mAP']:.3f}  "
              f"gap={c2['domain_gap_top1']:.3f}  domainAUC={report['control3_domain_sep']['mean_auc']:.3f}  "
              f"leak={report['control4_leakage']['union_leak_frac']:.4f}", flush=True)

    if args.stage == "partitions":
        import pandas as pd
        embs = load_embeddings(df, models)
        sd = args.splits_dir or OUT
        tr = pd.read_csv(os.path.join(sd, "train_clean.csv"))
        ev = pd.read_csv(os.path.join(sd, "query_eval_fold0.csv"))
        print(f"[fitness] partitions: {len(ev)} query đánh giá so với {len(tr)} ảnh huấn luyện", flush=True)
        xp, top = cross_partition_scan(df, embs, list(tr["crop_rel"]), list(ev["crop_rel"]),
                                       list(tr["label"]), list(ev["label"]))
        xp["splits_dir"] = sd
        xp["top_pairs"] = top.to_dict(orient="records")
        with open(os.path.join(OUT, _out_name("cross_partition.json", args.tag)), "w") as f:
            json.dump(xp, f, indent=2, ensure_ascii=False)
        nq = nearest_query_reference_pairs(df, embs)
        top = top.assign(kind="eval_query_vs_training_image")
        nq = nq.assign(kind="query_vs_own_reference")
        review = pd.concat([top.rename(columns={"eval_crop": "crop_rel", "train_crop": "other_crop_rel",
                                                "eval_label": "label", "train_label": "other_label"}),
                            nq.rename(columns={"reference_crop_rel": "other_crop_rel",
                                               "class_id": "label"})], ignore_index=True)
        review.to_csv(os.path.join(OUT, _out_name("nearest_pairs_for_review.csv", args.tag)), index=False)
        for m, v in xp["per_model"].items():
            print(f"  {m}: max {v['max']:.4f} | " + " ".join(
                f">={t}: {n}" for t, n in v["n_at_or_above"].items()), flush=True)
        print("[fitness] xong -> cross_partition.json, nearest_pairs_for_review.csv", flush=True)


if __name__ == "__main__":
    main()
