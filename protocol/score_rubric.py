"""
score_rubric.py — cham do phu hop cua bo du lieu sau chuyen doi (conversion fitness rubric, 100 diem).

Đọc outputs/fitness_report.json + removal_log.json + các file split đã dựng, in bảng điểm,
xep hang ACCEPT / ACCEPT WITH RESERVATION / REJECT, va ghi outputs/rubric_score.json.

Dieu kien chan: nhom B (giu do kho giua hai nguon anh) bang 0 thi tong diem du cao van ha tran,
vi mat do kho thi phep do truy hoi khong con phan biet duoc mo hinh.
"""
import os
import json
import argparse

OUT = os.environ.get("CURE_FITNESS_OUT",
                     os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default=os.path.join(OUT, "fitness_report.json"),
                    help="fitness_report.json cần chấm (vd outputs/hardened/fitness_report.json)")
    ap.add_argument("--tag", default="", help="hậu tố tên file điểm (vd hardened)")
    args = ap.parse_args()
    with open(args.report) as f:
        rep = json.load(f)
    log_p = os.path.join(OUT, "removal_log.json")
    rlog = json.load(open(log_p)) if os.path.exists(log_p) else None

    c1 = rep["control1_lowshot"]
    c2 = rep["control2_headroom"]["mean"]
    auc = rep["control3_domain_sep"]["mean_auc"]
    c4 = rep["control4_leakage"]
    dd = rep["dedup_scan"]

    items = []

    # ---- A. Tương đương cấu trúc (30)
    ok_dir = (os.path.exists(os.path.join(OUT, "gallery_full.csv"))
              and os.path.exists(os.path.join(OUT, "query_full.csv")))
    items.append(("A1 gallery=reference, query=consumer, mAP trên query", 10 if ok_dir else 0, 10,
                  "files split đúng chiều" if ok_dir else "chưa dựng splits"))

    k1_ok = os.path.exists(os.path.join(OUT, "gallery_fewshot_K1.csv"))
    med_ref = c1["ref_per_class"]["median"]
    a2 = 10 if (k1_ok and med_ref == 1) else 0
    items.append(("A2 regime few shot (median ref mỗi lớp = K = 1)", a2, 10,
                  f"median ref/class = {med_ref} (K=1 là cấu trúc tự nhiên của CURE)"))

    a3 = 10 if (c1["n_classes_two_sided"] > c1["n_classes_one_sided"]) else 0
    items.append(("A3 side as class, báo N_classes hai quy ước", a3, 10,
                  f"two sided {c1['n_classes_two_sided']} / one sided {c1['n_classes_one_sided']}"))

    # ---- B. Giữ độ khó cross domain (30) — ĐIỀU KIỆN CHẶN
    cross_t1 = c2["cross_top1"]
    b1 = 15 if cross_t1 < 0.75 else (7 if cross_t1 < 0.90 else 0)
    items.append(("B1 frozen cross domain top1 dưới ceiling", b1, 15, f"cross top1 = {cross_t1:.3f}"))

    gap = c2["domain_gap_top1"]
    b2 = 15 if gap >= 0.15 else (7 if gap >= 0.05 else 0)
    items.append(("B2 domain gap đủ lớn (within - cross top1)", b2, 15, f"gap = {gap:.3f}"))

    # ---- C. Sạch (20)
    if rlog is not None:
        residual = rlog.get("leak_frac_after", 0.0)
        c1s = 10 if residual <= 0.02 else (5 if residual <= 0.05 else 0)
        note = (f"leak trước = {rlog['leak_frac_before']:.4f}, đã loại "
                f"{rlog['n_leaked_removed']} ảnh, sau = {residual:.4f}")
    else:
        raw = c4["union_leak_frac"]
        c1s = 10 if raw <= 0.02 else (5 if raw <= 0.05 else 0)
        note = f"CHƯA de leak (stage clean chưa chạy), leak thô = {raw:.4f}"
    items.append(("C1 leakage đã loại (residual <= 0.02)", c1s, 10, note))

    dedup_done = rlog is not None and os.path.exists(os.path.join(OUT, "dedup_map.csv"))
    residual_flag = dd["flagged_class_frac"] if not dedup_done else max(
        0.0, dd["flagged_class_frac"] - 0)  # sau merge: cặp suggest=merge đã gộp, còn cặp review
    if dedup_done:
        merged = rlog["n_dedup_pairs_merged"]
        review_left = dd["n_flagged_pairs"] - merged
        frac_left = review_left / max(rep["control1_lowshot"]["n_classes_two_sided"], 1)
        c2s = 10 if frac_left <= 0.05 else (5 if frac_left <= 0.10 else 0)
        note = (f"gộp {merged} cặp, còn {review_left} cặp mức review "
                f"({frac_left:.3f} so với tổng lớp)")
    else:
        c2s = 0
        note = f"dedup_map có {dd['n_flagged_pairs']} cặp cờ nhưng CHƯA áp (stage clean chưa chạy)"
    items.append(("C2 nhiễu nhãn đã dedup", c2s, 10, note))

    # ---- D. So sánh được với ePillID (10)
    cross_map = c2["cross_mAP"]
    d1 = 10 if (0.05 < cross_map < 0.90) else 0
    items.append(("D1 mAP frozen cross domain trong dải khó nhưng học được", d1, 10,
                  f"cross mAP = {cross_map:.3f} (yêu cầu trong (0.05, 0.90))"))

    # ---- E. Cảnh báo shortcut nền (10)
    hardened = os.path.exists(os.path.join(OUT, "gallery_fewshot_K1_hardened.csv"))
    if auc <= 0.95:
        e1, enote = 10, f"domain AUC = {auc:.3f} <= 0.95, không cần làm cứng nền"
    elif hardened:
        e1, enote = 10, f"domain AUC = {auc:.3f} > 0.95 nhưng đã có gallery hardened"
    else:
        e1, enote = 5, (f"domain AUC = {auc:.3f} > 0.95, Phase F chưa bật -> phải ghi rõ "
                        f"hạn chế mild shift trong Methods")
    items.append(("E1 báo cáo domain AUC + xử lý shortcut nền", e1, 10, enote))

    total = sum(s for _, s, _, _ in items)
    group_b = b1 + b2
    if total >= 80:
        verdict = "ACCEPT: suitable as a primary few shot retrieval benchmark"
    elif total >= 60:
        verdict = "ACCEPT WITH RESERVATION: usable only as a mild domain shift probe; declare the limitation"
    else:
        verdict = "REJECT: the converted dataset does not meet the fitness bar"
    capped = False
    if group_b == 0 and total >= 60:
        verdict = "CAPPED to mild domain shift probe: group B is zero, the difficulty property is absent"
        capped = True

    w = max(len(n) for n, _, _, _ in items)
    print("=" * (w + 30))
    for name, s, mx, note in items:
        print(f"{name:<{w}}  {s:>2}/{mx:<2}  {note}")
    print("=" * (w + 30))
    print(f"{'TỔNG':<{w}}  {total:>2}/100  nhóm B = {group_b}/30")
    print(f"XẾP HẠNG: {verdict}")

    out_name = f"rubric_score{'_' + args.tag if args.tag else ''}.json"
    with open(os.path.join(OUT, out_name), "w") as f:
        json.dump({"items": [{"name": n, "score": s, "max": mx, "note": note}
                             for n, s, mx, note in items],
                   "report": args.report,
                   "total": total, "group_b": group_b, "group_b_capped": capped,
                   "verdict": verdict}, f, indent=2, ensure_ascii=False)
    print(f"-> outputs/{out_name}")


if __name__ == "__main__":
    main()
