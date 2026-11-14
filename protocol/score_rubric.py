"""
score_rubric.py — cham do phu hop cua bo du lieu sau chuyen doi (conversion fitness rubric, 100 diem).

Đọc outputs/fitness_report.json + removal_log.json + các file split đã dựng, in bảng điểm,
xep hang ACCEPT / ACCEPT WITH RESERVATION / REJECT, va ghi outputs/rubric_score.json.

Dieu kien chan: bat ky tieu chi nao trong nhom B (giu do kho giua hai nguon anh) bang 0 thi tong
diem du cao van ha tran xuong muc "mild domain shift probe", vi mat do kho thi phep do truy hoi
khong con phan biet duoc mo hinh.

Mọi ngưỡng nằm trong THRESHOLDS và được ghi vào file điểm; `rubric_sensitivity.py` xoay từng
ngưỡng trên cùng hàm `score()` để báo verdict đổi ở đâu. Lý do chọn trọng số và ngưỡng: xem
bài báo kèm theo (mục rubric) và REPRODUCE.md mục 6.
"""
import os
import json
import argparse

OUT = os.environ.get("CURE_FITNESS_OUT",
                     os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs"))

# Ngưỡng của từng tiêu chí. B1 và B2 là nhóm chịu tải: cross top1 của probe đóng băng phải nằm
# dưới trần (dưới 0.75 trọn điểm, dưới 0.90 nửa điểm), và gap within - cross phải đủ lớn (0.15
# trọn, 0.05 nửa). C: rò rỉ còn lại và cặp lớp chưa xử lý. D: mAP trong dải khó nhưng học được.
# E: AUC phân biệt domain, trên 0.95 thì phải có gallery hardened. accept/reserve: biên xếp hạng.
# cap_rule: một tiêu chí nhóm B bằng 0 thì hạ trần xuống dải giữa dù tổng cao.
THRESHOLDS = {
    "b1_full": 0.75, "b1_half": 0.90,
    "b2_full": 0.15, "b2_half": 0.05,
    "c1_full": 0.02, "c1_half": 0.05,
    "c2_full": 0.05, "c2_half": 0.10,
    "d1_lo": 0.05, "d1_hi": 0.90,
    "e1_auc": 0.95,
    "accept": 80, "reserve": 60,
    "cap_rule": True,
}

VERDICTS = {
    "ACCEPT": "ACCEPT: suitable as a primary few shot retrieval benchmark",
    "RESERVATION": "ACCEPT WITH RESERVATION: usable only as a mild domain shift probe; declare the limitation",
    "REJECT": "REJECT: the converted dataset does not meet the fitness bar",
    "CAPPED": ("CAPPED to mild domain shift probe: a group B criterion is zero, so the two "
               "image sources are not materially separated or the task is too easy"),
}


def file_facts(out_dir):
    """Bốn sự thật về file mà rubric cần, đọc từ thư mục CURE_FITNESS_OUT (xem REPRODUCE.md mục 6)."""
    ex = lambda n: os.path.exists(os.path.join(out_dir, n))   # noqa: E731
    return {"splits_ok": ex("gallery_full.csv") and ex("query_full.csv"),
            "k1_ok": ex("gallery_fewshot_K1.csv"),
            "dedup_map": ex("dedup_map.csv"),
            "hardened": ex("gallery_fewshot_K1_hardened.csv")}


def score(rep, rlog, facts, thr=None):
    """Chấm một báo cáo controls. rep: fitness_report.json đã đọc; rlog: removal_log.json đã đọc
    hoặc None; facts: dict của file_facts(); thr: dict ngưỡng (mặc định THRESHOLDS).
    Trả dict items/total/group_b/group_b_capped/verdict/verdict_class."""
    t = dict(THRESHOLDS, **(thr or {}))
    c1 = rep["control1_lowshot"]
    c2 = rep["control2_headroom"]["mean"]
    auc = rep["control3_domain_sep"]["mean_auc"]
    c4 = rep["control4_leakage"]
    dd = rep["dedup_scan"]
    items = []

    # ---- A. Tương đương cấu trúc (30)
    ok_dir = bool(facts.get("splits_ok"))
    items.append(("A1 gallery=reference, query=consumer, mAP trên query", 10 if ok_dir else 0, 10,
                  "files split đúng chiều" if ok_dir else "chưa dựng splits"))
    med_ref = c1["ref_per_class"]["median"]
    a2 = 10 if (facts.get("k1_ok") and med_ref == 1) else 0
    items.append(("A2 regime few shot (median ref mỗi lớp = K = 1)", a2, 10,
                  f"median ref/class = {med_ref} (K=1 là cấu trúc tự nhiên của CURE)"))
    a3 = 10 if (c1["n_classes_two_sided"] > c1["n_classes_one_sided"]) else 0
    items.append(("A3 side as class, báo N_classes hai quy ước", a3, 10,
                  f"two sided {c1['n_classes_two_sided']} / one sided {c1['n_classes_one_sided']}"))

    # ---- B. Giữ độ khó cross domain (30) — ĐIỀU KIỆN CHẶN
    cross_t1 = c2["cross_top1"]
    b1 = 15 if cross_t1 < t["b1_full"] else (7 if cross_t1 < t["b1_half"] else 0)
    items.append(("B1 frozen cross domain top1 dưới ceiling", b1, 15, f"cross top1 = {cross_t1:.3f}"))
    gap = c2["domain_gap_top1"]
    b2 = 15 if gap >= t["b2_full"] else (7 if gap >= t["b2_half"] else 0)
    items.append(("B2 domain gap đủ lớn (within - cross top1)", b2, 15, f"gap = {gap:.3f}"))

    # ---- C. Sạch (20)
    if rlog is not None:
        residual = rlog.get("leak_frac_after", 0.0)
        c1s = 10 if residual <= t["c1_full"] else (5 if residual <= t["c1_half"] else 0)
        note = (f"leak trước = {rlog['leak_frac_before']:.4f}, đã loại "
                f"{rlog['n_leaked_removed']} ảnh, sau = {residual:.4f}")
    else:
        raw = c4["union_leak_frac"]
        c1s = 10 if raw <= t["c1_full"] else (5 if raw <= t["c1_half"] else 0)
        note = f"CHƯA de leak (stage clean chưa chạy), leak thô = {raw:.4f}"
    items.append(("C1 leakage đã loại (residual <= 0.02)", c1s, 10, note))

    dedup_done = rlog is not None and bool(facts.get("dedup_map"))
    if dedup_done:
        merged = rlog["n_dedup_pairs_merged"]
        resolved = rlog.get("n_dedup_pairs_resolved_images", 0)
        review_left = dd["n_flagged_pairs"] - merged - resolved
        frac_left = review_left / max(c1["n_classes_two_sided"], 1)
        c2s = 10 if frac_left <= t["c2_full"] else (5 if frac_left <= t["c2_half"] else 0)
        note = (f"gộp {merged} cặp" + (f", xử lý mức ảnh {resolved} cặp" if resolved else "")
                + f", còn {review_left} cặp mức review ({frac_left:.3f} so với tổng lớp)")
    else:
        c2s = 0
        note = f"dedup_map có {dd['n_flagged_pairs']} cặp cờ nhưng CHƯA áp (stage clean chưa chạy)"
    items.append(("C2 nhiễu nhãn đã dedup", c2s, 10, note))

    # ---- D. So sánh được với ePillID (10)
    cross_map = c2["cross_mAP"]
    d1 = 10 if (t["d1_lo"] < cross_map < t["d1_hi"]) else 0
    items.append(("D1 mAP frozen cross domain trong dải khó nhưng học được", d1, 10,
                  f"cross mAP = {cross_map:.3f} (yêu cầu trong ({t['d1_lo']}, {t['d1_hi']}))"))

    # ---- E. Cảnh báo shortcut nền (10)
    if auc <= t["e1_auc"]:
        e1, enote = 10, f"domain AUC = {auc:.3f} <= {t['e1_auc']}, không cần làm cứng nền"
    elif facts.get("hardened"):
        e1, enote = 10, f"domain AUC = {auc:.3f} > {t['e1_auc']} nhưng đã có gallery hardened"
    else:
        e1, enote = 5, (f"domain AUC = {auc:.3f} > {t['e1_auc']}, Phase F chưa bật -> phải ghi rõ "
                        f"hạn chế mild shift trong Methods")
    items.append(("E1 báo cáo domain AUC + xử lý shortcut nền", e1, 10, enote))

    total = sum(s for _, s, _, _ in items)
    group_b = b1 + b2
    if total >= t["accept"]:
        cls = "ACCEPT"
    elif total >= t["reserve"]:
        cls = "RESERVATION"
    else:
        cls = "REJECT"
    capped = False
    if t["cap_rule"] and (b1 == 0 or b2 == 0) and total >= t["reserve"]:
        cls, capped = "CAPPED", True
    return {"items": [{"name": n, "score": s, "max": mx, "note": note} for n, s, mx, note in items],
            "total": total, "group_b": group_b, "group_b_capped": capped,
            "verdict": VERDICTS[cls], "verdict_class": cls, "thresholds": t}


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
    res = score(rep, rlog, file_facts(OUT))

    items = res["items"]
    w = max(len(i["name"]) for i in items)
    print("=" * (w + 30))
    for i in items:
        print(f"{i['name']:<{w}}  {i['score']:>2}/{i['max']:<2}  {i['note']}")
    print("=" * (w + 30))
    print(f"{'TỔNG':<{w}}  {res['total']:>2}/100  nhóm B = {res['group_b']}/30")
    print(f"XẾP HẠNG: {res['verdict']}")

    out_name = f"rubric_score{'_' + args.tag if args.tag else ''}.json"
    with open(os.path.join(OUT, out_name), "w") as f:
        json.dump({**res, "report": args.report}, f, indent=2, ensure_ascii=False)
    print(f"-> outputs/{out_name}")


if __name__ == "__main__":
    main()
