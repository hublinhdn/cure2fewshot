"""
rubric_sensitivity.py — xoay từng ngưỡng của rubric và báo verdict đổi ở đâu.

Chạy sau khi đã có các fitness_report và rubric_score của từng biến thể:

  python protocol/rubric_sensitivity.py \\
      --variant direct=outputs/fitness_report.json:outputs/rubric_score_direct.json \\
      --variant v2=outputs_v2/fitness_v2/fitness_report.json:outputs_v2/rubric_v2/rubric_score_v2.json \\
      --variant v1=outputs_v1ungated/fitness_v1/fitness_report.json:outputs_v1ungated/rubric_v1/rubric_score_v1.json \\
      --removal-log outputs/removal_log.json --out outputs/rubric_sensitivity.json [--tex file.tex]

Mỗi ngưỡng xoay một mình, các ngưỡng khác giữ mặc định (one at a time). Với mỗi biến thể ghi:
điểm và lớp verdict ở mặc định; biên của từng số đo tới ngưỡng gần nhất; điểm và verdict ở từng
giá trị của từng ngưỡng; và danh sách mọi chỗ verdict đổi lớp. Các sự thật về file (splits, K=1,
gallery hardened) lấy từ rubric_score đã chấm, nên script không cần thư mục outputs.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score_rubric as SR  # noqa: E402

GRID = {
    "b1_full": [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90],
    "b2_full": [0.10, 0.12, 0.14, 0.15, 0.16, 0.18, 0.20],
    "b2_half": [0.03, 0.04, 0.05, 0.06, 0.08],
    "c1_full": [0.01, 0.02, 0.03, 0.05],
    "d1_hi": [0.80, 0.85, 0.90, 0.95],
    "e1_auc": [0.90, 0.95, 0.98, 0.995],
    "accept": [75, 80, 85],
    "cap_rule": [True, False],
}


def facts_from_rubric(base, rep):
    """Suy các sự thật về file từ điểm đã chấm: A1 = 10 nghĩa là split đúng chiều có mặt, A2 = 10
    nghĩa là gallery K=1 có mặt, E1 = 10 khi AUC vượt ngưỡng nghĩa là gallery hardened có mặt."""
    by = {i["name"][:2]: i["score"] for i in base["items"]}
    auc = rep["control3_domain_sep"]["mean_auc"]
    e1_thr = base.get("thresholds", SR.THRESHOLDS)["e1_auc"]
    return {"splits_ok": by["A1"] == 10, "k1_ok": by["A2"] == 10, "dedup_map": True,
            "hardened": by["E1"] == 10 and auc > e1_thr}


def margins(rep, rlog, thr=None):
    """Số đo của từng tiêu chí có ngưỡng, ngưỡng gần nhất và khoảng cách tới nó."""
    t = dict(SR.THRESHOLDS, **(thr or {}))
    c2 = rep["control2_headroom"]["mean"]
    auc = rep["control3_domain_sep"]["mean_auc"]
    residual = (rlog or {}).get("leak_frac_after", rep["control4_leakage"]["union_leak_frac"])
    meas = {"B1": (c2["cross_top1"], [t["b1_full"], t["b1_half"]]),
            "B2": (c2["domain_gap_top1"], [t["b2_half"], t["b2_full"]]),
            "C1": (residual, [t["c1_full"], t["c1_half"]]),
            "D1": (c2["cross_mAP"], [t["d1_lo"], t["d1_hi"]]),
            "E1": (auc, [t["e1_auc"]])}
    out = {}
    for k, (x, ths) in meas.items():
        nearest = min(ths, key=lambda v: abs(x - v))
        out[k] = {"measured": float(x), "nearest_threshold": float(nearest),
                  "margin": float(x - nearest)}
    return out


def sweep(variants, grid=GRID):
    """variants: {name: (rep, rlog, facts)}. Trả dict JSON hoá được."""
    res = {"grid": {k: v for k, v in grid.items()}, "variants": {}, "verdict_changes": []}
    for name, (rep, rlog, facts) in variants.items():
        base = SR.score(rep, rlog, facts)
        v = {"baseline": {"total": base["total"], "verdict_class": base["verdict_class"]},
             "margins": margins(rep, rlog), "sweeps": {}}
        for param, values in grid.items():
            rows = []
            for val in values:
                r = SR.score(rep, rlog, facts, {param: val})
                rows.append({"value": val, "total": r["total"], "verdict_class": r["verdict_class"]})
                if r["verdict_class"] != base["verdict_class"]:
                    res["verdict_changes"].append({"variant": name, "parameter": param, "value": val,
                                                   "from": base["verdict_class"], "to": r["verdict_class"],
                                                   "total": r["total"]})
            v["sweeps"][param] = rows
        res["variants"][name] = v
    return res


def to_tex(res, order=("direct", "v1", "v2"), labels=None):
    """Bảng LaTeX: mỗi hàng một giá trị ngưỡng, mỗi cột biến thể, ô = tổng và lớp verdict."""
    labels = labels or {"direct": "direct", "v1": "ungated mask", "v2": "gated mask"}
    names = [n for n in order if n in res["variants"]]
    short = {"ACCEPT": "A", "CAPPED": "C", "RESERVATION": "R", "REJECT": "X"}
    lines = ["\\begin{tabular}{ll" + "r" * len(names) + "}", "\\hline",
             "Threshold & Value & " + " & ".join(labels[n] for n in names) + " \\\\", "\\hline"]
    base = "Default & & " + " & ".join(
        f"{res['variants'][n]['baseline']['total']} {short[res['variants'][n]['baseline']['verdict_class']]}"
        for n in names) + " \\\\"
    lines.append(base)
    for param, values in res["grid"].items():
        for val in values:
            cells = []
            for n in names:
                row = next(r for r in res["variants"][n]["sweeps"][param] if r["value"] == val)
                cells.append(f"{row['total']} {short[row['verdict_class']]}")
            lines.append(f"{param.replace('_', ' ')} & {val} & " + " & ".join(cells) + " \\\\")
    lines += ["\\hline", "\\end{tabular}"]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--variant", action="append", required=True,
                    help="name=fitness_report.json:rubric_score.json")
    ap.add_argument("--removal-log", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tex", default=None)
    args = ap.parse_args()
    rlog = json.load(open(args.removal_log))
    variants = {}
    for spec in args.variant:
        name, paths = spec.split("=", 1)
        rep_p, base_p = paths.split(":", 1)
        rep, base = json.load(open(rep_p)), json.load(open(base_p))
        variants[name] = (rep, rlog, facts_from_rubric(base, rep))
    res = sweep(variants)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    for name, v in res["variants"].items():
        print(f"{name}: baseline {v['baseline']['total']} {v['baseline']['verdict_class']}; margins "
              + ", ".join(f"{k} {m['measured']:.3f} vs {m['nearest_threshold']} ({m['margin']:+.3f})"
                          for k, m in v["margins"].items()))
    print(f"verdict changes: {len(res['verdict_changes'])}")
    for c in res["verdict_changes"]:
        print(f"  {c['variant']}: {c['parameter']} = {c['value']} -> {c['to']} ({c['total']})")
    if args.tex:
        with open(args.tex, "w") as f:
            f.write(to_tex(res))
        print(f"-> {args.tex}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
