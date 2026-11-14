"""score_rubric.py as a function of its thresholds, and rubric_sensitivity.py on top of it."""
import json
import os

import score_rubric as SR
import rubric_sensitivity as RS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(ROOT, "expected")


def _load(variant):
    rep = json.load(open(os.path.join(EXP, f"fitness_report_{variant}.json")))
    base = json.load(open(os.path.join(EXP, f"rubric_score_{variant}.json")))
    rlog = json.load(open(os.path.join(EXP, "removal_log.json")))
    return rep, base, rlog


def test_default_thresholds_reproduce_every_shipped_score():
    for variant in ("direct", "v1", "v2"):
        rep, base, rlog = _load(variant)
        facts = RS.facts_from_rubric(base, rep)
        got = SR.score(rep, rlog, facts)
        assert got["total"] == base["total"], variant
        assert got["group_b"] == base["group_b"], variant
        assert got["group_b_capped"] == base["group_b_capped"], variant
        assert got["verdict"] == base["verdict"], variant
        assert [i["score"] for i in got["items"]] == [i["score"] for i in base["items"]], variant


def test_verdict_class_names_the_band():
    rep, base, rlog = _load("direct")
    got = SR.score(rep, rlog, RS.facts_from_rubric(base, rep))
    assert got["verdict_class"] == "CAPPED"
    rep, base, rlog = _load("v2")
    assert SR.score(rep, rlog, RS.facts_from_rubric(base, rep))["verdict_class"] == "ACCEPT"


def test_lowering_the_partial_gap_threshold_flips_the_direct_verdict():
    rep, base, rlog = _load("direct")
    thr = dict(SR.THRESHOLDS, b2_half=0.03)
    got = SR.score(rep, rlog, RS.facts_from_rubric(base, rep), thr)
    assert got["total"] == 87
    assert got["verdict_class"] == "ACCEPT"


def test_switching_the_cap_off_accepts_the_direct_conversion_at_80():
    rep, base, rlog = _load("direct")
    thr = dict(SR.THRESHOLDS, cap_rule=False)
    got = SR.score(rep, rlog, RS.facts_from_rubric(base, rep), thr)
    assert got["total"] == 80 and got["verdict_class"] == "ACCEPT"


def test_resolved_image_pairs_count_as_handled_in_c2():
    rep, base, rlog = _load("direct")
    rep = json.loads(json.dumps(rep))
    rep["dedup_scan"]["n_flagged_pairs"] = 5
    rlog = dict(rlog, n_dedup_pairs_resolved_images=1)
    got = SR.score(rep, rlog, RS.facts_from_rubric(base, rep))
    c2 = [i for i in got["items"] if i["name"].startswith("C2")][0]
    assert c2["score"] == 10
    assert "còn 2 cặp" in c2["note"]


def test_sweep_reports_margins_and_verdict_changes():
    variants = {v: _load(v) for v in ("direct", "v1", "v2")}
    res = RS.sweep({v: (rep, rlog, RS.facts_from_rubric(base, rep)) for v, (rep, base, rlog) in variants.items()})
    d = res["variants"]["direct"]
    assert d["baseline"] == {"total": 80, "verdict_class": "CAPPED"}
    assert abs(d["margins"]["B2"]["measured"] - 0.044) < 0.001
    assert d["margins"]["B2"]["nearest_threshold"] == 0.05
    changes = {(c["variant"], c["parameter"], c["value"]) for c in res["verdict_changes"]}
    assert ("direct", "b2_half", 0.03) in changes
    assert all(c["variant"] != "v2" for c in res["verdict_changes"])
    assert res["variants"]["v1"]["baseline"]["total"] == 100
    v1_full = {r["value"]: r["total"] for r in res["variants"]["v1"]["sweeps"]["b2_full"]}
    assert v1_full[0.18] == 92 and v1_full[0.16] == 100
