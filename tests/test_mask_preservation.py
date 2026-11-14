"""measure_mask_damage.py against an independent mask, and reference_masks.py helpers."""
import numpy as np

import measure_mask_damage as MD
import reference_masks as RM


def _disc(h, w, cy, cx, r):
    Y, X = np.ogrid[:h, :w]
    return (Y - cy) ** 2 + (X - cx) ** 2 <= r * r


def test_preservation_stats_on_an_intact_pill_is_clean():
    mask = _disc(64, 64, 32, 32, 20)
    keep = _disc(64, 64, 32, 32, 22)                 # hardening kept a 2 px halo as well
    st = MD.preservation_stats(keep, mask, band=3)
    assert st["fg_removed_frac"] == 0.0
    assert st["fg_removed_core_frac"] == 0.0
    assert 0.0 < st["bg_kept_frac"] < 0.25
    assert st["fg_px"] == int(mask.sum()) and st["keep_px"] == int(keep.sum())


def test_preservation_stats_sees_a_straight_cut_that_solidity_cannot():
    mask = _disc(64, 64, 32, 32, 20)
    keep = mask & (np.arange(64)[None, :] >= 32)      # the left half of the tablet was replaced
    st = MD.preservation_stats(keep, mask, band=3)
    assert abs(st["fg_removed_frac"] - 0.5) < 0.05
    assert abs(st["fg_removed_core_frac"] - 0.5) < 0.05
    assert st["bg_kept_frac"] == 0.0


def test_preservation_stats_ignores_a_boundary_disagreement_in_the_core_figure():
    mask = _disc(64, 64, 32, 32, 20)
    keep = _disc(64, 64, 32, 32, 18)                  # hardening ate a 2 px rim all round
    st = MD.preservation_stats(keep, mask, band=3)
    assert st["fg_removed_frac"] > 0.1
    assert st["fg_removed_core_frac"] == 0.0


def test_pick_mask_prefers_the_best_score_within_the_area_band():
    h = w = 32
    masks = np.stack([np.ones((h, w), bool), _disc(h, w, 16, 16, 8), _disc(h, w, 16, 16, 2)])
    scores = np.array([0.99, 0.80, 0.95])
    assert RM.pick_mask(masks, scores, lo=0.05, hi=0.90) == 1
    assert RM.pick_mask(masks, np.array([0.1, 0.1, 0.1]), lo=0.9, hi=0.95) is None


def test_pick_mask_rejects_a_mask_that_touches_the_field_border():
    """The synthetic field around a reference tablet reaches the crop border; the tablet never
    does (every box carries a 15 percent margin). A mask touching the border is the field."""
    h = w = 40
    valid = np.zeros((h, w), bool); valid[4:36, 2:38] = True          # non letterbox region
    field = valid.copy()                                               # pill plus field
    pill = _disc(h, w, 20, 20, 9)
    masks = np.stack([field, pill]); scores = np.array([1.0, 0.7])
    assert RM.pick_mask(masks, scores, lo=0.05, hi=0.90, valid=valid) == 1
    assert RM.touches_border(field, valid) and not RM.touches_border(pill, valid)


def test_largest_component_fills_holes_and_drops_specks():
    m = _disc(40, 40, 20, 20, 10)
    m[20, 20] = False                                 # a hole
    m[2, 2] = True                                    # a speck
    out = RM.largest_component(m)
    assert out[20, 20] and not out[2, 2]
    assert out.sum() == _disc(40, 40, 20, 20, 10).sum()


def test_select_review_sample_is_deterministic_and_covers_the_tail():
    sol = {f"c{i}": 1.0 - i * 0.001 for i in range(100)}
    suspects = ["c3", "c50"]
    s1 = RM.select_review_sample(sol, suspects, n=20, n_lowest=5)
    s2 = RM.select_review_sample(sol, suspects, n=20, n_lowest=5)
    assert s1 == s2 and len(s1) == 20 and len(set(s1)) == 20
    assert set(suspects) <= set(s1)
    assert {"c99", "c98", "c97", "c96", "c95"} <= set(s1)


def test_classes_for_sheet_from_damage_ranking(tmp_path):
    import json
    dmg = {"per_class": {"a": {"fg_removed_core_frac": 0.5}, "b": {"fg_removed_core_frac": 0.0},
                         "c": {"fg_removed_core_frac": 0.2}, "d": {"solidity": 0.9}}}
    p = tmp_path / "d.json"; p.write_text(json.dumps(dmg))
    assert RM.classes_by_damage(str(p), worst=2) == ["a", "c"]
    assert RM.classes_by_damage(str(p), worst=10, above=0.01) == ["a", "c"]
