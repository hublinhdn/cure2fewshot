"""harden_refs_v2.py: the suspect references are flagged and an alternative gallery without them is written."""
import pandas as pd

import harden_refs_v2 as H


def test_split_gallery_by_suspect_uses_the_hardened_file_name():
    gallery = pd.DataFrame({
        "crop_rel": ["/x/hardened_refs_v2/52__top.png", "/x/hardened_refs_v2/7__bottom.png",
                     "/x/hardened_refs_v2/20__top.png"],
        "label": ["52__top", "7__bottom", "20__top"], "fold": [0, 1, 2]})
    per = {"52__top": {"suspect": True, "repair": "ellipse", "solidity": 0.9, "frac": 0.3},
           "7__bottom": {"suspect": False, "repair": "none", "solidity": 0.99, "frac": 0.4},
           "20__top": {"suspect": False, "repair": "convex_hull", "solidity": 0.98, "frac": 0.5}}
    kept, suspect = H.split_gallery_by_suspect(gallery, per)
    assert list(kept["label"]) == ["7__bottom", "20__top"]
    assert list(suspect["class_id"]) == ["52__top"]
    assert list(suspect.columns) == ["class_id", "fold", "repair", "solidity", "frac"]


def test_reviewed_mask_replaces_the_rule_mask_only_where_the_review_says_so(tmp_path):
    import numpy as np
    from PIL import Image
    mask_dir = tmp_path / "reference_masks"; mask_dir.mkdir()
    sam = np.zeros((20, 20), bool); sam[5:15, 5:15] = True
    Image.fromarray((sam * 255).astype(np.uint8)).save(mask_dir / "4__bottom.png")
    review = pd.DataFrame({"class_id": ["4__bottom", "51__bottom"], "decision": ["use_sam", "keep_rule"],
                           "reason": ["rule mask cut the tablet in half", "SAM merged the field"]})
    review.to_csv(tmp_path / "reference_mask_review.csv", index=False)
    rv = H.load_review(str(tmp_path / "reference_mask_review.csv"), str(mask_dir))
    rule = np.zeros((20, 20), bool); rule[5:10, 5:15] = True          # the half tablet the rule kept
    m, info = H.apply_review("4__bottom", rule, {"repair": "none", "suspect": False, "solidity": 1.0}, rv)
    assert m.sum() > rule.sum() and m[12, 10] and info["repair"] == "independent_mask"
    assert info["review"] == "use_sam" and info["suspect"] is False
    m2, info2 = H.apply_review("51__bottom", rule, {"repair": "none", "suspect": False}, rv)
    assert np.array_equal(m2, rule) and info2["repair"] == "none" and info2["review"] == "keep_rule"
    m3, info3 = H.apply_review("7__top", rule, {"repair": "none", "suspect": False}, rv)
    assert np.array_equal(m3, rule) and "review" not in info3


def test_load_review_refuses_a_use_sam_row_without_a_mask(tmp_path):
    import pytest
    (tmp_path / "masks").mkdir()
    pd.DataFrame({"class_id": ["9__top"], "decision": ["use_sam"], "reason": [""]}).to_csv(
        tmp_path / "r.csv", index=False)
    with pytest.raises(SystemExit):
        H.load_review(str(tmp_path / "r.csv"), str(tmp_path / "masks"))
