"""make_duplicate_resolution.py: the frozen decision file is derived from the source duplicate scan."""
import json

import pandas as pd

import make_duplicate_resolution as MR


def test_resolution_keeps_the_copy_under_the_class_that_sorts_first(tmp_path):
    scan = {"groups": [
        {"files": ["raw/81/bottom/Customer/81_bottom_3601.jpg", "raw/81/top/Customer/81_top_3578.jpg"],
         "classes": ["81__bottom", "81__top"], "kinds": ["customer", "customer"],
         "crops": ["crops/81/bottom/customer/81_bottom_3601.png", "crops/81/top/customer/81_top_3578.png"]},
        {"files": ["raw/9/top/Customer/9_top_1.jpg", "raw/9/top/Customer/9_top_2.jpg"],
         "classes": ["9__top", "9__top"], "kinds": ["customer", "customer"],
         "crops": ["crops/9/top/customer/9_top_1.png", "crops/9/top/customer/9_top_2.png"]},
    ]}
    p = tmp_path / "source_duplicates.json"; p.write_text(json.dumps(scan))
    out = tmp_path / "duplicate_resolution.csv"
    rows = MR.build(str(p), str(out))
    d = pd.read_csv(out)
    assert len(d) == 2 and rows == 2
    a = d.iloc[0]
    assert a["keep_crop_rel"] == "crops/81/bottom/customer/81_bottom_3601.png"
    assert a["drop_crop_rel"] == "crops/81/top/customer/81_top_3578.png"
    assert (a["keep_class"], a["drop_class"]) == ("81__bottom", "81__top")
    assert a["rule"] == MR.RULE
    b = d.iloc[1]                      # same class twice: keep the first crop name, drop the second
    assert b["keep_crop_rel"] == "crops/9/top/customer/9_top_1.png"
    assert list(d.columns) == ["group", "keep_crop_rel", "drop_crop_rel", "keep_class", "drop_class", "rule", "note"]
