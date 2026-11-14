"""review_sheet.py: a contact sheet of image pairs for a manual check, from a pairs CSV."""
import os

import numpy as np
import pandas as pd
from PIL import Image

import review_sheet as RS


def test_sheet_has_one_row_per_pair_and_three_panels(tmp_path):
    crops = tmp_path / "crops"
    crops.mkdir()
    for name, col in (("a.png", 40), ("b.png", 200), ("c.png", 120)):
        Image.fromarray(np.full((64, 64, 3), col, np.uint8)).save(crops / name)
    pairs = pd.DataFrame({"crop_rel": ["a.png", "b.png"], "other_crop_rel": ["b.png", "c.png"],
                          "label": ["1__top", "2__top"], "other_label": ["3__top", "2__top"],
                          "cos_max": [0.97, 0.93], "kind": ["eval_query_vs_training_image", "query_vs_own_reference"]})
    p = tmp_path / "pairs.csv"; pairs.to_csv(p, index=False)
    out = tmp_path / "sheet.png"
    n = RS.build(str(p), str(crops), str(out), tile=48)
    assert n == 2 and out.exists()
    im = Image.open(out)
    assert im.size == (2 * 48 + RS.GAP, 2 * (48 + RS.HEADER))
