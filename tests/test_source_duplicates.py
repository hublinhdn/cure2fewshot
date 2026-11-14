"""make_crops.py `duplicates`: byte identical source photographs filed under more than one entry."""
import json
import os

import make_crops as MC


def _write(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)


def test_hash_groups_returns_only_repeated_contents(tmp_path):
    a = tmp_path / "a.jpg"; b = tmp_path / "b.jpg"; c = tmp_path / "c.jpg"
    _write(str(a), b"same bytes"); _write(str(b), b"same bytes"); _write(str(c), b"other")
    groups = MC.hash_groups([("ka", str(a)), ("kb", str(b)), ("kc", str(c))])
    assert groups == [["ka", "kb"]]


def test_stage_duplicates_writes_groups_with_classes(tmp_path):
    raw = tmp_path / "raw"
    rows = []
    for name, pill, side, data in [("81_top_1.jpg", "81", "top", b"X" * 100),
                                   ("81_bottom_2.jpg", "81", "bottom", b"X" * 100),
                                   ("5_top_3.jpg", "5", "top", b"Y" * 100)]:
        rel = f"data/raw/CURE_dataset/Pill_Images/{pill}/{side}/Customer/{name}"
        _write(str(raw / rel), data)
        rows.append({"src_rel": rel, "crop_rel": f"crops/{name}.png", "pillId": pill,
                     "side": side, "kind": "customer"})
    out = tmp_path / "source_duplicates.json"
    summary = MC.stage_duplicates(rows, str(raw), str(out))
    got = json.load(open(out))
    assert got == summary
    assert got["n_files_hashed"] == 3
    assert got["n_groups"] == 1
    assert got["n_files_in_groups"] == 2
    assert got["n_groups_cross_class"] == 1
    g = got["groups"][0]
    assert g["classes"] == ["81__bottom", "81__top"]
    assert g["kinds"] == ["customer", "customer"]
    assert [os.path.basename(f) for f in g["files"]] == ["81_bottom_2.jpg", "81_top_1.jpg"]
