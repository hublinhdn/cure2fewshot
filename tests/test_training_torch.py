"""training/: model forward, a two step training smoke run and the evaluation script, on tiny
synthetic images. Skipped where torch is not installed (the protocol's own tests need no torch)."""
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pytorch_metric_learning")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "training"))
import common as T  # noqa: E402


def _tiny_dataset(tmp_path, n_classes=6, per_class=4):
    from PIL import Image
    crops = tmp_path / "crops"; hard = tmp_path / "hard"; data = tmp_path / "data"
    for d in (crops, hard, data):
        d.mkdir()
    rng = np.random.default_rng(0)
    tr, q, g = [], [], []
    for c in range(n_classes):
        cid = f"{c}__top"
        colour = rng.integers(0, 255, 3)
        for k in range(per_class):
            rel = f"data/processed/cure_crops/{c}/top/customer/{c}_top_{k}.png"
            os.makedirs(os.path.dirname(crops / rel), exist_ok=True)
            arr = np.clip(colour + rng.integers(-30, 30, (64, 64, 3)), 0, 255).astype(np.uint8)
            Image.fromarray(arr).save(crops / rel)
            (tr if c < 4 else q).append({"crop_rel": rel, "label": cid, "domain": "consumer", "fold": 1 if c < 4 else 0})
        rel = f"data/processed/cure_crops/{c}/top/ref/{c}_ref_top.png"
        os.makedirs(os.path.dirname(crops / rel), exist_ok=True)
        Image.fromarray(np.tile(colour, (64, 64, 1)).astype(np.uint8)).save(crops / rel)
        Image.fromarray(np.tile(colour, (64, 64, 1)).astype(np.uint8)).save(hard / f"{cid}.png")
        if c < 4:
            tr.append({"crop_rel": rel, "label": cid, "domain": "reference", "fold": 1})
        g.append({"crop_rel": str(hard / f"{cid}.png"), "label": cid, "fold": 0 if c >= 4 else 1})
    pd.DataFrame(tr).to_csv(data / "train_clean.csv", index=False)
    pd.DataFrame(q)[["crop_rel", "label", "fold"]].to_csv(data / "query_eval_fold0.csv", index=False)
    pd.DataFrame(g).to_csv(data / "gallery_fewshot_K1_hardened_v2.csv", index=False)
    return str(data), str(crops), str(hard)


def test_model_forward_shapes_resnet18():
    m = T.build_model("resnet18_tv", num_classes=6).eval()
    x = torch.zeros(2, 3, 96, 96)
    with torch.no_grad():
        e = m(x)
        a, b, e2 = m(x, labels=torch.tensor([0, 1]))
    assert e.shape == (2, 512) and torch.allclose(e.norm(dim=1), torch.ones(2), atol=1e-4)
    assert a.shape == (2, 6) and b.shape == (2, 6) and torch.allclose(e, e2)


def test_train_two_epochs_then_eval_script_round_trip(tmp_path):
    data, crops, hard = _tiny_dataset(tmp_path)
    out = tmp_path / "run"
    env = dict(os.environ, PYTHONPATH=os.path.join(ROOT, "training"))
    cmd = [sys.executable, os.path.join(ROOT, "training", "train_baseline.py"), "--backbone", "resnet18_tv",
           "--seed", "7", "--data-dir", data, "--crops-root", crops, "--hardened-dir", hard, "--out-dir", str(out),
           "--epochs", "2", "--eval-every", "1", "--num-workers", "0", "--device", "cpu"]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    res = json.load(open(out / "resnet18_tv_seed7.json"))
    assert res["seed"] == 7 and res["recipe"]["total_epochs"] == 2
    assert len(res["history"]) == 2 and {"epoch", "mAP", "Rank-1", "CMC@5"} <= set(res["history"][0])
    assert set(res["selected"]) >= {"epoch", "mAP"} and set(res["last"]) >= {"epoch", "mAP"}
    assert (out / "best_resnet18_tv_seed7.pth").exists() and (out / "last_resnet18_tv_seed7.pth").exists()
    cmd = [sys.executable, os.path.join(ROOT, "training", "eval_baseline.py"), "--backbone", "resnet18_tv",
           "--checkpoint", str(out / "best_resnet18_tv_seed7.pth"), "--data-dir", data, "--crops-root", crops,
           "--out", str(out / "eval.json"), "--num-workers", "0", "--device", "cpu"]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    ev = json.load(open(out / "eval.json"))
    assert ev["n_query"] == 8 and ev["n_gallery_classes"] == 6
    assert abs(ev["mAP"] - res["selected"]["mAP"]) < 1e-6
