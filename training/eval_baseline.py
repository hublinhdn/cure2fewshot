#!/usr/bin/env python3
"""eval_baseline.py — score a trained checkpoint on the benchmark (queries of the held out fold
against the one reference per class gallery), with the same scoring as the sweep.

  python training/eval_baseline.py --backbone convnext_small_timm --checkpoint best_convnext_small_timm.pth \\
      --data-dir outputs --gallery outputs_v2/gallery_fewshot_K1_hardened_v2.csv --crops-root $CURE_CROPS_ROOT \\
      --out eval_convnext_small.json

The checkpoint is a state_dict of the model that train_baseline.py builds (the deposited sweep
checkpoints have the same layout). --gallery may point at the gallery without the five suspect
references; queries whose class then has no gallery entry are reported as misses and counted.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--crops-root", required=True)
    ap.add_argument("--gallery", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    import torch
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    _, df_eval, num_classes, _ = C.build_frames(args.data_dir, args.crops_root, None, args.gallery)
    model = C.build_model(args.backbone, num_classes)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise SystemExit(f"[eval] state_dict does not match the model: missing {missing[:5]} unexpected {unexpected[:5]}")
    model.to(device)
    met = C.evaluate(model, df_eval, device, args.batch_size, args.num_workers)
    out = {"backbone": args.backbone, "checkpoint": os.path.abspath(args.checkpoint), "num_classes": num_classes,
           "gallery": os.path.abspath(args.gallery) if args.gallery else None, **met, "versions": C.versions()}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[eval] {args.backbone}: mAP {met['mAP']:.4f} Rank-1 {met['Rank-1']:.4f} CMC@5 {met['CMC@5']:.4f} "
          f"(queries {met['n_query']}, without gallery {met['n_query_without_gallery']}) -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
