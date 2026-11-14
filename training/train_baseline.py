#!/usr/bin/env python3
"""train_baseline.py — train one backbone on the benchmark with the recipe of the 27 backbone sweep.

  python training/train_baseline.py --backbone resnet18_tv --seed 42 \\
      --data-dir outputs --gallery outputs_v2/gallery_fewshot_K1_hardened_v2.csv \\
      --hardened-dir outputs_v2/hardened_refs_v2 --crops-root $CURE_CROPS_ROOT --out-dir runs

Reads train_clean.csv and query_eval_fold0.csv from --data-dir and the gallery from --gallery
(default: gallery_fewshot_K1_hardened_v2.csv in --data-dir). Training references are swapped
for their hardened image when --hardened-dir holds one, as in the sweep. Scores the held out
queries against the gallery every --eval-every epochs and in the last five, keeps the checkpoint
with the best score (the sweep's selection rule; note that this selects on the evaluation
queries themselves, see TRAINING_RECIPE.md) and the last epoch's checkpoint, and writes
<out-dir>/<backbone>_seed<seed>.json with the whole history, both scores, the recipe and the
software versions.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--crops-root", required=True)
    ap.add_argument("--gallery", default=None)
    ap.add_argument("--hardened-dir", default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--epochs", type=int, default=None, help="override the recipe's epoch count")
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-harden-train-refs", action="store_true")
    args = ap.parse_args()

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from pytorch_metric_learning import losses, miners

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    C.seed_everything(args.seed)
    r = C.recipe_for(args.backbone)
    if args.epochs is not None:
        r["total_epochs"] = args.epochs
        r["warmup_epochs"] = min(r["warmup_epochs"], max(args.epochs - 1, 1))
    df_train, df_eval, num_classes, labels = C.build_frames(
        args.data_dir, args.crops_root, args.hardened_dir, args.gallery, not args.no_harden_train_refs)
    print(f"[train] {args.backbone} seed {args.seed} on {device}: train {len(df_train)} "
          f"(ref {int(df_train['is_ref'].sum())}), eval {len(df_eval)}, classes {num_classes}", flush=True)
    print("[train] recipe: " + ", ".join(f"{k}={v}" for k, v in r.items()
                                           if k not in ("backbone", "is_pure_vit", "is_dinov2", "is_light_vit")), flush=True)

    train_tf, ref_tf, _ = C.make_transforms(r["img_size"])
    loader = DataLoader(C.PillDataset(df_train, transform=train_tf, transform_ref=ref_tf),
                        batch_sampler=C.BalancedBatchSampler(df_train["label_idx"].values, r["n_classes_batch"], r["n_samples"]),
                        num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    model = C.build_model(args.backbone, num_classes, r).to(device)
    ce = nn.CrossEntropyLoss(label_smoothing=r["label_smoothing"])
    triplet = losses.TripletMarginLoss(margin=r["triplet_margin"])
    contrastive = losses.ContrastiveLoss(pos_margin=0, neg_margin=1)
    miner = miners.TripletMarginMiner(margin=r["miner_margin"], type_of_triplets="semihard")
    head_params = [p for n, p in model.named_parameters() if "features" not in n]
    backbone_params = [p for n, p in model.named_parameters() if "features" in n]
    opt = torch.optim.AdamW([{"params": backbone_params, "lr": r["lr_backbone"]},
                             {"params": head_params, "lr": r["lr_head"]}], weight_decay=r["weight_decay"])
    warm = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.01, total_iters=r["warmup_epochs"])
    cos = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(r["total_epochs"] - r["warmup_epochs"], 1), eta_min=1e-6)
    sched = torch.optim.lr_scheduler.SequentialLR(opt, schedulers=[warm, cos], milestones=[r["warmup_epochs"]])
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    os.makedirs(args.out_dir, exist_ok=True)
    stem = f"{args.backbone}_seed{args.seed}"
    best_p, last_p = os.path.join(args.out_dir, f"best_{stem}.pth"), os.path.join(args.out_dir, f"last_{stem}.pth")
    history, selected = [], None
    t0 = time.time()
    E = r["total_epochs"]
    for epoch in range(1, E + 1):
        if r["freeze_epochs"]:
            if epoch <= r["freeze_epochs"]:
                for p in model.features.parameters():
                    p.requires_grad = False
            elif epoch == r["freeze_epochs"] + 1:
                for p in model.features.parameters():
                    p.requires_grad = True
        model.train()
        total, n_steps = 0.0, 0
        opt.zero_grad()
        for i, (imgs, sub, _lbl, _ir) in enumerate(loader):
            imgs, sub = imgs.to(device), sub.to(device).long()
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                l_sce, l_csce, emb = model(imgs, labels=sub)
                loss_sce, loss_csce = ce(l_sce, sub), ce(l_csce, sub)
                hard = miner(emb, sub)
                loss_tri, loss_con = triplet(emb, sub, hard), contrastive(emb, sub)
                loss = (r["w_sce"] * loss_sce + r["w_csce"] * loss_csce + r["w_triplet"] * loss_tri
                        + r["w_cont"] * loss_con) / r["accumulation_steps"]
            scaler.scale(loss).backward()
            if (i + 1) % r["accumulation_steps"] == 0 or (i + 1) == len(loader):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=r["clip_grad"])
                scaler.step(opt); scaler.update(); opt.zero_grad()
            total += loss.item() * r["accumulation_steps"]; n_steps += 1
        sched.step()
        avg = total / max(n_steps, 1)
        if epoch % args.eval_every == 0 or epoch > E - 5:
            met = C.evaluate(model, df_eval, device, num_workers=args.num_workers)
            rec = {"epoch": epoch, "loss": avg, **met, "seconds": round(time.time() - t0, 1)}
            history.append(rec)
            print(f"[train] epoch {epoch}/{E} loss {avg:.4f} mAP {met['mAP']:.4f} R1 {met['Rank-1']:.4f} "
                  f"CMC5 {met['CMC@5']:.4f} ({rec['seconds']}s)", flush=True)
            if selected is None or met["mAP"] > selected["mAP"]:
                selected = rec
                torch.save(model.state_dict(), best_p)
        else:
            print(f"[train] epoch {epoch}/{E} loss {avg:.4f} ({round(time.time() - t0)}s)", flush=True)
    torch.save(model.state_dict(), last_p)
    last = history[-1] if history and history[-1]["epoch"] == E else {"epoch": E, **C.evaluate(model, df_eval, device, num_workers=args.num_workers)}
    out = {"backbone": args.backbone, "seed": args.seed, "num_classes": num_classes,
           "n_train": int(len(df_train)), "n_train_ref": int(df_train["is_ref"].sum()),
           "n_query": int((df_eval["is_ref"] == 0).sum()), "n_gallery": int((df_eval["is_ref"] == 1).sum()),
           "data_dir": os.path.abspath(args.data_dir), "gallery": os.path.abspath(args.gallery) if args.gallery else None,
           "hardened_dir": os.path.abspath(args.hardened_dir) if args.hardened_dir else None,
           "recipe": r, "history": history, "selected": selected, "last": last,
           "selection_rule": "best mAP on the evaluation queries among the epochs scored (the sweep's rule)",
           "seconds": round(time.time() - t0, 1), "device": str(device), "versions": C.versions(),
           "checkpoints": {"best": best_p, "last": last_p}}
    with open(os.path.join(args.out_dir, f"{stem}.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"[train] done: selected epoch {selected['epoch']} mAP {selected['mAP']:.4f} | last epoch mAP "
          f"{last['mAP']:.4f} | {out['seconds']}s -> {stem}.json", flush=True)


if __name__ == "__main__":
    main()
