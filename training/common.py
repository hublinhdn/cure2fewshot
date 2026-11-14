"""common.py — recipe table, data frames, model and metric shared by train_baseline.py and
eval_baseline.py.

Ported from the training stack that produced expected/retrieval_baselines.csv: same architecture
(GeM pooling, 512 dimensional projection, batch normalised embedding, plain cosine head and
additive angular margin head), same losses, same sampler, same schedule, same per group settings.
Everything the 27 sweep backbones do not need (MPNCOV pooling, DINOv2 layer wise decay) is left
out. Differences from the original stack that could move a number are listed in TRAINING_RECIPE.md.

The pure Python parts (recipe_for, build_frames, retrieval_metrics) import nothing heavy, so they
are unit tested without torch; the model and data classes import torch lazily.
"""
import math
import os

import numpy as np
import pandas as pd

SWEEP_BACKBONES = [
    "convnext_small_timm", "eva02_large_patch14_448_timm", "convnext_large_timm",
    "convnextv2_base.fcmae_ft_in22k_in1k_384_timm", "tf_efficientnetv2_l.in21k_ft_in1k_timm",
    "convnext_base_timm", "eva02_base_patch14_448_timm", "efficientnet_b5_timm", "tresnet_l_timm",
    "swin_base_patch4_window12_384_timm", "maxvit_base_tf_384_timm", "densenet161_tv",
    "seresnext101_32x4d_timm", "resnet101_tv", "resnest101e_timm", "efficientnet_b1_timm",
    "resnet18_tv", "repvgg_a0_timm", "mobilenet_v3_large_tv", "efficientnet_b0_timm",
    "efficientformerv2_s1_timm", "squeezenet1_0_tv", "efficientformerv2_s0_timm",
    "mobilevit_s_timm", "mobilevit_xs_timm", "ghostnet_100_timm", "levit_128s_timm",
]

# Backbone groups, by clean name (the _tv / _timm suffix stripped). Membership decides batch,
# learning rates, margin, epochs, resolution and the head warm up, exactly as in the sweep.
SUPER_LARGE = ["vit_large_patch14_dinov2.lvd142m", "eva02_large_patch14_448", "convnext_large",
               "convnextv2_large", "swin_large_patch4_window12_384", "tf_efficientnetv2_l.in21k_ft_in1k",
               "internimage_l", "replknet_31b"]
LARGE = ["vit_base_patch14_dinov2.lvd142m", "eva02_base_patch14_448", "deit_base_patch16_224",
         "convnext_base", "convnextv2_base.fcmae_ft_in22k_in1k_384", "swin_base_patch4_window12_384",
         "efficientnet_b5", "resnest101e", "seresnext101_32x4d", "densenet161", "tresnet_l", "resnet101",
         "maxvit_base_tf_384"]
MEDIUM = ["vit_small_patch14_dinov2.lvd142m", "deit_small_patch16_224", "swin_small_patch4_window7_224",
          "swin_tiny_patch4_window7_224", "convnext_small", "efficientnet_b3", "resnet50",
          "maxvit_small_tf_224", "maxvit_tiny_tf_224"]
SMALL = ["resnet18", "resnet34", "efficientnet_b0", "efficientnet_b1", "mobilenetv3_large_100",
         "mobilenetv2_100", "mobilenet_v3_large", "mobilenet_v2", "ghostnet_100", "shufflenet_v2_x1_0",
         "squeezenet1_0", "repvgg_a0"]
VIT = ["vit_base_patch14_dinov2.lvd142m", "vit_large_patch14_dinov2.lvd142m",
       "vit_small_patch14_dinov2.lvd142m", "eva02_base_patch14_448", "eva02_large_patch14_448",
       "deit_base_patch16_224", "deit_small_patch16_224"]
SWIN = ["swin_base_patch4_window12_384", "swin_large_patch4_window12_384",
        "swin_small_patch4_window7_224", "swin_tiny_patch4_window7_224"]
LIGHT_VIT = ["efficientformerv2_s0", "efficientformerv2_s1", "mobilevit_s", "mobilevit_xs",
             "mobilevit_xxs", "fasternet_t0", "tiny_vit_11m_224", "levit_128s"]
GHOSTNET_LIKE = ["ghostnet_100"]
DW_MODERATE = ["mobilenet_v2", "mobilenet_v3_large", "mobilenetv2_100", "mobilenetv3_large_100",
               "shufflenet_v2_x1_0"]
FRAGILE_MEDIUM = ["swin_tiny_patch4_window7_224", "swin_small_patch4_window7_224",
                  "vit_small_patch14_dinov2.lvd142m"]
MEDIUM_EXPERIMENTAL = ["maxvit_tiny_tf_224", "vit_base_patch14_dinov2.lvd142m"]
DINOV2 = ["vit_small_patch14_dinov2.lvd142m", "vit_base_patch14_dinov2.lvd142m",
          "vit_large_patch14_dinov2.lvd142m"]
EFFICIENTNET = ["tf_efficientnetv2_l.in21k_ft_in1k", "efficientnet_b5", "efficientnet_b3",
                "efficientnet_b0", "efficientnet_b1"]
HIGH_WD = ["replknet_31b", "eva02_base_patch14_448", "eva02_large_patch14_448"]

IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
EVAL_SIZE = 384          # every backbone is scored at 384, see TRAINING_RECIPE.md "Resolution"


def clean_name(backbone):
    return backbone.lower().replace("_tv", "").replace("_timm", "")


def recipe_for(backbone, epochs=60):
    """Every setting the sweep derived from the backbone's group. Raises KeyError for a backbone
    that belongs to no group, as the sweep did."""
    name = clean_name(backbone)
    is_swin, is_vit, is_light = name in SWIN, name in VIT, name in LIGHT_VIT
    is_small, is_massive, is_large, is_medium = name in SMALL, name in SUPER_LARGE, name in LARGE, name in MEDIUM
    is_pure_vit, is_fragile, is_dinov2, is_eff = (is_swin or is_vit), (is_small or is_light), name in DINOV2, name in EFFICIENTNET
    if is_massive:
        group, ncb, ns, accum, lr_b, lr_h = "super_large", 4, 2, (16 if (is_dinov2 or is_eff) else 20), 2e-5, 2e-4
    elif is_large:
        group, ncb, ns, accum, lr_b, lr_h = "large", 8, 2, 8, 3e-5, 3e-4
    elif is_medium:
        group, ncb, ns, accum, lr_b, lr_h = "medium", 8, 2, 8, 3e-5, 3e-4
    elif is_light:
        group, ncb, ns, accum, lr_b, lr_h = "lightweight_vit", 16, 2, 4, 3e-5, 2e-4
    elif is_small:
        group, ncb, ns, accum, lr_b, lr_h = "small", 16, 2, 4, 5e-5, 5e-4
    else:
        raise KeyError(f"{backbone}: not in any backbone group")
    if name in GHOSTNET_LIKE:
        lr_b, lr_h = 2e-5, 2e-4
    elif name in DW_MODERATE:
        lr_b, lr_h = 4e-5, 4e-4
    if is_light:
        margin = 0.25
    elif is_small:
        margin = 0.20
    elif name in FRAGILE_MEDIUM:
        margin = 0.25
    elif name in MEDIUM_EXPERIMENTAL:
        margin = 0.30
    else:
        margin = 0.35
    w_csce = min(0.2, 0.10) if is_light else 0.2
    warmup, total = 5, epochs
    if is_light:
        warmup, total = 10, max(epochs, 80)
    elif is_small or name in FRAGILE_MEDIUM:
        total = max(epochs, 80)
    elif name in MEDIUM_EXPERIMENTAL:
        total = max(epochs, 70)
    if is_pure_vit:
        lr_b = lr_b * (0.1 if is_dinov2 else 0.3)
    if "_224" in name:
        img = 224
    elif "_384" in name:
        img = 384
    elif is_light or is_small:
        img = 224
    elif "patch14" in name:
        img = 392
    else:
        img = 384
    wd = 0.1 if (is_massive or is_large or name in HIGH_WD) else 5e-2
    if is_dinov2:
        wd = 0.05
    freeze = 0
    if ((is_fragile or is_massive) and not is_pure_vit) or is_dinov2:
        freeze = 10 if (is_dinov2 and (is_massive or is_large)) else (4 if (is_massive or is_dinov2) else 2)
    clip = 3.0 if (is_fragile or is_pure_vit) else 5.0
    drop_path = 0.1 if is_dinov2 else (0.3 if (is_massive or is_large) else 0.2)
    return {"backbone": backbone, "group": group, "n_classes_batch": ncb, "n_samples": ns,
            "accumulation_steps": accum, "lr_backbone": lr_b, "lr_head": lr_h, "weight_decay": wd,
            "arcface_margin": margin, "w_sce": 1.0, "w_csce": w_csce, "w_triplet": 1.0, "w_cont": 1.0,
            "total_epochs": total, "warmup_epochs": warmup, "img_size": img, "freeze_epochs": freeze,
            "clip_grad": clip, "drop_path": drop_path, "is_pure_vit": is_pure_vit, "is_dinov2": is_dinov2,
            "is_light_vit": is_light, "label_smoothing": 0.1, "triplet_margin": 0.3,
            "miner_margin": 0.2, "scale": 64.0, "embedding_size": 512}


# ----------------------------------------------------------------------------- data frames
def _hardened_for(crop_rel, hardened_dir):
    parts = str(crop_rel).replace("\\", "/").split("/")
    if hardened_dir and "cure_crops" in parts:
        k = parts.index("cure_crops")
        if k + 2 < len(parts):
            p = os.path.join(hardened_dir, f"{parts[k + 1]}__{parts[k + 2]}.png")
            if os.path.exists(p):
                return p
    return None


def _resolve(crop_rel, crops_root):
    s = str(crop_rel)
    return s if os.path.isabs(s) else os.path.join(crops_root, s)


def build_frames(data_dir, crops_root, hardened_dir=None, gallery=None, harden_train_refs=True,
                 train="train_clean.csv", queries="query_eval_fold0.csv"):
    """(df_train, df_eval, num_classes, labels). df_eval holds the gallery rows (is_ref 1) followed
    by the evaluation queries (is_ref 0). Reference images in the training frame are replaced by
    their hardened version when hardened_dir holds one, as in the sweep, so the model never sees
    the uniform synthetic background."""
    tr = pd.read_csv(os.path.join(data_dir, train))
    q = pd.read_csv(os.path.join(data_dir, queries))
    g = pd.read_csv(gallery or os.path.join(data_dir, "gallery_fewshot_K1_hardened_v2.csv"))
    labels = sorted(set(tr["label"]) | set(q["label"]) | set(g["label"]))
    lut = {c: i for i, c in enumerate(labels)}
    rows = []
    for _, r in tr.iterrows():
        is_ref = int(r.get("domain", "consumer") == "reference")
        p = _hardened_for(r["crop_rel"], hardened_dir) if (is_ref and harden_train_refs) else None
        rows.append({"image_path": p or _resolve(r["crop_rel"], crops_root), "label_idx": lut[r["label"]],
                     "is_ref": is_ref, "split": "train"})
    for _, r in g.iterrows():
        rows.append({"image_path": _resolve(r["crop_rel"], crops_root), "label_idx": lut[r["label"]],
                     "is_ref": 1, "split": "eval"})
    for _, r in q.iterrows():
        rows.append({"image_path": _resolve(r["crop_rel"], crops_root), "label_idx": lut[r["label"]],
                     "is_ref": 0, "split": "eval"})
    df = pd.DataFrame(rows)
    return (df[df["split"] == "train"].drop(columns="split").reset_index(drop=True),
            df[df["split"] == "eval"].drop(columns="split").reset_index(drop=True), len(labels), labels)


# ----------------------------------------------------------------------------- metric
def retrieval_metrics(qf, ql, gf, gl):
    """Cosine retrieval of each query against the gallery, one score per class (the maximum over
    that class's gallery entries), classes ranked. With one reference per class the average
    precision of a query is the reciprocal rank of its class, so mAP is the mean reciprocal rank.
    A query whose class has no gallery entry cannot be retrieved: it is a miss at every rank and
    contributes 0 to mAP; the count of such queries is returned."""
    qf = np.asarray(qf, np.float32); gf = np.asarray(gf, np.float32)
    ql = np.asarray(ql); gl = np.asarray(gl)
    classes = np.unique(gl)
    S = qf @ gf.T
    cls_scores = np.stack([S[:, gl == c].max(axis=1) for c in classes], axis=1)
    order = np.argsort(-cls_scores, axis=1, kind="stable")
    ranks = np.full(len(qf), np.inf)
    present = np.isin(ql, classes)
    for i in np.flatnonzero(present):
        ranks[i] = int(np.where(classes[order[i]] == ql[i])[0][0]) + 1
    rr = np.where(np.isfinite(ranks), 1.0 / ranks, 0.0)
    return {"mAP": float(rr.mean()), "Rank-1": float((ranks == 1).mean()),
            "CMC@5": float((ranks <= 5).mean()), "n_query": int(len(qf)),
            "n_gallery_classes": int(len(classes)),
            "n_query_without_gallery": int((~present).sum())}


# ----------------------------------------------------------------------------- torch parts
def seed_everything(seed):
    import random
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_transforms(img_size):
    from torchvision import transforms as T
    resize_scale = int(img_size * 1.15)
    norm = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    train = T.Compose([T.Resize((resize_scale, resize_scale)), T.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
                       T.RandomHorizontalFlip(), T.RandomVerticalFlip(), T.RandomRotation(180),
                       T.ColorJitter(0.2, 0.1), T.ToTensor(), norm, T.RandomErasing(p=0.3, scale=(0.02, 0.1))])
    train_ref = T.Compose([T.Resize((img_size, img_size)), T.RandomHorizontalFlip(), T.RandomRotation(15),
                           T.ColorJitter(0.2, 0.1), T.ToTensor(), norm])
    evaluate = T.Compose([T.Resize((EVAL_SIZE, EVAL_SIZE)), T.ToTensor(), norm])
    return train, train_ref, evaluate


class PillDataset:
    """Rows of a frame from build_frames; reference rows get transform_ref."""

    def __init__(self, df, transform=None, transform_ref=None):
        self.df, self.transform = df, transform
        self.transform_ref = transform_ref if transform_ref is not None else transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        from PIL import Image
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        tf = self.transform_ref if int(row["is_ref"]) == 1 else self.transform
        if tf is not None:
            image = tf(image)
        return image, int(row["label_idx"]), int(row["label_idx"]), int(row["is_ref"])


class BalancedBatchSampler:
    """n_classes classes per batch, n_samples images each, drawn with numpy's global RNG as in the
    sweep (seed_everything seeds it)."""

    def __init__(self, labels, n_classes, n_samples):
        self.labels = np.asarray(labels)
        self.n_classes, self.n_samples = n_classes, n_samples
        self.label_to_indices = {l: np.where(self.labels == l)[0] for l in np.unique(self.labels)}
        self.labels_list = list(self.label_to_indices)

    def __iter__(self):
        for _ in range(len(self)):
            batch = []
            for cls in np.random.choice(self.labels_list, self.n_classes, replace=False):
                batch.extend(np.random.choice(self.label_to_indices[cls], self.n_samples, replace=True))
            yield batch

    def __len__(self):
        return len(self.labels) // (self.n_classes * self.n_samples)


def build_model(backbone, num_classes, recipe=None):
    """The retrieval model of the sweep. Returns an nn.Module whose forward(x) gives the L2
    normalised 512 dimensional embedding, and forward(x, labels) the two logit heads as well."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    recipe = recipe or recipe_for(backbone)
    name = clean_name(backbone)
    force_tv = backbone.lower().endswith("_tv")
    is_pure_vit, is_light = recipe["is_pure_vit"], recipe["is_light_vit"]

    class GeM(nn.Module):
        def __init__(self, p=3.0, eps=1e-6):
            super().__init__()
            self.p = nn.Parameter(torch.ones(1) * p); self.eps = eps

        def forward(self, x):
            x = x.clamp(min=self.eps).pow(self.p)
            return F.adaptive_avg_pool2d(x, (1, 1)).pow(1.0 / self.p)

    def load_backbone():
        if force_tv:
            from torchvision import models
            base = getattr(models, name)(weights="DEFAULT")
            if "resnet" in name:
                if name in ("resnet18", "resnet34"):
                    base.layer4[0].conv1.stride = (1, 1)
                else:
                    base.layer4[0].conv2.stride = (1, 1)
                base.layer4[0].downsample[0].stride = (1, 1)
                return nn.Sequential(*list(base.children())[:-2])
            if hasattr(base, "features"):
                return base.features
            return nn.Sequential(*list(base.children())[:-2])
        import timm
        kw = {"pretrained": True, "num_classes": 0, "drop_path_rate": recipe["drop_path"]}
        if any(x in name for x in ("ghostnet", "mobilenet", "shufflenet", "squeezenet", "repvgg")):
            kw.pop("drop_path_rate")
        if name in SWIN:
            kw["global_pool"], kw["dynamic_img_size"] = "avg", True
        elif "maxvit" in name:
            kw["global_pool"] = ""
        elif name in VIT:
            kw["global_pool"] = "token" if recipe["is_dinov2"] else "avg"
            kw["dynamic_img_size"] = True
        elif is_light:
            kw["global_pool"] = "avg"
        else:
            kw["global_pool"] = ""
        m = timm.create_model(name, **kw)
        if name in SUPER_LARGE or name in LARGE:
            try:
                m.set_grad_checkpointing(True)
            except Exception:
                pass
        return m

    class PillRetrievalModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.s, m = recipe["scale"], recipe["arcface_margin"]
            self.features = load_backbone()
            if "_224" in name or is_light or name in SMALL:
                self.safe_size = 224
            elif "_384" in name:
                self.safe_size = 384
            elif "patch14" in name:
                self.safe_size = 392
            else:
                self.safe_size = 384
            self.features.eval()
            with torch.no_grad():
                out = self.features(torch.zeros(1, 3, self.safe_size, self.safe_size))
                out = out[0] if isinstance(out, tuple) else out
                in_ch = out.shape[1]
            self.features.train()
            emb = recipe["embedding_size"]
            self.pool = GeM(p=3.0)
            if is_pure_vit or is_light:
                if in_ch < emb:
                    self.fc_projection = nn.Sequential(nn.Linear(in_ch, emb, bias=False), nn.LayerNorm(emb), nn.GELU(),
                                                       nn.Dropout(p=0.2), nn.Linear(emb, emb, bias=False))
                else:
                    self.fc_projection = nn.Sequential(nn.Linear(in_ch, in_ch, bias=False), nn.LayerNorm(in_ch), nn.GELU(),
                                                       nn.Dropout(p=0.2), nn.Linear(in_ch, emb, bias=False))
            elif in_ch >= 1024:
                self.fc_projection = nn.Sequential(nn.Linear(in_ch, 1024, bias=False), nn.BatchNorm1d(1024), nn.GELU(),
                                                   nn.Dropout(p=0.2), nn.Linear(1024, emb, bias=False))
            else:
                self.fc_projection = nn.Linear(in_ch, emb, bias=False)
            self.bn_head = nn.BatchNorm1d(emb)
            self.fc_ce = nn.Linear(emb, num_classes, bias=False)
            self.weight = nn.Parameter(torch.FloatTensor(num_classes, emb))
            nn.init.xavier_uniform_(self.weight)
            self.cos_m, self.sin_m = math.cos(m), math.sin(m)
            self.th, self.mm = math.cos(math.pi - m), math.sin(math.pi - m) * m

        def forward(self, x, labels=None):
            if x.shape[-1] != self.safe_size:
                x = F.interpolate(x, size=(self.safe_size, self.safe_size), mode="bilinear", align_corners=False)
            feat = self.features(x)
            pooled = feat if feat.dim() == 2 else self.pool(feat).view(feat.size(0), -1)
            embedding = self.bn_head(self.fc_projection(pooled))
            norm_embedding = F.normalize(embedding, p=2, dim=1)
            if labels is None:
                return norm_embedding
            logits_sce = F.linear(norm_embedding, F.normalize(self.fc_ce.weight, p=2, dim=1)) * self.s
            cosine = F.linear(norm_embedding, F.normalize(self.weight)).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
            sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
            phi = cosine * self.cos_m - sine * self.sin_m
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
            one_hot = torch.zeros_like(cosine)
            one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
            logits_csce = ((one_hot * phi) + ((1.0 - one_hot) * cosine)) * self.s
            return logits_sce, logits_csce, norm_embedding

    return PillRetrievalModel()


def extract_features(model, df, device, batch_size=32, num_workers=4):
    """L2 normalised embeddings of every row of df at EVAL_SIZE, fp16 autocast on CUDA as in the
    sweep's evaluation. Returns (features, labels, is_ref) as numpy arrays."""
    import torch
    from torch.utils.data import DataLoader
    _, _, tf = make_transforms(EVAL_SIZE)
    loader = DataLoader(PillDataset(df, transform=tf), batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=(device.type == "cuda"))
    feats, labels, isref = [], [], []
    model.eval()
    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16,
                                                enabled=(device.type == "cuda")):
        for imgs, _sub, lbl, ir in loader:
            emb = model(imgs.to(device, non_blocking=True))
            feats.append(torch.nn.functional.normalize(emb.float(), dim=1).cpu())
            labels.append(torch.as_tensor(lbl)); isref.append(torch.as_tensor(ir))
    return (torch.cat(feats).numpy(), torch.cat(labels).numpy(), torch.cat(isref).numpy())


def evaluate(model, df_eval, device, batch_size=32, num_workers=4):
    f, l, r = extract_features(model, df_eval, device, batch_size, num_workers)
    q, g = r == 0, r == 1
    return retrieval_metrics(f[q], l[q], f[g], l[g])


def versions():
    out = {}
    for m in ("torch", "torchvision", "timm", "pytorch_metric_learning", "numpy", "PIL"):
        try:
            out[m] = __import__(m).__version__
        except Exception:
            out[m] = None
    try:
        import torch
        out["cuda"] = torch.version.cuda
        out["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        pass
    return out
