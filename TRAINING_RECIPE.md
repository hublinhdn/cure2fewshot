# Training recipe behind `expected/retrieval_baselines.csv`

The baseline scores come from a training sweep that is **not part of this protocol**. The
protocol's scripts stop at the measurements the rubric consumes, and no rubric score depends on
the sweep. The recipe is recorded here so the numbers can be interpreted and, with a training
stack of the reader's own, reproduced.

## What was trained

27 backbones from `timm`, spanning mobile scale networks to large transformers, each taken with
its pretrained weights and trained on the converted benchmark with one shared procedure and a
single seed, then evaluated on the held out fold. The data are the benchmark files described in
`REPRODUCE.md` section 6b: gallery `gallery_fewshot_K1_hardened_v2.csv` (386 hardened
references), evaluation queries `query_eval_fold0.csv` (1715 consumer photographs), training
images `train_clean.csv` (7016).

## Model, shared by every backbone

* Generalised mean pooling over the feature map, exponent fixed at 3.
* Projection to a 512 dimensional embedding, then batch normalisation of that embedding.
* Two cosine classification heads scaled by 64: one plain, one carrying an additive angular
  margin (ArcFace).

## Objective, shared

The sum of four terms: cross entropy with label smoothing 0.1 on the plain head (weight 1); the
same loss on the angular margin head (weight 0.2); a triplet margin loss with margin 0.3 mined for
semihard triplets (weight 1); a contrastive loss (weight 1).

## Optimisation and data, shared

AdamW with separate learning rates for the backbone and the heads; a linear warm up followed by
cosine annealing; a class balanced sampler with gradient accumulation chosen so that every
backbone sees an effective batch of 128. Consumer images are augmented with a random resized
crop, flips in both directions, rotation up to 180 degrees, colour jitter and random erasing;
reference images receive a milder transform. Seed 42 everywhere, non deterministic kernels
disabled. In every case the checkpoint with the best validation score is the one evaluated.

## Settings that vary by backbone size

| Group | Backbones | Physical batch | Backbone learning rate | Angular margin | Epochs | Backbone frozen at the start |
|---|---|---|---|---|---|---|
| Large | 15 | 16 | 3e-5 | 0.35 | 60 | first 4 epochs, for the two largest only |
| Compact | 12 | 32 | 3e-5 to 5e-5 | smaller | 80 | first 2 epochs, while the heads stabilise |

Learning rates of pure transformers are scaled down by a further factor of 0.3.

## Resolution

Evaluation is carried out at 384 pixels square for every backbone. Training resolution follows
each backbone's own convention: 384 for thirteen of the twenty seven, 224 for the twelve compact
ones, and 392 for the two whose patch size requires it. Holding the evaluation resolution fixed
keeps the scores directly comparable across backbones, at the cost of a shift between training
and evaluation resolution for fourteen of them. The shift is reported here rather than corrected
for (see `KNOWN_ISSUES.md` section 4).

## Scoring

Embeddings are normalised; the cosine similarity of each query against every gallery entry is
reduced to one score per class and classes are ranked. Because the gallery holds exactly one
reference per class, the average precision of a query is the reciprocal of the rank of its
correct class, so the reported mAP is the mean of those reciprocals, computed exactly. Rank 1 and
CMC@5 are reported alongside.

## Software versions

Python 3.12.3, PyTorch 2.11.0 built against CUDA 13.0, torchvision 0.26.0, timm 1.0.26, one
consumer graphics card with 10 GB of memory.

## Two notes on the numbers

* LeViT-128S converged poorly under the shared recipe (0.0932 mAP) and is reported unchanged,
  because the result reflects the interaction between one architecture and a recipe held
  constant across all backbones, not a property of the benchmark. The other 26 backbones fall
  between 0.5512 and 0.8399.
* The sweep ran on a machine that had rebuilt its crops before the ten box corrections recorded
  in `KNOWN_ISSUES.md` section 1, so it saw eight pre correction crops among its 7016 training
  images and two among its 1715 evaluation queries. The bound on the effect is stated there; no
  conclusion drawn from the sweep changes.
