# The baseline recipe, and the scripts that run it

`expected/retrieval_baselines.csv` holds the scores of 27 backbones on the released benchmark;
`expected/baseline_seeds.csv` holds four of them retrained on the released benchmark with three
seeds each. The scripts under `training/` produced both, and reproduce the recipe of the sweep
that trained the 27 checkpoints (July 2026). No rubric score depends on any of this.

## Two provenances, stated plainly

The 27 checkpoints were trained in July 2026 by the stack the scripts here are ported from, on
data that was not yet the released benchmark (`KNOWN_ISSUES.md` section 11): the crop directory
of the training project on that machine, which predates most of the manual box corrections, and
a hardened reference set made by the pre-protocol version of the ungated script, in which 58 of
the 388 references had lost the tablet itself. The 1.0 article reported the scores measured on
that data. `eval_baseline.py` run on that data reproduces them to four decimals (resnet18:
0.7087, 0.5592, 0.8810), which is how the port was checked.

What 1.1 reports instead:

| File | What it is |
|---|---|
| `expected/retrieval_baselines.csv` | the 27 deposited checkpoints scored by `eval_baseline.py` on the released benchmark: released crops, the gated gallery of 1.1 |
| `expected/retrieval_baselines_sweep_2026-07.csv` | the July figures, for the record |
| `expected/baseline_seeds.csv` | resnet18, efficientnet_b0, levit_128s and convnext_small retrained on the released benchmark, seeds 42, 123 and 223 |

A score in the first file is that of a model trained on a near relative of the benchmark and
tested on the benchmark. The third file is the clean measurement; the article reports it with
mean and standard deviation.

## What is trained

Each backbone from `timm` or torchvision, pretrained, trained on `train_clean.csv` (7006 images:
consumer photographs of folds 1 to 4 plus the reference images of the training pills, those
reference images replaced by their hardened version) and scored on `query_eval_fold0.csv` (1715
consumer photographs of 80 held out pills) against `gallery_fewshot_K1_hardened_v2.csv` (386
hardened references, one per class). The head covers all 390 classes; the 80 evaluation classes
have no training sample.

## Model, shared by every backbone (`training/common.py`, `build_model`)

* Generalised mean pooling over the feature map, exponent initialised at 3 and learned. Pure
  transformers (Swin, EVA) return a pooled vector and skip it.
* Projection to a 512 dimensional embedding (a two layer projection with layer normalisation for
  transformers and lightweight hybrids, a 1024 wide bottleneck for CNNs with 1024 channels or
  more, a linear map otherwise), then batch normalisation of the embedding, L2 normalised.
* Two cosine classification heads scaled by 64: one plain, one with an additive angular margin.

## Objective, shared

The sum of four terms: cross entropy with label smoothing 0.1 on the plain head (weight 1); the
same on the angular margin head (weight 0.2, 0.1 for lightweight hybrids); a triplet margin loss
with margin 0.3 on triplets mined as semihard at margin 0.2 (weight 1); a contrastive loss with
margins 0 and 1 (weight 1). Losses from `pytorch-metric-learning`.

## Optimisation and data, shared

AdamW with separate learning rates for backbone and heads; a linear warm up (5 epochs, 10 for
lightweight hybrids) then cosine annealing to 1e-6; a class balanced sampler (n classes times 2
images per batch) with gradient accumulation so that every backbone updates on an effective
batch of 128; gradient norm clipped at 5 (3 for small backbones and pure transformers); mixed
precision. Consumer images: resize to 1.15 times the training size, random resized crop (scale
0.8 to 1), flips in both directions, rotation up to 180 degrees, colour jitter 0.2 and 0.1, random
erasing with probability 0.3. Reference images: resize, horizontal flip, rotation up to 15
degrees, the same jitter. Deterministic cuDNN kernels, seed everywhere.

## Settings that vary by backbone group (`recipe_for`)

| Group | Backbones in the sweep | Physical batch | Accumulation | Backbone and head learning rates | Weight decay | Margin | Epochs | Training size | Backbone frozen at the start |
|---|---|---:|---:|---|---|---|---:|---:|---|
| super large | eva02_large, convnext_large, tf_efficientnetv2_l | 8 | 20 (16 for EfficientNet) | 2e-5, 2e-4 | 0.1 | 0.35 | 60 | 392 or 384 | 4 epochs, not for pure transformers |
| large | 12 | 16 | 8 | 3e-5, 3e-4 | 0.1 | 0.35 | 60 | 384 | no |
| medium | convnext_small | 16 | 8 | 3e-5, 3e-4 | 0.05 | 0.35 | 60 | 384 | no |
| small CNN | 7 | 32 | 4 | 5e-5, 5e-4 (ghostnet 2e-5, 2e-4; mobilenet_v3 4e-5, 4e-4) | 0.05 | 0.20 | 80 | 224 | 2 epochs |
| lightweight hybrid | efficientformerv2 s0 and s1, mobilevit s and xs, levit_128s | 32 | 4 | 3e-5, 2e-4 | 0.05 | 0.25 | 80 | 224 | 2 epochs |

Pure transformers take 0.3 times the backbone learning rate of their group. Every setting is
printed at the start of a run and written into the run's JSON.

## Resolution and scoring

Scoring resizes every image to 384 and the model interpolates to its own input size (224 for the
small and lightweight groups, 392 for the patch 14 transformers), so the evaluation resolution is
the same for every backbone and fourteen of them train at a resolution other than the one they
are scored at (`KNOWN_ISSUES.md` section 4). Embeddings are L2 normalised, each query is scored
against the gallery by cosine, the maximum over a class's entries is the class score and classes
are ranked; with one reference per class, mAP is the mean reciprocal rank. Rank 1 and CMC at 5
are reported beside it. A query whose class has no gallery entry counts as a miss at every rank;
none occurs in the benchmark.

## Checkpoint selection

The sweep kept, for each backbone, the checkpoint with the best score on the evaluation queries
among the epochs it scored (every tenth and the last five). That is selection on the test set,
and it is stated here because it was the sweep's rule. `train_baseline.py` keeps the rule so that
the deposited checkpoints and the seed study are comparable, and reports the last epoch's score
beside the selected one; `expected/baseline_seeds.csv` carries both.

## Software versions

The sweep: Python 3.12.3, PyTorch 2.11.0 with CUDA 13.0, torchvision 0.26.0, timm 1.0.26, one RTX
3080 with 10 GB. The scores on the released benchmark and the seed study: the same machine,
PyTorch 2.14.0 with CUDA 13.0, torchvision 0.29.0, timm 1.0.30, pytorch-metric-learning 2.9;
`versions` in every run's JSON records them.

## Two notes on the numbers

* LeViT-128S converged poorly under the shared recipe in the sweep (0.0932 mAP) and is reported
  unchanged; the seed study shows whether that repeats across seeds on the released data.
* The seed study's resnet18 lands well above the July figure for the same backbone. The gain has
  three sources that the July data lacked: corrected crops, a gallery whose references are
  tablets rather than donor background, and hardened training references of the same kind.
