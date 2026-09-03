# Reproducing the protocol

Every number in `expected/` was produced by the steps below. Nothing here needs a GPU except the
optional embedding extraction, which runs on CPU too, only slower.

The repository ships **no image**. Step 0 obtains the collection; step 1 regenerates every crop
from the frozen bounding box manifest. From step 2 onward, the work is CSV and JSON.

---

## 0. Obtain the collection

The CURE collection is released by its own authors at
<https://github.com/suiyiling/Few-shot-pill-recognition>. Follow their instructions, cite their
paper (Ling et al., CVPR 2020), and respect the terms they state. This repository redistributes
none of their images and no image derived from them.

Unpack it so the layout is:

```
<DATA_ROOT>/
  data/raw/CURE_dataset/Pill_Images/<pillId>/<side>/{Customer,Reference}/*.jpg
```

`<DATA_ROOT>` is whatever directory you like. The paths in the manifest are relative to it.

## 1. Environment

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

**Pillow is pinned to 12.0.0 on purpose.** Letterboxing resamples, so pixel identical crops need
that exact version. A different version yields the same boxes and the same content but may differ
by a few pixels at the padding boundary. It changes no split and no rubric score, but step 1's
`verify` will report the difference rather than hide it.

Set the two roots once per shell. Both usually point at the same directory:

```bash
export CURE_RAW_ROOT=/absolute/path/to/DATA_ROOT
export CURE_CROPS_ROOT=$CURE_RAW_ROOT
export CURE_MANIFEST=$PWD/protocol/frozen/cure_crops_manifest.curated.csv
export CURE_FITNESS_OUT=$PWD/outputs
```

| Variable | Meaning |
|---|---|
| `CURE_RAW_ROOT` | root the manifest's `src_rel` is relative to, so it contains `data/raw/CURE_dataset/` |
| `CURE_CROPS_ROOT` | root the manifest's `crop_rel` is written under |
| `CURE_MANIFEST` | the frozen bounding box manifest, 8811 rows |
| `CURE_FITNESS_OUT` | where splits, reports and scores are written |

## 2. Phase A: check integrity, then regenerate the crops

The collection carries no version number, so integrity is established by inventory. Run this
**before anything else**: a silently different archive would make every recorded box meaningless.

```bash
python protocol/make_crops.py inventory
```

It reads every `src_rel` and checks it exists and matches the `orig_w` x `orig_h` recorded in the
manifest. It covers the 8811 files the protocol uses, not all 8971 files in the distribution. It
writes nothing. Expect `INVENTORY PASS`.

Then build the crops, and verify them pixel by pixel:

```bash
python protocol/make_crops.py build
python protocol/make_crops.py verify
```

`build` writes 8811 PNGs of 384 x 384 under `CURE_CROPS_ROOT`, about 1.5 GB. `verify` re-crops each
one in memory and compares every pixel against what is on disk.

Expect `VERIFY PASS` with all 8811 crops identical. On the reference machine that is exactly what
this reports, so treat anything less as a problem with your archive or your Pillow version rather
than with the manifest. Ten boxes were corrected before the first release, for exactly this reason;
every old and new value is in `protocol/frozen/box_corrections.csv` and the episode is described in
[KNOWN_ISSUES.md](KNOWN_ISSUES.md).

## 3. Phases B to E and G: splits, cleaning, evaluation sample

`build_cure_fewshot.py` runs in two stages. The `clean` stage consumes the frozen leakage list and
the frozen label merge decisions, so copy them into the output directory first:

```bash
mkdir -p "$CURE_FITNESS_OUT"
cp protocol/frozen/leak_list.csv protocol/frozen/dedup_map.csv \
   protocol/frozen/manual_merges.csv "$CURE_FITNESS_OUT"/

python protocol/build_cure_fewshot.py splits
python protocol/build_cure_fewshot.py clean
```

This stage is pure CSV work under seed 42, so it reproduces byte for byte on any platform and any
library version. Check it against the frozen summary:

```bash
diff <(python -m json.tool "$CURE_FITNESS_OUT/cure_processed_summary.json") \
     <(python -m json.tool expected/cure_processed_summary.json)
```

Expected counts:

| Quantity | Value |
|---|---|
| classes after merging duplicated labels | 390 |
| gallery entries, one reference per class | 386 |
| evaluation queries, fold 0 | 1715 |
| training images, folds 1 to 4 | 7016 |
| fixed evaluation sample | 1280 |
| reference images available to harden | 388 |

## 4. Phase F: reference hardening

Phase F needs the raw photographs as well, because the replacement background is cropped from a
raw photograph of a different pill.

```bash
export CURE_GALLERY="$CURE_FITNESS_OUT/gallery_fewshot_K1_clean.csv"
export PHASE_F_V2_OUT=$PWD/outputs_v2

python protocol/harden_refs_v2.py
```

Compare against the frozen log. Every count should match:

```bash
python - <<'EOF'
import json
got=json.load(open("outputs_v2/hardened_v2_log.json"))["stats"]
exp=json.load(open("expected/hardened_log_v2.json"))["stats"]
print("got", got); print("expected", exp)
print("MATCH" if got==exp else "DIFFERS")
EOF
```

Expected: 388 references processed, median mask solidity 0.996, minimum 0.971, noise fallback
never used, 19 masks repaired by convex hull, 5 fell back to an ellipse and are recorded as
suspect.

To reproduce the **ungated** variant of the mask, the one whose defect the article reports, run
`protocol/harden_refs_v1_ungated.py` instead. It writes to its own directory and overwrites
nothing. It is kept only so the comparison in the article is reproducible; do not use it to build
a benchmark.

## 5. The four controls on frozen embeddings

```bash
python protocol/cure_fitness_check.py extract     # embeddings, cached as .npy, run once
python protocol/cure_fitness_check.py controls    # controls 1 to 4 plus the duplicate label scan
```

`extract` downloads two pretrained backbones through `timm`, `resnet50` and
`vit_base_patch16_224`, and embeds all 8811 crops. On a CPU this takes tens of minutes; on any GPU
it takes a few minutes. The embeddings are cached, so `controls` can be re-run freely.

Run it twice, once on the direct conversion and once on the hardened references, because the
article reports both:

```bash
# direct conversion
CURE_FITNESS_OUT=$PWD/outputs python protocol/cure_fitness_check.py all

# hardened references
CURE_MANIFEST=$PWD/outputs_v2/manifest_hardened_v2.csv \
CURE_FITNESS_OUT=$PWD/outputs_v2/fitness_v2 \
  python protocol/cure_fitness_check.py all
```

## 6. Score the rubric

**Read this before running it.** `score_rubric.py` takes the measurements from `--report`,
but it decides four of its criteria by looking for files inside `CURE_FITNESS_OUT`, by
name:

| Criterion | File it looks for in `CURE_FITNESS_OUT` |
|---|---|
| A1 retrieval direction | `gallery_full.csv` and `query_full.csv` |
| A2 few shot regime | `gallery_fewshot_K1.csv` |
| C1, C2 cleanliness | `removal_log.json` and `dedup_map.csv` |
| E1 shortcut handled | `gallery_fewshot_K1_hardened.csv`, **without a `v2` suffix** |

Point it at the wrong directory and it will not error. It will quietly score those
criteria lower. In particular, E1 awards 10 when a hardened gallery is present and 5 when
it is not, so scoring the hardened variant from a directory that has no hardened gallery
yields **87 rather than 92**, and the difference looks like a measurement change when it
is only a missing file.

The direct conversion scores from the splits directory as it stands:

```bash
CURE_FITNESS_OUT=$PWD/outputs \
  python protocol/score_rubric.py --report outputs/fitness_report.json --tag direct
```

The hardened variant needs one directory holding both the splits and the hardened gallery
under the name above. Assemble it explicitly rather than hoping:

```bash
mkdir -p outputs_v2/rubric_v2
cp outputs/gallery_full.csv outputs/query_full.csv outputs/gallery_fewshot_K1.csv \
   outputs/removal_log.json outputs/dedup_map.csv outputs_v2/rubric_v2/
cp outputs_v2/gallery_fewshot_K1_hardened_v2.csv \
   outputs_v2/rubric_v2/gallery_fewshot_K1_hardened.csv
cp outputs_v2/fitness_v2/fitness_report.json outputs_v2/rubric_v2/

CURE_FITNESS_OUT=$PWD/outputs_v2/rubric_v2 \
  python protocol/score_rubric.py --report outputs_v2/rubric_v2/fitness_report.json --tag v2
```

The copy that renames `gallery_fewshot_K1_hardened_v2.csv` is the step that is easy to
miss, and it is worth five rubric points.

Expected verdicts:

| Variant | Score | Verdict |
|---|---|---|
| direct conversion | 80 / 100 | accepted, fails criterion B2 only |
| hardened, integrity gated mask | 92 / 100 | accepted as a primary benchmark |
| hardened, ungated mask | 100 / 100 | **do not use**, the mask cut into pale tablets |

That last row is the point of the exercise. A rubric scores the artefact, not the correctness of
the operations that produced it, so the broken variant scores highest. Read the article's section
on the integrity gate before drawing any conclusion from a rubric total.

## 6b. What you now have, and which files are the benchmark

The scripts write intermediates as well as results, and several intermediates carry a
name close to a file you are meant to use. **Read this table before loading anything.**

### The benchmark. A study reusing the converted collection needs only these four

| File | Rows | Content |
|---|---:|---|
| `outputs_v2/gallery_fewshot_K1_hardened_v2.csv` | 386 | the gallery: one hardened reference image per class |
| `query_eval_fold0.csv` | 1715 | evaluation queries, the consumer photographs of the held out fold |
| `train_clean.csv` | 7016 | training images: consumer photographs of folds 1 to 4, plus the reference images of the training pills |
| `eval_sample_1280.csv` | 1280 | fixed class balanced subset, for measurements too expensive to repeat on all queries |

Columns are `crop_rel` (path relative to `CURE_CROPS_ROOT`), `label` (the class after
duplicate labels were merged) and `fold`.

### Manifests and split assignment

| File | Rows | Content |
|---|---:|---|
| `manifest_dedup.csv` | 8811 | every crop with its final class label and its fold |
| `outputs_v2/manifest_hardened_v2.csv` | 8811 | the same, with reference rows pointing at the hardened images |
| `folds.csv` | 392 | the fold assigned to each pill and side |

### The audit trail

| File | Rows | Content |
|---|---:|---|
| `leak_list.csv` | 8423 | cosine of every query to its own class in both backbones, and the leak flag |
| `dedup_map.csv` | 4 | class pairs the centroid scan flagged, with the decision taken on each |
| `removal_log.json` | | what was removed, at which threshold, and how much remained |
| `phaseD_log.json` | | the classes dropped for want of a reference image |
| `outputs_v2/hardened_v2_log.json` | 388 | per reference: accepted threshold, mask solidity, repair applied, suspect flag |

### Measurements and the decision

| File | Content |
|---|---|
| `fitness_report.json` | the four controls, per backbone and averaged |
| `rubric_score_*.json` | the rubric criterion by criterion, the total and the verdict |
| `cure_processed_summary.json` | the counts reported in the accompanying article |

### Do not load these by mistake

Five files carry `gallery` in the name and four carry `query`. Only the ones above are
the benchmark. In particular:

| Looks right, is wrong | Why |
|---|---|
| `gallery_fewshot_K1.csv` (388) | before duplicate labels were merged, and before hardening |
| `gallery_fewshot_K1_clean.csv` (386) | labels merged, but the uniform background shortcut is still there |
| `gallery_full.csv`, `gallery_eval_clean.csv` (388) | not reduced to one reference per class |
| `query_full.csv` (8423) | before the near duplicate check |
| `query_eval_fold0_preclean.csv` (1715) | same rows as the clean file but the pre merge labels |

Loading any of these changes what is being measured, and nothing will raise an error.

## 7. Compare everything against the frozen expectations

```bash
bash scripts/verify.sh
```

It diffs each JSON your run produced against the copy in `expected/`, ignoring only the fields
that record local absolute paths.

### What is in `expected/`

| File | Produced by | Checked by `verify.sh` |
|---|---|---|
| `cure_processed_summary.json` | step 3 | yes |
| `phaseD_log.json` | step 3, the few shot gallery | yes |
| `removal_log.json` | step 3, leakage and label merges | yes |
| `fitness_report_direct.json` | step 5 on the direct conversion | yes |
| `rubric_score_direct.json` | step 6 on the direct conversion | yes |
| `hardened_log_v2.json` | step 4, integrity gated mask | yes |
| `fitness_report_v2.json` | step 5 on the hardened references | yes |
| `rubric_score_v2.json` | step 6 on the hardened references | yes |
| `fitness_report_v1.json` | step 5 on the **ungated** variant | no, that variant is optional |
| `rubric_score_v1.json` | step 6 on the **ungated** variant | no, that variant is optional |
| `fitness_report_probe3_direct.json` | section 8, third probe on the direct conversion | no, optional test |
| `fitness_report_probe3_hardened.json` | section 8, third probe on the hardened references | no, optional test |
| `cure_crops_meta.json` | the original localisation run | no, see below |

`cure_crops_meta.json` is provenance, not an expectation: it records the segmentation model
variant, the SHA256 of its weights, the seed, the imaging library version and the letterboxing
rule that produced the frozen boxes. Nothing in this repository reproduces it, by design. The boxes
are frozen precisely so that the segmentation model never becomes a dependency.

### Why some fields are allowed to differ

The files in `expected/` were sanitised before release: fields recording an absolute path on the
machine that produced them, such as `manifest`, `report` and `sam_ckpt`, were rewritten as
repository relative paths. No measured value was touched. `scripts/verify.sh` ignores exactly
those fields and compares everything else strictly.

## 8. Optional: swap the probe and rescore

The article reports a sensitivity check: the four controls and the rubric rerun with a third
frozen probe, a self supervised ViT trained without labels, to measure how much of each number
belongs to the probe pair rather than to the collection. Everything it needs is already in this
repository; the probe is an argument, not a code change:

```bash
CURE_MANIFEST=$PWD/protocol/frozen/cure_crops_manifest.curated.csv \
CURE_FITNESS_OUT=$PWD/outputs_probe3 \
  python protocol/cure_fitness_check.py all --models vit_small_patch14_dinov2.lvd142m

CURE_MANIFEST=$PWD/outputs_v2/manifest_hardened_v2.csv \
CURE_FITNESS_OUT=$PWD/outputs_probe3_hardened \
  python protocol/cure_fitness_check.py all --models vit_small_patch14_dinov2.lvd142m
```

The probe weights download from the Hugging Face hub on first use, about 90 MB. Extraction is
a forward pass over 8811 crops per variant: minutes on a GPU, about an hour per variant on a
CPU. Then score each report with the same directory assembly as step 6, swapping in these
reports and output directories.

What to expect, measured on 5 August 2026. Reference copies sit in
`expected/fitness_report_probe3_direct.json` and `expected/fitness_report_probe3_hardened.json`:

| Quantity | published, two probes | third probe |
|---|---|---|
| direct gap in top1 | 0.046 | 0.021, B2 still zero |
| hardened gap in top1 | 0.132 | 0.113, B2 still the middle band |
| leakage at 0.985 | 0, max same class cosine 0.928 | 0, max 0.952 |
| duplicate label scan at 0.97 | 4 pairs flagged | 22 flagged, C2 falls to 5 of 10 |
| totals | 80 and 92 | 75 and 87 |
| verdicts | middle band cap; accepted as primary | both unchanged |

Both verdicts survive, and the whole difference in the totals sits in C2, whose centroid
threshold is calibrated to the supervised probe pair. Expect the bands and the criterion scores
to reproduce exactly; the third decimal of the measurements can move with your torch, timm and
imaging stack, and the hardened reference values were themselves measured on an independently
rebuilt copy. `scripts/verify.sh` does not check these two files. They anchor this optional
test only.

## Versions used for the published numbers

| Component | Version |
|---|---|
| Python | 3.12.3 |
| Pillow | 12.0.0 (pinned; affects crop pixels only) |
| PyTorch | 2.11.0, built against CUDA 13.0 |
| torchvision | 0.26.0 |
| timm | 1.0.26 |
| GPU | one consumer card, 10 GB |

The rubric depends only on the frozen backbones and the split files, both of which reproduce
exactly, so its scores are unaffected by a resampled padding pixel.

## Optional: the 27 baseline backbones

`expected/retrieval_baselines.csv` holds the scores of 27 backbones from `timm` trained on the
converted benchmark with one shared recipe and seed 42. That sweep is not part of this protocol
and is not required to reproduce any rubric score. The recipe is stated in full in
[TRAINING_RECIPE.md](TRAINING_RECIPE.md).
