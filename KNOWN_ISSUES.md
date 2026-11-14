# Known issues

Defects known at the time of release. Each one is listed with its measured effect, so that anyone
reusing this material can judge whether it matters for their purpose.

---

## 1. Ten bounding boxes were corrected before release. RESOLVED

**Status: fixed on 5 August 2026, before the first release. Nothing measured changed.**

Recorded here because the correction is part of the provenance of the manifest, and because the
audit trail is more useful than a silent edit.

### What was wrong

`protocol/frozen/cure_crops_manifest.curated.csv` has 8811 rows. Regenerating every crop from the
recorded boxes and comparing pixel by pixel showed 8801 identical and **ten different**. In each of
the ten the recorded box framed a different region of the same photograph, and in each it **cut
into the pill**, while the crop every published measurement had been computed on contained the whole
pill. That is the same failure mode as the Phase F mask defect this protocol exists to catch, so it
was caught by the same kind of check applied one stage earlier.

### Cause

The review tool rewrites the crop PNG the moment a box is redrawn, while the curated manifest
is only updated by a separate merge step run afterwards. That merge was run after the main
review sessions and carried 934 redrawn boxes into the manifest, marked `method=manual`. One
final resumed session then redrew ten more crops, and the merge step was never run again.
The ten are the last ten rows of the review decision log. From that moment the crop files were
correct and the manifest lagged one session behind, keeping the stale automatic box and the
stale `method=sam` label for exactly those ten.

Nothing in the source archive is implicated. The raw photographs carry no bounding boxes;
every box is an annotation produced by this work, and the divergence was a bookkeeping lag
between two artefacts of the annotation process on the machine that produced them.

### The correction

The redrawn boxes were recovered from the audit's own decision record, and every one was
verified by regenerating its crop and comparing pixel by pixel with the crop the measurements were
computed on. All ten now match exactly, so the manifest describes the measured artefact rather than
contradicting it.

Two of the ten were independently recovered by template matching the stored crop against its source
photograph, without using the audit record at all. Both agreed with the audit record exactly, which
is why the other eight are trusted.

Every old and new value is recorded in `protocol/frozen/box_corrections.csv`.

### What changed, and what did not

| Quantity | Before | After |
|---|---|---|
| crops regenerating pixel identically | 8801 of 8811 | **8811 of 8811** |
| consumer boxes from the segmentation model | 7547 | 7537 |
| consumer boxes corrected by hand | 876 | 886 |
| reference boxes, classical rule and hand corrected | 330 and 58 | unchanged |
| total crops, classes, gallery, queries | 8811, 390, 386, 8423 | unchanged |

Nothing measured moved. The box coordinates are read only when regenerating crops and when Phase F
picks a background donor window; no split file and no control reads them, and
`cure_fitness_check.py` needs only `crop_rel`, `pillId`, `side` and `kind`. Phase F was re-run in
full against the corrected manifest: all 388 hardened reference images came out pixel identical and
the log matched field for field, so none of the ten had been used as a donor. Every rubric
measurement was computed on the corrected crop files, so every control value, every rubric
criterion and every verdict is unaffected. The only numbers that moved are the two provenance
counts above, which were wrong before and are right now.

One provenance nuance was recorded here in 1.0: that the 27 backbone sweep had seen the ten
pre-correction crops. Re-examined on 14 November 2026 while answering the reviewers, that
statement was wrong by an order of magnitude. The sweep read the crop directory of the training
project on the machine it ran on, and that directory predates most of the manual box corrections,
not only the last ten: 150 of the 1715 evaluation queries and about an eighth of the training
photographs differ from the crops this manifest regenerates, by more than five percent of their
pixels. The sweep also read a hardened reference set that is not the one this protocol produces.
Section 11 records what was measured, how the published numbers were reproduced to the fourth
decimal from that data, and what 1.1 replaces them with.

---

## 2. Five hardened reference masks fall back to an inscribed ellipse. SUPERSEDED in 1.1

**Severity: low in 1.0, reported by design; resolved by section 10 in 1.1.**

Update, 1.1: all five were examined against an independent mask (section 10). In each case the
inscribed ellipse either clipped the tablet or kept a ring of the synthetic field around it, and in
each case the independent mask fitted the tablet. They now carry that mask, recorded as
`use_sam` in `protocol/frozen/reference_mask_review.csv`, and no reference of the gated variant is
flagged `suspect` any more. The paragraph below describes 1.0.

Phase F resolves the foreground mask by trying colour distance thresholds in ascending order and
accepting the first whose mask has a plausible area and a solidity of at least 0.97. For five of
the 388 reference images no threshold reaches that, and not even the convex hull repair does, so
the mask falls back to an inscribed ellipse. Those five carry a thin ring of the original synthetic
field around the tablet.

They are recorded as `suspect` in `outputs_v2/hardened_v2_log.json` rather than silently accepted.
This is discussed in the accompanying article. It is a property of those five images, not
a bug to fix.

---

## 3. Pixel identical crops are guaranteed only for tested Pillow versions

**Severity: cosmetic, documented.**

Letterboxing resamples with a LANCZOS kernel, so the padding boundary could differ by a few pixels
between Pillow versions. The crops were produced under 12.0.0, which `requirements.txt` pins;
11.2.1 was tested on 22 September 2026 and reproduces all 8811 crops pixel for pixel; other
versions are untested. A version that did move a pixel would still reproduce the same boxes and
the same image content, and every split file would be unaffected, because splits depend only on
the manifest. `make_crops.py verify` notices, by design.

One stage is measurably sensitive to the exact pixels: the Phase F threshold search. Rerun on a
second machine with a different imaging stack (5 August 2026), it resolved borderline masks
differently, thirteen ellipse fallbacks rather than five, with the same accepted solidity minimum
of 0.971. The integrity gate flagged every unresolved case as suspect rather than passing it, and
in the third probe rerun performed on that machine every rubric criterion still landed in the same
band. So the statistics of the mask search vary with library versions; the decisions gated on it
did not, in the one rerun observed.

Update, 23 September 2026: a fresh clone on that same machine, this time with Pillow pinned to
12.0.0 as `requirements.txt` requires, reproduced all 8811 crops pixel for pixel, all 388 hardened
references pixel for pixel and the hardening log field for field (five ellipse fallbacks, minimum
solidity 0.971). The thirteen fallback result above therefore belongs to the imaging stack of that
earlier environment, not to the machine or its GPU, and the pin is what makes Phase F reproduce.

---

## 4. Fourteen of the 27 baseline backbones train at a resolution other than 384

**Severity: none, reported rather than corrected.**

The baseline sweep holds the evaluation resolution fixed at 384 for every backbone so that scores
stay directly comparable. Training resolution follows each backbone's own convention: 384 for
thirteen of the twenty seven, 224 for the twelve compact ones, and 392 for the two whose patch
size requires it. The shift between training and evaluation resolution therefore applies to
fourteen backbones, wide for the twelve and slight for the other two, and is reported in the
article rather than corrected for. The sweep is not part of this protocol and is not needed to
reproduce any rubric score.

---

## 5. TF32 convolutions on Ampere GPUs perturbed the ResNet50 probe. RESOLVED in 1.0.1

**Status: fixed on 23 September 2026 in `protocol/cure_fitness_check.py`. `expected/` is unchanged
and no rubric score was ever affected.**

### What was observed

A fresh clone of 1.0.0 on a Linux machine with an RTX 3080 (REPRODUCE.md, "Versions used")
reproduced every split file byte for byte, every hardened reference pixel for pixel, both rubric
totals and both verdicts, but `scripts/verify.sh` reported the two controls reports as differing:
24 of the 42 measured floats in the direct report and 25 of 42 in the hardened one, by up to
9.6e-4, every one of them a ResNet50 field or a mean that includes ResNet50. All 26 integers and
6 strings, every rubric score and both verdicts were identical, and the ViT fields agreed to
1.7e-7. Two `note` strings of the direct rubric file also differed, because 0.2515 had become
0.2518 and rounds to a different third decimal; the scores beside them were equal.

### Cause

PyTorch enables TF32 for fp32 convolutions by default on NVIDIA GPUs of the Ampere generation and
later (`torch.backends.cudnn.allow_tf32` is `True`), so a convolutional probe computes with 10 bit
mantissas on such a GPU and with full fp32 everywhere else. Measured against the reference
machine's embeddings: ResNet50 differed by up to 7.0e-3 (median per image 1.5e-3, minimum cosine
0.99982) on every one of the 8811 crops. Counted directly rather than inferred from the rounded
score, **138 of the 8423 consumer queries retrieved a different nearest reference**, and because
the swaps went both ways the number of correct top1 decisions moved by only five, from 1802 to
1797. The ViT probe, dominated by matrix products for which TF32 is off by default, differed by at
most 1.6e-5. Rerunning the extraction on the same machine with the two TF32 flags off, and nothing
else changed, brought the ResNet50 embeddings to within 3.0e-6 of the reference machine, the 42
floats of the direct report to within 1.0e-6, the two `note` strings back to their expected text,
and **all 8423 retrieval decisions to exactly those of the reference machine**.

### The correction

`extract()` sets `torch.backends.cudnn.allow_tf32` and `torch.backends.cuda.matmul.allow_tf32` to
`False` before loading the probes, which is harmless on CPU and on Apple silicon. `verify.sh` no
longer compares the `note` strings, prints the largest float deviation of every file, uses a
default tolerance of 1e-5 sized from the two measurements in REPRODUCE.md section 7, and names
this pattern in its exit message when it sees it.

### What changed, and what did not

| Quantity | 1.0.0 code on the CUDA machine (TF32 on) | 1.0.1 (TF32 off) |
|---|---|---|
| ResNet50 embeddings, largest difference to the reference machine | 7.0e-3 | 3.0e-6 |
| direct controls report, largest float deviation | 9.6e-4 | 1.0e-6 |
| hardened controls report, largest float deviation | 6.0e-4 | 1.1e-5, in control 3, see section 6 |
| queries retrieving a different nearest reference, direct | 138 of 8423 | 0 of 8423 |
| correct cross domain top1 decisions, ResNet50, direct | 1797 against 1802 | identical |
| rubric scores, totals and verdicts | identical | identical |
| `verify.sh` | 5 of 8 files match | 8 of 8 |

The published numbers were computed with full fp32 on the reference machine and needed no
change. Anyone who ran the 1.0.0 code on an Ampere or later GPU should re-extract the embeddings
with the current code before comparing against `expected/`.

---

## 6. The domain separability AUC depends on the scikit-learn version at the 1e-5 level

**Severity: cosmetic, documented, below every decision boundary.**

Control 3 fits a logistic regression with the lbfgs solver over 5 stratified folds and scores the
area under the ROC curve of its out of fold decision values. The solver stops on a tolerance, so a
different scikit-learn version, or a different BLAS under it, can settle on slightly different
coefficients from the same embeddings.

Measured on 23 September 2026 for the hardened variant's ResNet50 probe. The reference value in
`expected/` is 0.9919574655062049.

| Embeddings | scikit-learn | AUC | Difference |
|---|---|---:|---:|
| reference machine | 1.6.1 | 0.9919574655062049 | 0 |
| CUDA machine, TF32 off | 1.6.1 | 0.9919580774780884 | 6.1e-7 |
| CUDA machine, TF32 off | 1.9.1 | 0.9919461440263590 | 1.1e-5 |

The AUC ranks 388 reference images against 8423 consumer ones, so one swapped pair moves it by
3.1e-7; 1.1e-5 is about 37 swapped pairs out of 3.3 million. Criterion E1 turns at 0.95 and the
value is 0.992, so nothing the rubric decides moves. `requirements.txt` therefore keeps a lower
bound on scikit-learn rather than pinning it, and `scripts/verify.sh` carries a default tolerance
of 5e-5, which covers this while still catching a single query changing rank (1.2e-4).

---

## 7. The ungated Phase F variant could not be reproduced from REPRODUCE.md. RESOLVED in 1.0.2

**Status: fixed on 23 September 2026 in `protocol/harden_refs_v1_ungated.py`. The gated variant,
every split file and every rubric score were never affected.**

`harden_refs_v1_ungated.py` read the environment variable `PILL_PROJ` for the raw photographs it
cuts donor backgrounds from, and for its default manifest. No other script uses that name and
REPRODUCE.md never mentions it, so following REPRODUCE.md exactly left it unset. It then defaulted
to a directory that does not exist in a fresh clone, every donor lookup failed, and the script fell
back to Gaussian noise for all 388 references while reporting success. The ungated images a reader
obtained were therefore not the ones the article describes, and nothing said so.

It now reads `CURE_RAW_ROOT` and `CURE_CROPS_ROOT`, the names `harden_refs_v2.py` already used, and
defaults its manifest to `protocol/frozen/`; `PILL_PROJ` is still honoured as a fallback. Verified
on 23 September 2026 on the CUDA machine from a fresh clone: 388 references, zero noise fallbacks,
and the log matches the frozen one field for field.

One trap remains, documented rather than changed, because the file names are what the first release
published: the ungated script writes `gallery_fewshot_K1_hardened.csv` into `CURE_FITNESS_OUT`,
which is the exact name `score_rubric.py` looks for when it awards E1. Run it into the directory
holding the direct conversion's splits and the direct conversion will then score 85 rather than 80.
REPRODUCE.md section 4 says to give it a directory of its own.

---

## 8. Mask damage is measured from the output images, not from the mask the script used

**Severity: none, stated so that the two solidity numbers in the release are not confused.**

`protocol/measure_mask_damage.py` never sees the mask a hardening run computed. It recovers the
preserved region by comparing the original crop with the hardened one, which is the honest test:
it measures what the step actually left alone rather than what it believed it was leaving alone.
The consequence is that the gated variant has two solidity figures, and they are not the same
quantity:

| Number | Where | What it is |
|---|---|---|
| 0.9708 | `expected/hardened_log_v2.json`, `solidity_min` | the mask the script accepted, measured by the script's own gate |
| 0.9730 | `expected/mask_damage_v2.json`, `solidity_min` | the region actually preserved in the written image, measured afterwards |

Compare like with like: the ungated variant's 0.8184 in `expected/mask_damage_v1.json` is the
second kind, so it belongs beside 0.9730, not beside 0.9708.

The exposure figure has the same character. Measured on the gated mask it says how much of the
**intact** tablet falls inside the band a luminance rule treats as background, 324 of 388
references above five percent. Measured on the ungated mask it says how much of what **survived**
falls there, 189 of 388, a smaller number for the worse mask, because the pale pixels it had
already removed are no longer part of what is measured.

---

## 9. The collection files ten photographs of pill 81 under both faces. RESOLVED in 1.1

**Status: found on 14 November 2026 by the image level duplicate scan that 1.1 adds; handled at the
clean stage from a frozen decision file. No evaluation query and no rubric score is affected.**

### What was found

Hashing the bytes of every consumer photograph the manifest uses (`make_crops.py duplicates`)
finds ten groups of byte identical files, every one of them a photograph of pill 81 stored once
under `81/top/Customer` and once under `81/bottom/Customer` (file numbers 3578 to 3588 against
3601 to 3611). The whole archive holds eleven such groups; one of the eleven photographs had been
excluded from the manifest at localisation. Nine of the ten pairs localise to identical boxes and
therefore identical crops; the tenth pair was boxed by hand on both copies and the two boxes
differ by a few pixels. This is a property of the collection as distributed: the file bytes are
identical, the names are not.

### Why the 1.0 scan missed it

The 1.0 duplicate label scan tested class pairs, and it tested a pair only after the centroids of
their consumer photographs agreed to cosine 0.97 in both probes. The two faces of pill 81 each
hold twelve photographs of their own besides the ten shared ones, and those twelve differ enough
that the centroids agree only to 0.916 and 0.944. The pair was never counted, so the ten image
pairs at cosine 1.000 were never seen. The reviewer's remark that "no detected samples above a
threshold does not establish the absence of leakage" was right in a way the 1.0 release could not
show.

### The correction

1.1 scans at the image level and without a gate: every pair of crops with cosine at least 0.99 in
either probe (`near_duplicate_scan`, 46 pairs in the direct conversion, 10 across classes, all
pill 81), and every pair of byte identical source files. The class pair scan keeps its centroid
trigger and gains an image trigger, three or more cross class image pairs, whose suggestion is
`resolve_images` rather than `merge`: the two classes are distinct, they share photographs.

The two reference photographs of pill 81 are themselves near identical under both probes
(cosine 0.9996, the highest of the 196 pills; the next is 0.934), so the face of a shared
photograph cannot be settled from the image. One copy of each is kept, under the face that sorts
first in the merge step's canonical order (bottom before top); the rule and every pair are in
`protocol/frozen/duplicate_resolution.csv`, generated by `make_duplicate_resolution.py` from the
scan's output and applied at the clean stage. Pill 81 lies entirely in fold 1.

### What changed, and what did not

| Quantity | 1.0.4 | 1.1 |
|---|---|---|
| consumer photographs after cleaning | 8423 | 8413 |
| training images (folds 1 to 4) | 7016 | 7006 |
| evaluation queries, gallery entries, classes | 1715, 386, 390 | unchanged |
| control values, rubric scores and verdicts | | unchanged (the controls measure the collection as distributed, before cleaning, like the leakage check) |
| class pairs flagged by the label scan | 4 | 5, the new one `resolve_images` |
| fixed evaluation sample of 1280 | | re-drawn: the per class permutations after `81__top` in sort order shift |

The 27 backbone sweep of 1.0 trained on the ten duplicated photographs under both labels; the
seed study of 1.1 (`expected/baseline_seeds.csv`) trains on the corrected set.

---

## 10. The solidity gate cannot see a straight cut, and the released gated mask had them. RESOLVED in 1.1

**Status: found on 14 November 2026 with an independent mask; 29 of 388 gated masks replaced after
a check by eye. The gated gallery of 1.0.4 is superseded.**

### What was wrong

The integrity gate of Phase F accepts a mask when its solidity, area over convex hull area, is at
least 0.97. A convex tablet whose lower half has been replaced by background is still convex:
solidity 1.000, gate passed. The 1.0 article stated the gate as evidence that the gated mask
"does not cut into the tablet". It is evidence only against bites.

The measurement that shows it (`measure_mask_damage.py --masks`) compares the region Phase F
preserved with a mask obtained by a different mechanism: the segmentation model that produced the
frozen boxes (SAM ViT-B, same weights, SHA256 in `expected/cure_crops_meta.json`), prompted on the
original reference crop with a positive point at the centre and negative points on the synthetic
field, candidates that touch the field's border rejected (`reference_masks.py`). The quantity
reported is the share of that mask's core, the mask eroded by three pixels so that a boundary
disagreement between two mechanisms does not count, that the hardened image replaced.

Against that mask, the gated variant of 1.0.4 replaced more than one percent of the core in 47
references and more than a fifth in 16. The ungated variant: 209 and 70. The cause in every real
case is a luminance gradient across the tablet: the colour distance threshold, taken from the
border, falls below the darker or the brighter half, the largest connected component is that half,
and its hull is convex.

### The check by eye, and the repair

An automatic mask is not evidence on its own: SAM merged the tablet with the field for 22 of the
388 references (pale tablets on a pale field), and the measurement then flags an intact reference.
Every reference the measurement flagged, and every reference for which SAM had no candidate inside
the field's border, 58 in all, was laid out as original crop, region preserved by the colour rule,
SAM mask, hardened image, and judged by eye. The decisions, one line each with the reason, are in
`protocol/frozen/reference_mask_review.csv`:

| Decision | References | What was seen |
|---|---:|---|
| `use_sam` | 29 | the colour rule had cut the tablet (24, straight cuts and lost rims), or the ellipse fallback had clipped it or kept field around it (5); the SAM mask fits |
| `keep_rule` | 29 | the SAM mask is the field or includes a strip of it (15), or the two masks agree to within their boundary (14) |

`harden_refs_v2.py` reads that file: for `use_sam` it uses the mask stored under
`protocol/frozen/reference_masks/`, dilated by two pixels exactly as the colour rule mask is, and
logs `repair: independent_mask`. The masks are binary silhouettes, derived annotations like the
boxes; no pixel of the collection is redistributed. The other 359 references are hardened exactly
as in 1.0.4 and are pixel identical to it.

### What changed, and what did not

The gated gallery differs from 1.0.4 in 29 of 386 entries. Every number measured on the hardened
references was regenerated and is recorded in CHANGELOG.md under 1.1.0: the hardening log, the
four controls, the rubric, the sensitivity sweep and the mask damage statistics. The measurement
against the independent mask now reports zero core removal for the 29 repaired references by
construction; its value lies in the other 359 and in the ungated variant, where it is unchanged.

---

## 11. The 27 backbone sweep did not run on the benchmark this protocol produces. RESOLVED in 1.1

**Status: found on 14 November 2026 while releasing the training scripts. The numbers in the 1.0
article's baseline table were reproduced to the fourth decimal from the data the sweep actually
used, and that data is not the released benchmark.**

### What was found

`TRAINING_RECIPE.md` of 1.0 described the sweep as trained and evaluated on the files of
REPRODUCE.md section 6b. Scoring the sweep's own resnet18 checkpoint with the 1.1 evaluation script
on those files gave mAP 0.7561 against the published 0.7087. The published value is recovered
exactly, 0.7087, 0.5592, 0.8810, only with two substitutions: the crop directory of the training
project on the sweep machine, which differs from the released crops in 150 of the 1715 evaluation
queries and about an eighth of the training photographs (section 1), and a hardened reference set
produced in July 2026 by the pre-protocol version of the ungated script, in which 58 of the 388
references kept between 0.1 and 1.9 percent of their pixels, that is, the tablet itself had been
replaced by the donor background; eleven of those 58 are gallery entries of evaluation classes.

### What 1.1 does

* `training/` holds the training and evaluation scripts, ported from the sweep's stack and
  checked against it: on the sweep's own data, `eval_baseline.py` reproduces every published
  figure of resnet18 to four decimals.
* The 27 checkpoints are deposited at `10.5281/zenodo.24217212` and
  `expected/retrieval_baselines.csv` now carries their scores on the released benchmark, computed
  by `eval_baseline.py`; the July figures are kept beside them in `expected/retrieval_baselines_sweep_2026-07.csv`
  for the record. The checkpoints were trained on the sweep's data, so these are scores of models
  trained on a near relative of the benchmark and tested on the benchmark.
* Four representative backbones (resnet18, efficientnet_b0, levit_128s, convnext_small) were
  retrained on the released benchmark with three seeds each, `expected/baseline_seeds.csv`; the
  article reports mean and standard deviation.
* One more property of the sweep is stated rather than hidden: the checkpoint it kept for each
  backbone is the one with the best score on the evaluation queries themselves, measured every
  ten epochs and in the last five. `train_baseline.py` keeps the same rule, because it is the
  sweep's rule, and reports the last epoch's score beside it; the seed study lists both.
