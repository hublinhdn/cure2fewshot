# cure2fewshot

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22908278.svg)](https://doi.org/10.5281/zenodo.22908278)

A deterministic protocol that converts a public pill image collection into a few shot retrieval
benchmark, with a scored fitness rubric that decides whether the converted result is usable.

This repository holds the code, the frozen annotations and the expected outputs described in the
accompanying article. **It does not contain any image.** The image collection must be obtained from
its original authors.

## What a visitor can do here

Three journeys, in increasing order of effort:

1. **Inspect, no images needed, minutes.** The frozen annotations are the release:
   `protocol/frozen/` holds the 8811 bounding boxes, the near duplicate list, the flagged label
   pairs, the merge decisions, the resolution of the ten photographs the collection files under
   both faces of one pill, the review of 58 reference masks with the 29 it adopted, and the
   record of the ten boxes corrected before release. `expected/` holds the value every
   measurement should produce, so each number in the accompanying article can be checked
   against a file rather than taken on trust. `KNOWN_ISSUES.md` says what we know is imperfect,
   with measured impact, including three defects of the first release found while answering
   its review.
2. **Reproduce, with the collection, an afternoon.** Obtain CURE from its authors, then follow
   [REPRODUCE.md](REPRODUCE.md) end to end: regenerate all 8811 crops and verify them pixel by
   pixel, rebuild every split byte for byte, rerun the hardening step and the four controls,
   rescore the rubric, and diff your run against `expected/` with `scripts/verify.sh`. A laptop
   is enough; a GPU only speeds up the embedding step.
3. **Stress it, or take it elsewhere.** Move every rubric threshold and watch the verdicts
   (`protocol/rubric_sensitivity.py`), swap the frozen probe and rescore (REPRODUCE.md,
   section 8), or apply the protocol to another collection: supply a localisation manifest and
   a class convention, and the audits, the hardening step and the rubric run unchanged. The
   training stack behind the baselines ships under `training/`
   ([TRAINING_RECIPE.md](TRAINING_RECIPE.md)); the 27 deposited checkpoints score on the
   released benchmark as `expected/retrieval_baselines.csv` says, and no rubric score depends
   on any of it.

## What the protocol does

The evaluation setting that matters for pill identification is retrieval across conditions: a
catalogue holds one canonical image per product, while the query is a photograph taken by whoever
needs the answer. Public collections are organised by product and capture session rather than by
that division, so they cannot serve such an evaluation until they are converted.

Seven phases, all deterministic under a fixed seed:

| Phase | What it produces |
|---|---|
| A, B | Crops from a frozen bounding box manifest; class identifier `pillId__side` |
| C | Gallery of reference images, queries of consumer photographs, five folds split by pill |
| D | Few shot gallery holding one reference per class; classes without a reference dropped and logged |
| E | Near duplicate leakage removed; duplicated labels and duplicated photographs flagged and resolved |
| F | Reference hardening: the uniform composed background is replaced by a real one, under two integrity gates |
| G | A fixed, class balanced evaluation sample |
| Rubric | 100 points over five groups, and a recorded accept or reject decision |

## The one result worth knowing before you reuse this

A fitness rubric scores the artefact, not the correctness of the operations that produced it.

Phase F edits images, and an earlier version of its mask thresholded luminance alone. On white and
pale tablets, whose grey level sits inside the same band as the synthetic field they were composed
onto, that mask cut into the tablet and pasted the replacement background over part of it. The
damage made reference images harder to match, which **inflated the very quantity the rubric rewards**.
The broken variant scored 100 out of 100; the corrected one scores 92.

So every step that edits an image is gated on the integrity of the object it claims not to touch.
Tablets are convex, so solidity against the convex hull catches a bite. It does not catch a
straight cut, because half a tablet is still convex, and the first release's gated mask had
exactly that defect in 29 of 388 references, found when the article's reviewers asked for an
independent check (`KNOWN_ISSUES.md` section 10). Since 1.1 a second gate compares the region the
colour rule preserved with a mask from a different mechanism, every reference it flags is judged
by eye, and the decisions are frozen. Both mask versions ship here (`harden_refs_v2.py` and
`harden_refs_v1_ungated.py`) so the comparison is reproducible.

## Layout

```
protocol/
  make_crops.py                phase A: regenerate every crop from the frozen box manifest,
                               plus an integrity check on the source archive and a
                               pixel by pixel verification of the result. RUN THIS FIRST.
  build_cure_fewshot.py        phases B to E and G: splits, cleaning, evaluation sample
  make_duplicate_resolution.py  turns the source file hashes into the frozen duplicate decisions
  cure_fitness_check.py        four controls on frozen embeddings, two duplicate scans, and the
                               check of every evaluation query against every training image
  harden_refs_v2.py            phase F, integrity gated mask with the reviewed independent masks (USE THIS)
  harden_refs_v1_ungated.py    phase F, luminance threshold only, kept for the comparison
  reference_masks.py           an independent mask for every reference, and the review sheets
  measure_mask_damage.py       how far each hardening mask cut into the tablets
  score_rubric.py              the 100 point rubric and the accept or reject decision
  rubric_sensitivity.py        every rubric threshold moved on its own
  review_sheet.py              lays out image pairs for a check by eye
  frozen/                      annotations produced by this work, not by the collection authors
training/                      the training and scoring scripts behind the baselines (own requirements)
tests/                         unit tests on synthetic inputs, no image needed: python -m pytest tests
expected/                      expected outputs, for checking your own run
scripts/verify.sh              compare your run against expected/, field by field
REPRODUCE.md                   step by step reproduction, and what the protocol writes
TRAINING_RECIPE.md             the recipe behind the baselines, and what the scripts reproduce
KNOWN_ISSUES.md                defects known at the time of release
CHANGELOG.md                   what changed per version
LICENSE, LICENSE.txt           MIT; NOTICE.md says the licence covers the code and the annotations, not the CURE images
```

`protocol/frozen/` holds our own measurements about publicly distributed files: bounding box
coordinates obtained by segmentation and manual correction, the near duplicate list, the flagged
label pairs and the merge decisions, the duplicate photograph decisions, and the mask review with
29 binary silhouettes. No pixel of the collection is in this repository.

## Quickstart

Obtain the collection, then follow [REPRODUCE.md](REPRODUCE.md):

```bash
pip install -r requirements.txt
export CURE_RAW_ROOT=/path/to/data-root      # contains data/raw/CURE_dataset/
export CURE_CROPS_ROOT=$CURE_RAW_ROOT
export CURE_MANIFEST=$PWD/protocol/frozen/cure_crops_manifest.curated.csv
export CURE_FITNESS_OUT=$PWD/outputs

python protocol/make_crops.py inventory   # check the archive matches the recorded boxes
python protocol/make_crops.py build       # regenerate all 8811 crops
python protocol/make_crops.py verify      # confirm them pixel by pixel
```

Everything after that runs on a laptop. Only the embedding extraction benefits from a GPU, and it
works on CPU. Read [KNOWN_ISSUES.md](KNOWN_ISSUES.md) before trusting a `verify` result.

## Data

The CURE collection is released by the authors of Ling et al., CVPR 2020, at
<https://github.com/suiyiling/Few-shot-pill-recognition>. Please cite their paper and respect
whatever terms they state. This repository redistributes none of their images and no derived image.

## Citing

See `CITATION.cff`. Please also cite the accompanying article and the collection authors.

## Licence

MIT for the code and for the annotations in `protocol/frozen/`. See [LICENSE](LICENSE). The image
collection is not covered by this licence.
