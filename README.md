# cure2fewshot

<!-- TODO(release): sau khi Zenodo cap DOI cho tag v1.0.0, mo dong badge duoi day va thay XXXXXXX. -->
<!-- [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX) -->

A deterministic protocol that converts a public pill image collection into a few shot retrieval
benchmark, with a scored fitness rubric that decides whether the converted result is usable.

This repository holds the code, the frozen annotations and the expected outputs described in the
accompanying article. **It does not contain any image.** The image collection must be obtained from
its original authors.

## What a visitor can do here

Three journeys, in increasing order of effort:

1. **Inspect, no images needed, minutes.** The frozen annotations are the release:
   `protocol/frozen/` holds the 8811 bounding boxes, the near duplicate list, the flagged label
   pairs, the merge decisions, and the record of the ten boxes corrected before release.
   `expected/` holds the value every measurement should produce, so each number in the
   accompanying article can be checked against a file rather than taken on trust.
   `KNOWN_ISSUES.md` says what we know is imperfect, with measured impact.
2. **Reproduce, with the collection, an afternoon.** Obtain CURE from its authors, then follow
   [REPRODUCE.md](REPRODUCE.md) end to end: regenerate all 8811 crops and verify them pixel by
   pixel, rebuild every split byte for byte, rerun the hardening step and the four controls,
   rescore the rubric, and diff your run against `expected/` with `scripts/verify.sh`. A laptop
   is enough; a GPU only speeds up the embedding step.
3. **Stress it, or take it elsewhere.** Swap the frozen probe and rescore the rubric
   (REPRODUCE.md, section 8, expected outcomes included). Or apply the protocol to another
   collection: supply a localisation manifest and a class convention, and the audits, the
   hardening step and the rubric run unchanged. The one thing not shipped is the 27 backbone
   training sweep; [TRAINING_RECIPE.md](TRAINING_RECIPE.md) states its recipe in full, its scores
   ship in `expected/retrieval_baselines.csv`, and no rubric score depends on it.

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
| E | Near duplicate leakage removed; duplicated labels flagged and resolved |
| F | Reference hardening: the uniform composed background is replaced by a real one |
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
Tablets are convex, so solidity against the convex hull is a cheap sufficient test. Both mask
versions ship here (`harden_refs_v2.py` and `harden_refs_v1_ungated.py`) so the comparison is
reproducible.

## Layout

```
protocol/
  make_crops.py                phase A: regenerate every crop from the frozen box manifest,
                               plus an integrity check on the source archive and a
                               pixel by pixel verification of the result. RUN THIS FIRST.
  build_cure_fewshot.py        phases B to E and G: splits, cleaning, evaluation sample
  cure_fitness_check.py        four controls on frozen embeddings + duplicate label scan
  harden_refs_v2.py            phase F, integrity gated mask (USE THIS)
  harden_refs_v1_ungated.py    phase F, luminance threshold only, kept for the comparison
  score_rubric.py              the 100 point rubric and the accept or reject decision
  frozen/                      annotations produced by this work, not by the collection authors
expected/                      expected outputs, for checking your own run
scripts/verify.sh              compare your run against expected/, field by field
REPRODUCE.md                   step by step reproduction, and what the protocol writes
TRAINING_RECIPE.md             the recipe behind the 27 baseline scores (the sweep is not shipped)
KNOWN_ISSUES.md                defects known at the time of release
CHANGELOG.md                   what changed per version
```

`protocol/frozen/` holds our own measurements about publicly distributed files: bounding box
coordinates obtained by segmentation and manual correction, the near duplicate list, the flagged
label pairs and the merge decisions. No pixel of the collection is in this repository.

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
