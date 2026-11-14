# Changelog

## 1.1.0, 2026-11 (revision of the accompanying article)

Zenodo Version DOI: to be recorded after the release. Concept DOI unchanged: `10.5281/zenodo.22908278`.

The peer review of the article asked for four things the 1.0 release could not show: a
sensitivity analysis of the rubric, an independent check of the integrity gate, a duplicate
check across the training and evaluation partitions, and the training scripts behind the
baselines. Doing them found three defects in 1.0.4, each recorded in `KNOWN_ISSUES.md` (sections
9 to 11) with its measured effect. Every number in `expected/` was regenerated on machine B
(REPRODUCE.md, "Versions used"); the direct conversion's control values and all three rubric
verdicts are unchanged.

Found and fixed:

* **Ten photographs of pill 81 are filed under both faces of the collection as distributed**
  (`KNOWN_ISSUES.md` section 9). The 1.0 label scan could not see them because it tested image
  pairs only after the class centroids agreed. 1.1 hashes every source file
  (`make_crops.py duplicates`), scans every image pair without a gate (`near_duplicate_scan`),
  adds a shared image trigger to the class pair scan, and resolves the ten groups at the clean
  stage from `protocol/frozen/duplicate_resolution.csv`: training images 7016 to 7006, consumer
  photographs 8423 to 8413, nothing else moved.
* **The solidity gate let straight cuts through, and the 1.0.4 gated gallery had them**
  (section 10). Measured against an independent mask from the segmentation model
  (`reference_masks.py`, `measure_mask_damage.py --masks`), the 1.0.4 gated mask had replaced
  more than a fifth of the tablet core in 16 references and more than one percent in 47; the
  ungated mask, 65 and 199. Every flagged reference was judged by eye against the independent
  mask (`protocol/frozen/reference_mask_review.csv`, 58 rows): 29 masks replaced, 29 kept because
  the independent mask was wrong or no better. The gated gallery differs from 1.0.4 in those 29
  entries; its controls moved in the fourth decimal (cross domain top1 0.1654, gap 0.1319) and
  its rubric is unchanged at 92. After the repair the measurement flags 18 references, all of
  them cases where the independent mask is the field and the colour rule mask was kept.
* **The 27 backbone sweep had run on data that is not the released benchmark** (section 11): a
  crop directory predating most manual box corrections and a hardened set in which 58
  references had lost the tablet. The 1.0 statement that the sweep saw "eight pre correction
  crops" was wrong. `training/` now ships the stack (`train_baseline.py`, `eval_baseline.py`,
  `summarize_seeds.py`, `common.py`), checked by reproducing the July figures to four decimals
  on the July data; `expected/retrieval_baselines.csv` is replaced by the deposited checkpoints'
  scores on the released benchmark, the July table kept as
  `expected/retrieval_baselines_sweep_2026-07.csv`, and `expected/baseline_seeds.csv` adds four
  backbones retrained on the released benchmark with three seeds.

Added:

* `protocol/rubric_sensitivity.py` and `expected/rubric_sensitivity.json`: every threshold of
  the rubric moved on its own, totals and verdict classes recorded, margins of every measured
  quantity. Three of 35 moves change a verdict, all on the direct conversion: the partial gap
  threshold lowered from 0.05 to 0.04 or 0.03 (the measured gap is 0.046), and the cap rule
  switched off. `score_rubric.py` keeps its thresholds in one table and writes them into the
  score file.
* `cure_fitness_check.py partitions` and `expected/cross_partition_direct.json`: every
  evaluation query against every training image; largest cosine 0.973, none at or above the
  leakage threshold; the twenty nearest pairs of each kind for a check by eye, with
  `review_sheet.py` to lay them out.
* `cure_fitness_check.py controls --drop-refs` and `harden_refs_v2.py`'s
  `gallery_fewshot_K1_hardened_v2_nosuspect.csv`: the controls and the gallery without chosen
  references, for anyone who prefers to drop a flagged reference rather than repair it. Since
  the repair, no reference is flagged suspect and the file equals the gallery.
* `tests/`: 38 unit tests on synthetic inputs, no image of the collection needed
  (`python -m pytest tests`).
* `scripts/verify.sh` checks three more files; `IGNORE_KEYS` covers the new path fields.

Changed:

* `expected/` regenerated on machine B. Against the 1.0.4 files the direct controls differ only
  in the duplicate scan section; the hardened controls in the fourth decimal; every rubric score
  and verdict is the same. `protocol/frozen/dedup_map.csv` has five rows and two new columns
  (`trigger`, `suggest` now includes `resolve_images`); `protocol/frozen/leak_list.csv` was
  regenerated with the same flags.
* `eval_sample_1280.csv` is re-drawn: the per class permutations after `81__top` in sort order
  shift once ten rows are gone.
* `measure_mask_damage.py` reports three more quantities when given `--masks`; the solidity and
  exposure figures of 1.0.2 are unchanged in meaning.
* `harden_refs_v2.py` can be imported without environment variables; `main()` reads them.

Verified before tagging: see the entry's closing note once the fresh clone verification has run.

## 1.0.4, 2026-09-30

Zenodo Version DOI `10.5281/zenodo.23051213`. Concept DOI unchanged: `10.5281/zenodo.22908278`.

Author records only, again. No script, frozen annotation or expected output changed; every
number in `expected/` is the one 1.0.2 shipped.

* `CITATION.cff`, `.zenodo.json` and `LICENSE`: the author list is Linh Do Nhat, Dzi Lam Tran
  Tuan, Bay Nguyen Van and Vinh Truong Hoang, matching the accompanying article. The fifth name
  recorded in 1.0.3 was withdrawn at the authors' request.

## 1.0.3, 2026-09-27

Zenodo Version DOI `10.5281/zenodo.22994539`. Concept DOI unchanged: `10.5281/zenodo.22908278`.

Author records only. No script, frozen annotation or expected output changed; every number in
`expected/` is the one 1.0.2 shipped, and `scripts/verify.sh` compares against the same files.

* `CITATION.cff`, `.zenodo.json` and `LICENSE`: Bay Nguyen Van (Faculty of Information
  Technology, Ho Chi Minh City Open University) joins the authors, third in the order, matching
  the accompanying article. Linh Do Nhat's affiliation is now the same faculty, and the contact
  address is linhdn.ai25@gmail.com.
* `CITATION.cff`: the commented `preferred-citation` block still split Dzi Lam Tran Tuan's name
  the old way; corrected to given name Dzi, family name Lam Tran Tuan, and the new author added
  there too.

## 1.0.2, 2026-09-23

Zenodo Version DOI `10.5281/zenodo.22910727`. Concept DOI unchanged: `10.5281/zenodo.22908278`.

Auditing the accompanying article line by line against this release found two gaps in the release
itself, and one number that had been inferred rather than counted. No split file, hardened image,
control value or rubric score changed.

* `protocol/harden_refs_v1_ungated.py` read an undocumented variable, `PILL_PROJ`, for its donor
  photographs. Following REPRODUCE.md exactly therefore did not reproduce the ungated variant: the
  donor lookup failed silently and every reference fell back to Gaussian noise. It now reads
  `CURE_RAW_ROOT` and `CURE_CROPS_ROOT` like the gated script (`KNOWN_ISSUES.md` section 7).
* New `protocol/measure_mask_damage.py`, with `expected/mask_damage_v1.json` and
  `expected/mask_damage_v2.json`. The article reports how far the ungated mask cut into the
  tablets; until now nothing in the release let a reader check it. The measurement does not trust
  the mask a run used: the pixels identical between the original and the hardened crop are the
  region the step actually preserved, and their largest component's solidity says whether a convex
  tablet was bitten into. Ungated minimum solidity 0.818 against 0.973 gated, 126 of 388 masks
  below 0.98 solidity against 4 (`KNOWN_ISSUES.md` section 8 defines both quantities).
* The TF32 effect of 1.0.1 is now counted rather than inferred from the rounded score: 138 of the
  8423 consumer queries retrieved a different nearest reference, and the correct top1 count moved
  from 1802 to 1797 because the swaps went both ways. With TF32 off all 8423 retrieval decisions
  match the reference machine exactly.
* REPRODUCE.md documents `CURE_EMB_REUSE`, gives the ungated variant its own output directory and
  says why that matters for criterion E1, states the crop footprint as the measured 1.06 GB rather
  than "about 1.5 GB", and adds step 4b for the new measurement.

Verified on 23 September 2026 on the CUDA machine from a fresh clone: the ungated variant now
reproduces the frozen log field for field with zero noise fallbacks, and the new measurement
reproduces the values the article reports.

## 1.0.1, 2026-09-23

Zenodo Version DOI `10.5281/zenodo.22910309`. Concept DOI unchanged: `10.5281/zenodo.22908278`.

A fresh clone of 1.0.0 on a second machine (Linux, RTX 3080, CUDA) reproduced all 8811 crops pixel
for pixel, the fifteen split CSV files byte for byte, the 388 hardened references pixel for pixel,
every rubric score and both verdicts, but not the ResNet50 control values: PyTorch computes fp32
convolutions in TF32 by default on that GPU generation, which moved the probe's embeddings by up to
7e-3 and the control values by up to 1e-3 (`KNOWN_ISSUES.md` section 5). Changes:

* `protocol/cure_fitness_check.py`: `extract` disables TF32 before loading the probes. With that,
  the CUDA machine agrees with the reference machine to 3e-6 in the embeddings and 1e-6 in the
  control values.
* `scripts/verify.sh`: default `FLOAT_TOL` 1e-5, sized from the two measurements in REPRODUCE.md
  section 7 (one query changing rank moves a control by 1.2e-4, so a changed result is still
  caught); the rubric's `note` strings, which embed values rounded to three decimals, are no longer
  compared, the scores beside them still are; every file prints its largest float deviation; the
  exit message separates a changed result from numerical noise and names the TF32 pattern.
* `REPRODUCE.md`, `KNOWN_ISSUES.md` (sections 3, 5 and 6) and `requirements.lock`: the second
  machine's environment and every measurement above recorded, including a new section on control
  3's dependence on the scikit-learn version.
* No change to `protocol/frozen/`, to `expected/`, or to any split, hardening or rubric output.

Verified before tagging, 23 September 2026, on the CUDA machine with this code: `verify.sh` matches
8 of 8 files, the largest float deviation being 1.0e-6 in the direct controls report and 1.1e-5 in
the hardened one, the latter traced to the scikit-learn version rather than to the data
(`KNOWN_ISSUES.md` section 6). Every count, flag, string and rubric score is identical.

## 1.0.0, 2026-09-23

First public release. Zenodo Version DOI `10.5281/zenodo.22908279`; concept DOI, which always
resolves to the latest version and is the one in `CITATION.cff` and the README badge:
`10.5281/zenodo.22908278`.

Contents:

* Protocol scripts for phases A to G, the four frozen embedding controls and the conversion
  fitness rubric (`protocol/`). The rubric caps its verdict at the middle band when any group B
  criterion is zero, whatever the total: two image sources that are not materially separated
  make the task retrieval within one domain.
* Frozen annotations: the bounding box manifest of 8811 crops, the near duplicate leakage list,
  the flagged duplicate label pairs, the merge decisions, and the record of ten boxes corrected
  before release (`protocol/frozen/`, see `KNOWN_ISSUES.md` section 1).
* Both variants of the Phase F mask, the integrity gated one and the earlier ungated one, so the
  comparison in the accompanying article is reproducible.
* Expected outputs for checking a local run, including the third probe reports and the 27
  backbone baseline scores, and `scripts/verify.sh` to compare against them: integers, strings
  and rubric scores exactly, measured floats to `FLOAT_TOL` (default 1e-6). The tolerance is
  sized from a measurement: re-extracting all 8811 probe embeddings from scratch on the reference
  machine reproduced ResNet50 bit for bit, moved the ViT embeddings by at most 3.6e-6 and the
  controls report's floats by at most 5.4e-7, with every count, flag and rubric score identical.
* `requirements.txt` with the version bounds and `requirements.lock` with the exact versions
  behind every file in `expected/`, by machine (REPRODUCE.md, "Versions used").
* `REPRODUCE.md` (step by step, with the list of what the protocol writes and which files are
  the benchmark), `TRAINING_RECIPE.md` (the recipe behind the baseline scores; the sweep itself
  is not part of the protocol) and `KNOWN_ISSUES.md`.
* Contains no image.

Verified before tagging, 22 September 2026, on the reference machine: REPRODUCE.md steps 3 to 7
run end to end give `verify.sh` 8 of 8 files matching; the 388 hardened references and the 9
split CSV files are byte identical to the run that produced `expected/`; all 8811 crops
regenerate pixel for pixel under Pillow 11.2.1 as well as 12.0.0.

### Procedure for cutting a tag

1. `CITATION.cff`: `version` and `date-released` (the tag date). The `doi` field holds the
   concept DOI and does not change between versions.
2. `.zenodo.json`: `version`.
3. `README.md`: the badge carries the concept DOI and does not change.
4. This file: move the entries under a dated version heading, and record the new Version DOI
   there once Zenodo has archived the tag.
5. The accompanying article cites the Version DOI of the tag it describes (C1 to C3 of its
   metadata table, the reference list entry and the cover letter), so a new tag means a new
   submission package.

The GitHub to Zenodo switch must be ON for this repository **before** the release is created; a
release created before the switch is never archived and gets no DOI. The DOI only exists after the
release, so `CITATION.cff` and the README badge are filled in a follow up commit, and the archived
copy of a tag therefore never contains its own DOI. That is normal.
