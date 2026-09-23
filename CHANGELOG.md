# Changelog

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
