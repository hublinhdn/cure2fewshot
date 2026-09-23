# Changelog

## 1.0.0, 2026-09-23

First public release. Zenodo Version DOI: recorded in the follow up commit once Zenodo has
archived the tag; the concept DOI, which always resolves to the latest version, goes into
`CITATION.cff` and the README badge at the same time.

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

The GitHub to Zenodo switch must be ON for this repository **before** the release is created; a
release created before the switch is never archived and gets no DOI. The DOI only exists after the
release, so `CITATION.cff` and the README badge are filled in a follow up commit, and the archived
copy of a tag therefore never contains its own DOI. That is normal.
