# Changelog

## 1.0.0, unreleased

First release under this name. Contents:

* Protocol scripts for phases A to G, the four frozen embedding controls and the conversion
  fitness rubric (`protocol/`).
* Frozen annotations: the bounding box manifest of 8811 crops, the near duplicate leakage list,
  the flagged duplicate label pairs, the merge decisions, and the record of ten boxes corrected
  before the first public release (`protocol/frozen/`, see `KNOWN_ISSUES.md` section 1).
* Both variants of the Phase F mask, the integrity gated one and the earlier ungated one, so the
  comparison in the accompanying article is reproducible.
* Expected outputs for checking a local run, including the third probe reports and the 27
  backbone baseline scores, and `scripts/verify.sh` to compare against them field by field.
* `REPRODUCE.md` (step by step, with the list of what the protocol writes and which files are
  the benchmark), `TRAINING_RECIPE.md` (the recipe behind the baseline scores; the sweep itself
  is not part of the protocol) and `KNOWN_ISSUES.md`.
* Contains no image.

Provenance: this repository supersedes the release lineage previously published under the name
`cure-fewshot-protocol`. Code and annotations are identical to that lineage's last version; the
documentation was rewritten and the version numbering restarts at 1.0.0.

### Procedure for cutting a tag

1. `CITATION.cff`: `version`; then `doi` and `date-released` once Zenodo has archived the tag.
2. `.zenodo.json`: `version`.
3. `README.md`: the DOI badge.
4. This file: move the entries under a dated version heading.

The GitHub to Zenodo switch must be ON for this repository **before** the release is created; a
release created before the switch is never archived and gets no DOI. The DOI only exists after the
release, so `CITATION.cff` and the README badge are filled in a follow up commit, and the archived
copy of a tag therefore never contains its own DOI. That is normal.
