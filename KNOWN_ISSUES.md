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

One provenance nuance is recorded here because it is the kind of detail this protocol exists to
surface. The 27 backbone baseline sweep ran on a machine that had rebuilt its crops from the
manifest, so it saw the ten pre-correction crops: eight among its 7016 training images and two
among its 1715 evaluation queries. The gallery was untouched, since all ten are consumer
photographs and Phase F donor windows are cut from raw photographs. Even if both affected queries
flipped rank, a backbone's top1 would move by at most 0.0012, three orders below the reported
0.55 to 0.84 spread, so no conclusion drawn from the sweep changes. It is noted so that anyone
rerunning the sweep from the corrected manifest knows why ten crops differ.

---

## 2. Five hardened reference masks fall back to an inscribed ellipse

**Severity: low, reported by design.**

Phase F resolves the foreground mask by trying colour distance thresholds in ascending order and
accepting the first whose mask has a plausible area and a solidity of at least 0.97. For five of
the 388 reference images no threshold reaches that, and not even the convex hull repair does, so
the mask falls back to an inscribed ellipse. Those five carry a thin ring of the original synthetic
field around the tablet.

They are recorded as `suspect` in `outputs_v2/hardened_v2_log.json` rather than silently accepted.
This is discussed in the accompanying article. It is a property of those five images, not
a bug to fix.

---

## 3. Pixel identical crops require Pillow 12.0.0

**Severity: cosmetic, documented.**

Letterboxing resamples with a LANCZOS kernel, so the padding boundary can differ by a few pixels
between Pillow versions. `requirements.txt` pins 12.0.0 for that reason. A different version
reproduces the same boxes and the same image content; every split file is unaffected, because
splits depend only on the manifest. `make_crops.py verify` notices, by design.

One stage is measurably sensitive to the exact pixels: the Phase F threshold search. Rerun on a
second machine with a different imaging stack (5 August 2026), it resolved borderline masks
differently, thirteen ellipse fallbacks rather than five, with the same accepted solidity minimum
of 0.971. The integrity gate flagged every unresolved case as suspect rather than passing it, and
in the third probe rerun performed on that machine every rubric criterion still landed in the same
band. So the statistics of the mask search vary with library versions; the decisions gated on it
did not, in the one rerun observed.

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
