# Full upper-outfit audit: in progress

Requested scope: every person photo under `MODEL PICTURES` (including `Class`,
male, female and root-level photos), against every upper-category inventory
item, across the locally available stores. The 60 outfit files are garment
references, not person-photo inputs. Lower/full-outfit products are outside
this upper-only run.

## Findings and changes before the GPU sweep

- Corrected the locally stored `black pant` product from upper to lower after
  inspecting its photo. The original product record was saved privately.
- Preserve garment silhouette metadata after removing a donor model. This
  prevents a short-sleeved reference from silently becoming an unknown sleeve
  type in mask generation and sleeve validation.
- Detect cropped donor models using visible face/shoulder landmarks rather
  than requiring visible hips. Confirm parser evidence before stripping a
  donor, since flat hoodies can trigger a false pose detection. Preserve the
  original cutout when that confirmation is absent.
- Added a bounded upper torso mask when no existing shirt can be segmented,
  requiring reliable shoulders/hips, independent person foreground agreement
  and compatible parser labels. This addresses the shirtless male sample
  without substituting an unverified rectangle over the photo.
- Added a resumable audit runner with input/code fingerprints, atomic progress
  writes, per-case journaling, private image outputs, CSV and searchable HTML.
  Busy/service errors are distinct from quality rejections and do not count
  as completed successful cases.

## Preflight issues to track

- `Class/duo.jpeg`: two people; the current pipeline accepts one person.
- `F.model.jpg`, `M.model.jpg`, and
  `WhatsApp Image 2025-12-19 at 10.44.48 AM.jpeg`: current source-pose checks
  reject cropped/uncertain hips. They are recorded as unsupported inputs,
  rather than claimed to be corrected.
- `male/male_person_19.png`: initially rejected because it has no existing
  upper garment. The new bounded torso mask is subject to live verification.
- Demo products `Everyday cotton shirt` and `Midnight tailored shirt` have no
  available product image. They are recorded rather than assigned a fabricated
  result.

## Validation and status

Final preflight: 72 person photos, 68 eligible; 32 upper inventory entries,
30 with usable images. This accounts for 2,304 photo/item combinations,
including 2,040 eligible GPU generations and 264 preflight-rejected pairs.
The full resumable GPU sweep has been started; it is not yet complete.

76 service regression tests and 3 audit persistence tests passed. The audit
tests cover resuming without duplicate cases, refusing stale inputs/code, and
not checkpointing server errors as completed generations.

This document does **not** certify completion of the full GPU matrix or visual
review. Current counts, rejected source names, outfit names, error reasons,
timings and output links are in the private audit directory:

- `instance/all-photo-audit/manifest.json`
- `instance/all-photo-audit/progress.json`
- `instance/all-photo-audit/run-state.json`
- `instance/all-photo-audit/results.csv`
- `instance/all-photo-audit/report.html`
- `instance/all-photo-audit/results.jsonl`

`technical_pass_needs_visual_review` means inference and automated structure
checks succeeded; it does not certify exact pattern, sleeve, identity or visual
quality. No guarantee is made for all future photos. Private photos, results,
database records and model weights are not published.

## Running or resuming locally

From `D:\ai-fit\ai-fit`, with the local GPU/web services started:

```powershell
.\.gpu-runtime\Scripts\python.exe -u scripts\audit_upper_tryon.py --run
```

The manifest is frozen for the run; code or input changes require preparing a
new version with `--prepare --output instance/all-photo-audit-vNEXT`, then
running that same output directory. Do not overwrite a completed manifest to
reuse results from different code. To checkpoint-stop after the current case,
create `instance/all-photo-audit/STOP`; remove that file before resuming.
The computer must remain awake for a long local GPU run to progress.
