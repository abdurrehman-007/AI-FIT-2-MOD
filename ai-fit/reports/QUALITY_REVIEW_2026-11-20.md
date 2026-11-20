# Recent garment detail review and default correction

After the user reported softer recent green-shirt previews, inspected the
actual latest Forest Green result and its still-present source photo, plus
recent green/cocoa helmet-photo outputs. The latest source and result are
461 x 664; the helmet outputs are 900 x 1600. These are different source
photos, so their appearance cannot isolate model quality. Output dimensions
also do not establish native generated garment detail: FASHN renders at
576 x 864 and DeCo at 512 x 768 before source compositing/resizing.

The speed work also reduced FASHN first candidates from 20 to 12/15 steps,
and DeCo from 50 to 35. Earlier matched probes already showed softer fabric
detail at lower sampling counts. This is a quality trade, not just a cache
optimization. Cache-only matched tests had zero maximum pixel difference for
both renderers; RAM parking is not supported as the cause of this detail loss.
Low available system RAM can still cause paging, UI lag and cold reloads.

## Correction

The launcher now defaults to `AIFIT_SPEED_PROFILE=balanced`: FASHN 20 and
DeCo 50 standard-upper steps. Existing output validation, image resolutions,
source masks, per-request segmentation reuse and bounded/file-backed renderer
parking remain enabled. The optional fast profile still exists, but is no
longer the default. Startup environment changes require a service restart.
Waiting copy now describes approximately 35–60 seconds for detailed previews
rather than promising 20–30 seconds. Long garments and retries can take longer.

No resize/upscale was added to pass a low-resolution input off as true HD.
The latest source remains at its original 461 x 664 output size.

## Matched live check

Using the exact still-present latest source and the same Forest Green
reference, the balanced local `/infer` check completed HTTP 200 in 38.33 s:
FASHN, 20 steps, one candidate, no regional refinement. Private evidence is
in `instance/quality-review-2026-11-20/balanced-recent.json` and its JPEG.
The record includes the loaded code/profile fingerprint. The earlier live
preview was retained; customer inventory and analytics were not modified by
this private comparison.

Full-resolution inspection shows more visible pocket seams, folds and colour
variation than the earlier recent fast preview. Source face/hands are retained.
However, rough/grey sleeve-edge patches remain, especially on the left of the
image. Those are mask/compositing or generation defects and are not resolved
by increasing steps. The restored setting is an improvement in fabric detail,
not proof of a flawless result or HD reconstruction of low-resolution inputs.

The backend code is unchanged from the tested v38 implementation (127 service
tests); this correction changes the launcher profile default, docs and waiting
copy. JavaScript syntax and whitespace checks pass. Full-matrix testing remains
checkpointed; earlier fast-profile manifests are not current balanced evidence.
