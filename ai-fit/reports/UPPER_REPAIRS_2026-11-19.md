# Upper try-on repairs and targeted verification

**Follow-up:** [remaining work and current evidence](REMAINING_UPPER_WORK_2026-11-19.md)
tracks newer cropped-photo/sleeve fixes, fresh preflight and unresolved cases.
The version-specific results below remain an earlier snapshot.

The full photo-by-inventory sweep is checkpointed while known visual failures
are repaired. This report covers targeted examples, not completion of all
2,304 combinations or a measured overall accuracy percentage.

## Changes

- FASHN is available locally for upper garment detail and collar/sleeve rendering.
  It runs with sequential guidance on the existing 6 GB CUDA GPU. No model was
  trained or fine-tuned. Setup, pinned dependencies and weight verification are
  documented in [LOCAL_MODELS.md](../gpu_service/LOCAL_MODELS.md).
- Source face, hands, scene and clothing outside the editable region remain
  protected. Visible mouth/nose anchors support neck alignment, including the
  helmet sample. Output pose and silhouette checks still apply.
- Short upper references no longer inherit ankle-length geometry from the
  source dress. Arm corridors remain editable when cuffs extend below the hips.
- Sleeve validation checks both directions: long sleeves becoming short and
  short sleeves becoming long. One regional repair is allowed; persistent
  FASHN sleeve failure can use a validated DeCo candidate in automatic mode.
- Background cleanup requires independent output foreground agreement, avoiding
  treating a white shirt as background while correcting shoulder halos.
- Plain neutral references have a conservative small-emblem correction. Actual
  reference print disables correction; centre closures and protected source
  pixels are excluded. Cutout antialias edges do not count as reference prints.
- The local puffer reference selects DeCo before inference through an exact
  image-pixel calibration. Safety checks run on both engines. Safety rejection
  does not trigger a retry through another engine.

## Named examples

| Person photo | Reference | Targeted observation and evidence |
|---|---|---|
| `male/male_person_19.png` | Forest Green Twill Shirt | v13: collar, closed front and full sleeves restored on shirtless source; visually reviewed |
| `female/female_person_02.png` | Forest Green Twill Shirt | v13: closed shirt with full sleeves; visually reviewed |
| `female/female_person_01.png` | Forest Green Twill Shirt | v13: short shirt with original lower skirt preserved; visually reviewed |
| `Class/Addan 2.jpeg` | Forest Green Twill Shirt | v13: shoulder halo corrected; visually reviewed |
| `Class/Ali Ammar.jpeg` | Forest Green Twill Shirt | v13: successful control, original body/pose retained |
| `Class/mohib.jpeg` | Forest Green Twill Shirt | v14: raised-arm full sleeve passes after automatic DeCo fallback; visually reviewed; 140.16 s |
| `ABD.jpeg` | Plain white T-shirt | v15: short sleeves retained, no invented chest logo; visually reviewed; 71.12 s including engine switch |
| `ABD.jpeg` | Shadow Black Hooded Puffer Jacket | v15: fit/quilting retained and invented chest logo removed; visually reviewed; 42.61 s including engine switch |

Private before/product/after comparisons and request journals are in
`instance/upper-repair-v13/`, `instance/upper-repair-v14/` and
`instance/upper-repair-v15/`; nonupper regression results are in
`instance/upper-repair-v16/`. Results from different code versions remain
separate. v15 changes the neutral-reference emblem repair; the green-shirt
rendering and fallback paths retain the v13/v14 behavior above. v16 changes
automatic nonupper selection, leaving the verified upper path unchanged.

## Verification

- 97 GPU-service regression tests passed without loading generation weights.
- 51 application, analytics and sizing tests passed.
- 5 audit checkpoint/persistence tests passed.
- JavaScript syntax check passed.
- Final v15 puffer and T-shirt live verification passed and was visually reviewed.
- Normal web-route upload, authenticated `/api/tryon`, result storage and media
  retrieval passed using the live GPU (38.12 seconds). This verifies integration,
  not exact design preservation on every source photo.
- A nonupper regression probe exposed cropped jeans with automatic FASHN.
  v16 restricts the new default to upper; lower/full outfits retain DeCo.
  v16 jeans completed in 31.48 seconds and were visually reviewed at ankle
  length. `female/female_person_06.png` with `Test Female 11 - Bodycon Dress`
  remained safety-rejected (24.00 seconds); no candidate was returned or saved.
  This is a named unresolved generation limit, not a successful visual test.
  The earlier FASHN dress probe was also safety-rejected; neither rejection
  triggered an alternate-model retry in its request.

Normal FASHN upper examples took approximately 60–67 seconds locally. The
raised-arm fallback took 140 seconds. Engine loading, a quality retry and GPU
contention add time; these are measurements on this laptop, not guaranteed
latencies.

## Remaining coverage and input limits

The old v10 sweep checkpoint contains 30 finished pairs, including technical
passes that still require visual review. Its renderer fingerprint is stale
after these repairs. Do not resume it with changed inference code: prepare a
new versioned audit directory and retain the old evidence.

Preflight accounts for 72 person photos and 32 upper records (30 real images).
`Class/duo.jpeg` contains two people. `F.model.jpg`, `M.model.jpg` and
`WhatsApp Image 2025-12-19 at 10.44.48 AM.jpeg` fail the current visible-hips
check. These remain named input rejections. Demo records `Everyday cotton
shirt` and `Midnight tailored shirt` have no real garment photo.

An automatic structure pass does not prove exact print, seam, fit or anatomy
preservation. Arbitrary future-photo correctness and the complete eligible
matrix remain unverified. Rejected candidates are not displayed as successes.
