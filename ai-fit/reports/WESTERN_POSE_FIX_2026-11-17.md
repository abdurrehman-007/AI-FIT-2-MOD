# Western clothing and local GPU connection

The deployed local service reports `preprocessing_version=garment-aware-mask-v2`.
The web app at http://127.0.0.1:5050 reports `tryon_ready=true`, and the DeCo
service on port 8001 reports `ready=true`.

## Changes

- Validate shoulder/torso proportions in pixel coordinates, accounting for the
  source image aspect ratio. The existing normalized landmarks still map back
  to the source coordinates.
- Trim product-photo margins using the foreground alpha mask and preserve its
  silhouette for category-specific masking. Small product images are no longer
  upscaled before segmentation.
- Bound short-sleeved targets to the upper arm, preserving existing exposed
  forearms. Short lower garments preserve existing leg skin.
- Keep the complete editable region inside inference crops. When aspect-ratio
  padding is necessary, pad with white and restore through the same crop box.
- Retain full-body context for long-sleeved tops when source ankles and torso
  landmarks are reliable. In the denim-jacket comparison, the previous tight
  torso crop repeatedly produced short sleeves. Full context produced longer
  sleeves, although they still do not exactly match the product.
- Use guidance 3.5 for upper garments, retaining 2.5 for lower/full outfits.
  Experimental full-body crops for lower garments produced worse results and
  were not enabled.
- Use Windows file locks during launcher startup and for the GPU service's
  lifetime. Duplicate services had consumed GPU memory. A second service launch
  now exits with an already-running message before model loading.
- Restart the web app with the GPU URL/key supplied by the launcher, resolving
  the missing-service error shown in the screenshot.

## Verification

- 49 GPU regression tests and 51 app/sizing/analytics tests passed.
- Four final real GPU cases (male jacket, female top, male jeans, female shorts)
  returned HTTP 200 and passed the existing output-pose check, in 10.61–14.07s.
- The exact most-recent uploaded photo from the screenshot was exercised through
  `/api/tryon` using its existing owner. `TEST-MALE-14` returned HTTP 200 in
  14.98s and `TEST-MALE-26` in 11.77s. Their successful try-on events/results were
  stored by the normal route.
- Duplicate-service lock behavior was checked against the running service.
- Private comparisons, raw trial outputs and timings remain under
  `instance/western-pose-fix/`.

## Remaining limitations

This updates preprocessing, inference settings and startup code. No checkpoint
training was performed. Passing the pose check does not establish exact garment
reconstruction. Visual review still finds sleeve-length errors, neck/garment
boundary seams, and incorrect jeans details. Those cases need further model and
dataset evaluation. The synthetic sample set has no independently labelled
landmark ground truth, so no landmark accuracy percentage is claimed.
