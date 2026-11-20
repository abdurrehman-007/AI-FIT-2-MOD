# Garment detail, neck and body-shape checks

## Implemented

- DeCo inference now runs at 512 x 768 instead of 384 x 512. The safety checker
  runs on CPU in float32 to leave GPU memory for the larger denoising frame.
  Safety checking remains enabled for every generated candidate.
- Original neck skin is protected inside a landmark-bounded area, including
  skin which the parser incorrectly labels as an arm.
- Output validation checks mouth-to-shoulder elongation as well as joint
  displacement. Excessive elongation is rejected before saving the result.
- Torso silhouette widths are checked at four relative heights using the
  source/output person segmentation. Two sections shrinking below 82% of the
  source width cause rejection. This is a conservative image-quality check,
  not a calibrated measurement of the person's body or physical garment fit.
- Full-body upper-garment context is retained when reliable head, torso and
  ankle landmarks are available. Short-sleeve detection uses the outer 10%
  of the garment silhouette instead of confusing the torso sides with sleeves.

## Verification

- 54 GPU regression tests and 51 application/sizing/analytics tests passed.
- Live GPU inference ran on the existing RTX 3060 with the larger resolution.
  Observed GPU allocation was approximately 4.9 GB within 6 GB VRAM.
- The screenshot's uploaded person image was tested through the normal
  application `/api/tryon` route with two existing inventory garments.
  Both returned HTTP 200 after the new checks were enabled, in 30.17s and
  28.15s. The normal route saved their results and successful try-on events.
- Patterned clothing and fuller-body source photos were also exercised through
  the running GPU endpoint. A distorted polo candidate was rejected with 422
  rather than returned as a valid preview. Final candidate outcomes are in
  the private `instance/western-pose-fix/v3-design-and-build.json` report.
- Comparisons and source/output shape diagnostics remain in the private
  `instance/western-pose-fix/` directory. Testing used live HTTP/API generation,
  not browser automation. Browser URL verification was unavailable previously.

## Limits observed during review

The patterned outfit retains its main color and trim more clearly, but the
checkpoint still redraws some fasteners, pocket details and sleeve shapes.
Exact garment/design preservation is not solved by these changes. Neck and
body-shape checks prevent certain distorted candidates from being delivered,
and can reject a request rather than producing a usable preview. No model
weights were retrained and no claim of physical fitting accuracy is made.
