# Varied-photo reliability checks

Implemented output validation with at most two candidates. A second seed is
used only after a structure/pose/sleeve validation failure. Invalid inputs,
safety rejections, authorization failures and CUDA errors are not retried.
No candidate is returned until the checks pass. The existing input file is
never overwritten by generation.

Additional validation:
- Require a visible face and a usable input person segmentation.
- Fail when the output person boundary cannot be assessed, rather than silently
  skipping body-width validation.
- Compare median silhouette widths over small bands to avoid single-pixel
  segmentation holes causing spurious failures. Require three usable sections.
- Check exposed forearms for targets classified as long-sleeved when reliable
  elbow/wrist landmarks are available. This check has no coverage guarantee for
  occluded joints or garment photos whose sleeve type cannot be determined.
- Keep failed-generation messages visible beside the preview. Show an estimated
  wait of 25–40 seconds, with up to 80 seconds for a second quality attempt.

Verification:
- 66 GPU regression tests and 51 app/sizing/analytics tests passed.
- JavaScript syntax and diff whitespace checks passed.
- Initial live batch used one existing upload and eight synthetic sample photos,
  including indoor/outdoor, varied builds and a seated pose. Eight candidates
  were returned and one seated-pose request was rejected after both attempts.
  Subsequent visual review identified sleeve failures missed by structure checks.
- Final live tests after enabling sleeve validation returned two requests in
  38.78s and 35.51s. The difficult sleeveless-source case was rejected after two
  candidates in 62.69s because both shortened the target's full sleeves.
- SHA-256 comparisons confirmed source files remained unchanged in every test.
- Detailed outcomes and comparisons are private under
  `instance/varied-photo-review/`. The first batch predates the sleeve gate and
  must not be treated as a final design-fidelity acceptance score.

Remaining limits: these checks do not establish exact design reconstruction or
guarantee a usable result for every photo. Visible review still found altered
garment length, sleeve rolls and added small details on some candidates. The
checkpoint was not retrained. Rejection protects against detected failures but
does not prove that every accepted preview is correct. Testing used live local
HTTP/GPU generation; browser automation was not exercised.
