# Neck and sleeve cuff correction

The recent Forest Green Twill Shirt previews lowered the shoulders slightly,
making the neck appear longer, and retained parts of the source white cuffs.

## Changes

- Bounded vertical alignment brings a small generated shoulder drop back to
  the source shoulder height. Mouth and hip anchors remain fixed. Large or
  uncertain changes are not corrected and remain subject to rejection.
- The original edit mask and full-resolution protected pixels are reapplied
  after alignment, followed by skin restoration and output validation.
- Neck extension checking is stricter: the tolerated increase is the larger
  of 2.5% of torso scale or 12% of the original mouth-to-shoulder distance.
- Upper-body hand protection no longer restores recognised source cuff fabric.
  For full sleeves, each reliable wrist guard stops on the hand side of the
  wrist plane, preventing cuffs misclassified as skin from being retained.
  Short-sleeve forearm preservation remains covered by regression tests.

## Verification

- 72 GPU service unit/regression tests passed, including protected-pixel
  preservation after alignment, subtle neck extension, wrist boundaries,
  old cuff removal and short/long sleeve behaviour.
- Two real GPU generations through the app's authenticated `/api/tryon` route
  succeeded with the Forest Green Twill Shirt: a white formal shirt source
  and a black kurta source. Final runs took 28.94 and 27.16 seconds.
- Both passed the live pose, body silhouette and full-sleeve checks. Visual
  comparisons showed improved neck proportions and removal of the white cuff
  remnants in the formal-shirt example.
- Estimated mouth-to-shoulder distance ratios (output/source) changed from
  1.184 to 1.023 and from 1.160 to 1.013. These are pose-landmark proxies,
  not anatomical measurements or calibrated accuracy scores.
- Web and GPU services are running locally; GPU health reports
  `neck-cuff-v7`. Whitespace checks passed for the changed source files.

This is an inference and compositing fix, not checkpoint training. Verification
covers these two examples and the regression suite; exact garment detail and
arbitrary-photo correctness are not guaranteed. Comparisons and source images
remain private under the ignored `instance/neck-sleeve-review` folder.
