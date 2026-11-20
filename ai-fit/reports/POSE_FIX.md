# Pose preservation fix — 23 September 2026

Follow-up: [OUTFIT_FIX.md](OUTFIT_FIX.md) supersedes this report's mask handling for leg-labelled pixels, trouser-colour findings and test count. Historical images/observations below describe the earlier revision.

## Reported defect and corrections

The supplied pink-outfit source has both hands together in front. The earlier brown full-outfit result moved the hands to the sides and changed the leg arrangement. The full-outfit mask explicitly included SCHP exposed-arm classes, removing important source pose evidence. Product foreground removal also retained the donor model's head/skin instead of isolating clothing.

The Windows DeCo-VTON route now uses the detected MediaPipe landmarks in mask construction. Wrist/finger landmarks protect hand regions, while full-outfit exposed arms, legs, face, hair, shoes and existing background remain protected. Full-resolution protected pixels are restored after the generation-resolution roundtrip so small fingers do not disappear. Original clothing remains editable; the approved model, checkpoint, resolution, sampler and safety checker are unchanged.

When a person is confidently detected in the product image, the parser extracts category-relevant clothing and excludes donor head/skin/background before conditioning. Flat-lay/cutout photos keep their existing preprocessing path. This removes donor body cues but does not magically unpose a worn garment or recover occluded fabric.

MediaPipe's person foreground and lower-leg landmarks can repair pale-trouser regions that the clothing parser incorrectly labels background. This repair is restricted to confident foreground near the subject's legs; it does not fill arbitrary scene background. It cannot guarantee exact trouser colour or details.

A final pose check compares reliable source/output joints in the same image coordinates, normalized by torso length. Arms/wrists/ankles use a 0.22 displacement bound; shoulders/hips/knees use 0.28. Lower-joint source visibility/presence must be at least 0.65; other joints at least 0.5. Low-confidence source estimates are not treated as ground truth. These are engineering heuristics, not calibrated accuracy or confidence percentages. Missing reliable output joints or excessive displacement return HTTP 422 with a useful message. The application does not save a rejected image or count it as a successful try-on.

**This is landmark-guided inpainting and compositing, not a newly trained ControlNet/pose adapter.** The DeCo-VTON UNet does not receive a new skeleton channel. Mask anchors give the existing model source-pose evidence; the quality check is additional protection, not the only fix.

## Exact-case verification

Used the original local image supplied by the user, 459×668, and the saved brown and blue full-outfit product photos. Both final requests returned HTTP 200 and passed the pose check: brown **15.75 s**, blue **15.37 s**, after initialization. Visual comparison shows both hands remain together in front and feet remain in the original arrangement, unlike the previous brown result. Generated garment details/drape differ from the product, and visible pink trouser colour remains in the candidates. Consequently this verifies the pose improvement, **not exact full-outfit reconstruction**. No broad accuracy or universal pose guarantee is claimed.

Private comparison: `work/pose-fix/pose-comparison.jpg`. It includes customer photos and is intentionally excluded from the portable source ZIP. No original photos, accounts, inventory or size charts were edited. Tests use temporary databases; finite real-GPU tests do not record live application events.

## Tests and operational notes

- 50 application/sizing/analytics tests passed, including forwarding pose rejection without saving results or inflating successful analytics.
- 33 GPU/preprocessing tests passed: hand masks, full-resolution skin protection, donor extraction, pose displacement, unreliable ankle handling, foreground-constrained leg repair, HTTP 422 and lock release.
- Existing generation UI state regression passed. No frontend code changed.
- Other finite real GPU regression case results are in `pose-fix-validation.json`.

Changed source: `gpu_service/pose_geometry.py` (new), `clothing_mask.py`, `preprocessing.py`, `local_windows.py`. Tests: `gpu_service/test_pose_geometry.py` (new), existing mask/runtime tests and `test_app.py`. Model weights and vendor pipeline unchanged. Module 2 and Module 3 implementation unchanged.

Restart the GPU service to load this code. Stop the prior launcher with Ctrl+C, then run `.\start-gpu.ps1` from the project folder. Previously saved generated images are historical; upload/select the source and generate again. `/health` now reports `pose_preservation: landmark-protected inpainting and output check` for version identification. Existing browser sessions do not update an already-running Python process.

Source ZIPs are refreshed by `scripts/package_module2.py` with CRC/member SHA-256 verification. They include code/tests/report but exclude private diagnostic images and model weights.
