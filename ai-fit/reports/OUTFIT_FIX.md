# Full-outfit artifact correction — 24 September 2026

This follows `POSE_FIX.md`. The prior result preserved hands but retained pink trousers and clothing fragments; pose success did not mean successful garment transfer.

## Changes

The full-outfit mask now edits SCHP leg classes as well as clothing. The parser had labelled some pale trousers as skin; protecting those fragments left old pink colour inside the conditioning image, influencing the generated outfit. Hands/arms, face/hair and shoes remain protected. Changing a full outfit may cover previously exposed legs; full leg-skin preservation is deliberately no longer promised.

Silhouette-edge repair uses the existing narrow mask dilation only where independent person foreground probability exceeds 0.5. This allows missed pale fabric at shoulders/edges to change without turning arbitrary background into an editable rectangle. Existing foreground/leg-corridor repair and full-resolution hand restoration remain active.

Footwear protection expands semantic shoe regions slightly and uses reliable heel/toe anchors where available. It excludes trouser legs. Pose validation still rejects excessive reliable joint movement; protected pixels are restored at original resolution before JPEG encoding. Pose thresholds were not loosened for this change.

No repainting, image-generator substitution, new model weights, model training, or claim of exact garment reconstruction is involved. The approved DeCo-VTON backend and its safety checker remain unchanged. Module 2/3 application behaviour and existing user records are unchanged.

## Verification and limitations

50 application/sizing/analytics tests and 37 GPU/preprocessing tests passed: **87 Python tests**. Generation-state JavaScript checks passed. New tests cover editable leg-labelled fabric, footwear preservation, foreground-limited edge repair, and unreliable foot landmarks. The final real-GPU results and visual acceptance findings are recorded in `outfit-fix-validation.json`.

The local exact-case outputs are `work/outfit-fix/pink-blue.jpg` and `pink-brown.jpg`. The reported image is the user's `Downloads/ai-fit (5).jpg`; the original input remains unchanged. Private photos/diagnostics are excluded from archives. This update addresses old-colour retention and mask boundaries; embroidery, drape, sleeve/hem cut and precise product colour remain generative-model limitations. Passing tests/pose checks does not establish visual fidelity across all outfits.

Restart the existing GPU launcher (`Ctrl+C`, then `.\start-gpu.ps1` from the project folder), reload the website, and generate again. Old saved outputs do not change. Current complete source ZIPs are rebuilt with CRC and member SHA-256 verification; installed environments, private instance data and weights remain local.


## Final visual acceptance

Both exact-case requests completed with the final source: brown 17.11 seconds, blue 16.25 seconds, HTTP 200. A four-column comparison (original, reported defective result, new blue, new brown) was visually inspected. The old pink trousers are replaced and both hands remain together. Garment edges are improved. However embroidery, silhouette/drape, exact colour and small ankle/cuff/footwear-boundary details remain imperfect. This is **partial visual improvement**, not completion of exact product reconstruction. Further quality improvement needs a separately evaluated stronger conditioning/model approach and representative full-outfit evidence; software test counts cannot establish it.

The source/test/report work and release packaging are complete for this correction. No running customer server was restarted or live database rewritten. Current files: `gpu_service/clothing_mask.py`, `gpu_service/pose_geometry.py`, their tests, README and reports. The last executable changes preserve the approved backend.
