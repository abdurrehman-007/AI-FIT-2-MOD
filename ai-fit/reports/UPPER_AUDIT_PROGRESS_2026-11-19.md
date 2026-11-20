# Full photo × upper-inventory audit — checkpointed snapshot

**Status update:** the sweep below was checkpoint-stopped to repair its known
failures. See [current repairs and verification](UPPER_REPAIRS_2026-11-19.md).
The findings below describe the earlier v10 snapshot; several now have passing
targeted replacements. Its inference fingerprint is stale after the repairs.

Preflight covered all 72 person photos under `MODEL PICTURES`, including male,
female, Class and root files. After correcting `black pant` to lower, there
are 32 upper inventory records across local stores. Thirty have usable images.
Sixty files in the outfits subfolders are garment references, not person inputs.

The matrix accounts for 2,304 pairs: 2,040 eligible GPU generations and 264
preflight rejections. Completion and visual correctness are **not** certified.

## Findings by filename

| Photo | Observed issue | Current disposition |
|---|---|---|
| `Class/duo.jpeg` | Two people | Unsupported by single-person pipeline; named rejection |
| `F.model.jpg` | Cropped/uncertain hips | Named input rejection |
| `M.model.jpg` | Cropped/uncertain hips | Named input rejection |
| `WhatsApp Image 2025-12-19 at 10.44.48 AM.jpeg` | Cropped/uncertain hips | Named input rejection |
| `male/male_person_19.png` | Shirtless torso/arms not recognised as replaceable clothing; full sleeves shortened | Torso/arm repair and focused crop added; v10 live retest still rolls sleeves. Unresolved and rejected by quality gate |
| `Class/Addan 2.jpeg` | Grey halo around shoulders on Forest Green Twill Shirt despite technical pass | v10 output is byte-identical to the inspected v8 result; remains unresolved |
| `Class/mohib.jpeg` | Forest Green Twill Shirt fails neck/shoulder validation | Rejected after two candidates; private diagnostic retained for review |
| `female/female_person_01.png` | Technical pass but the short Forest Green shirt becomes ankle-length, following the source kimono | Visually reviewed and marked unresolved; automated structure checks do not catch this style/length failure |
| `female/female_person_02.png` | Forest Green Twill Shirt sleeves are shortened | Rejected after two candidates; unresolved |

The demo products `Everyday cotton shirt` and `Midnight tailored shirt` have
no available garment image and are recorded as inventory rejections.

## Changes and verification

- Preserve sleeve silhouette after donor removal; detect cropped donors.
- Require additional parser evidence to avoid cutting a flat hoodie when pose
  detection hallucinates a model. Retain its existing cutout otherwise.
- Add bounded shirtless torso and arm repair using reliable pose and independent
  foreground segmentation. Keep source face, hands, pants and scene protected.
- Private opt-in diagnostics retain rejected *structure* candidates for audit
  review. Safety-blocked outputs are not retained by this diagnostic path.
- Checkpoint runner records every pair separately, preserves source/code hashes,
  distinguishes technical success from visual review, and recovers an incomplete
  final journal write. JSON/image writes are flushed before recording completion.
- 78 service regression tests and 5 audit persistence tests passed.

The first v8 run was interrupted after three cases: two technical passes and one
sleeve rejection. Those results remain separate from the corrected v9 run.

Latest private results and run state are under
`instance/all-photo-audit-v10/`: `report.html`, `results.csv`, `results.jsonl`,
`progress.json`, `run-state.json`, and `visual-issues.json`. Earlier diagnostic
comparisons remain under `instance/all-photo-audit/`.

v9 was checkpointed after three cases. v10 reruns the full GPU matrix with the
focused shirtless crop. It reuses v9's completed preflight classifications
because eligibility checks did not change, verifies every input hash, and
records that provenance separately from the new inference code fingerprint.

The full sweep is a long local GPU job. The computer must remain awake.
An automated structure pass is not a visual quality or exact-design guarantee;
future arbitrary-photo correctness remains unproven. Further issue fixing and
visual review are still required after generation coverage is complete.

The old snapshot can only resume with its matching inference code. With the
current repaired service, prepare a fresh versioned audit from
`D:\ai-fit\ai-fit` instead; keep the old evidence separate:

```powershell
.\.gpu-runtime\Scripts\python.exe -u scripts\audit_upper_tryon.py --prepare --output instance\all-photo-audit-v15
```
