# Dual-model latency work, v38

Subsequent correction: `QUALITY_REVIEW_2026-11-20.md` restores balanced as the
launcher default after the user's recent detail complaint. Fast measurements
below remain historical profile-specific evidence, not the current default.

Both FASHN VTON 1.5 and DeCo-VTON remain available with the existing automatic
selection rules and per-garment calibration. The objective is faster standard
upper previews, not replacement of either renderer or universal 30-second
completion. Long outfits, queue waits, first loads and quality retries can take
longer. Existing neckline, cuff and exact-design limitations are not closed by
this performance work.

## Implementation

- Keep at most one inactive renderer in CPU RAM, while only one occupies CUDA.
  Park the active weights before activating the next renderer. Reuse cached
  weights on return instead of rebuilding the pipeline from disk. Other CPU
  preprocessing/safety components remain on CPU.
- Check available RAM against the estimated additional parking bytes plus a 1.5 GiB reserve before
  parking. If memory is insufficient or the optional `psutil` import is missing,
  use the established discard-and-reload path. Transfer failures restore the
  previous renderer where possible; they do not return a partially loaded image.
- v38 parks FASHN using its original all-BF16 safetensors checkpoint as
  file-backed CPU parameter storage (`assign=True`), with a small wrapper-memory
  estimate. This avoids another approximately 1.9 GB dirty CPU copy and lets
  Windows reclaim clean file pages. DeCo retains synchronized CPU transfers.
  The BF16 parameter values are unchanged; this is not INT8 quantization.
- Enqueue parameter transfers without per-parameter blocking, then synchronize
  CUDA before the weights are used. Cached-to-CUDA switches avoid a redundant
  full garbage collection when both previous and next models remain referenced.
- Fast profile: first standard-upper FASHN candidate uses 12 steps, or 15 when
  source segmentation finds an existing short top that must become long-sleeved.
  Unknown elbow coverage with source clothing is conservative. Bare source
  torsos and already-covered elbows do not trigger this extra-step rule.
- First standard-upper DeCo fast candidate uses 35 steps. Balanced retains
  FASHN 20 / DeCo 50. Long outfits and retries retain FASHN 30 / DeCo 50 in
  either profile. Regional sleeve correction remains separately bounded.
- Keep BF16 precision and existing image resolutions. No quantization, CUDA
  upgrade, new weights, training or validation bypass was introduced.
- Preserve source compositing, output safety and existing pose, neck, body,
  sleeve and applicable garment checks. These checks do not certify exact
  appearance. Fast sampling can soften fabric detail.
- `/health` reports the speed profile and cached backend. `X-AIFIT-Steps`
  records the final main candidate's sampling count; existing attempt,
  refinement, queue and stage timing headers remain available.
- The local launcher now defaults to fast. Direct service execution defaults
  to balanced unless `AIFIT_SPEED_PROFILE=fast` is set. Profile changes require
  restarting the service; running the launcher against an already-ready service
  does not change that process's environment.

## Live measurements and visual review

Local RTX 3060 Laptop GPU, 6 GB, torch 2.5.1+cu118. Inputs use the application's
JPEG preparation. All reported successful probes below used one candidate,
zero queue wait and no regional refinement. Durations cover the local `/infer`
request, not application startup or browser upload time.

| Probe | Version | Seconds | Result |
|---|---|---:|---|
| male19 / Forest Green, FASHN 15 | v35 | 33.53 | Usable collar and sleeves; missed speed target |
| male22 / Forest Green, DeCo 35 first load | v35 | 41.80 | 13.09 s renderer stage; exploratory cold run |
| female02 / Forest Green, return to FASHN 15 | v35 | 35.20 | Warm renderer stage reduced to 3.38 s |
| male19 / Forest Green, FASHN 12 | v36 | 27.20 | Usable neck/pose/sleeves; softer fabric detail |
| male22 / Forest Green, DeCo 35 first load | v36 | 32.88 | 11.06 s renderer stage; not a warm latency claim |
| female02 / Forest Green, FASHN 12 | v36 | 29.88 | **Visual hem regression; rejected as a quality success** |
| male22 / red black shirt, FASHN 12 | v36 | 26.33 | Checks and full sleeves recognizable; cuff edge remains imperfect |
| male22 / Forest Green, DeCo 35 warm return | v36 | 25.42 | Renderer stage 2.53 s; output pixels identical to its cold run |
| female02 / Forest Green, guarded FASHN 15 | v37 | 32.06 | More conservative hem result; known neckline edge remains |
| male22 / Forest Green, DeCo 35 first load | v37 | 32.22 | Includes 10.50 s cold renderer stage |
| male22 / red black shirt, cached FASHN 12 | v37 | 29.09 | Includes 0.92 s renderer stage; checks/full sleeves retained |
| male22 / puffer jacket, cached DeCo 35 | v37 | 31.81 | Renderer stage 1.06 s; heavier segmentation postprocessing; closed jacket/full sleeves inspected |
| male22 / Forest Green, first DeCo activation | v38 | 29.72 | Includes 8.25 s cold renderer stage after file-backed parking |
| male19 / Forest Green, FASHN return | v38 | 31.88 | 5.20 s first mapped-weight return; identical pixels to prior 12-step probe |
| male22 / Forest Green, DeCo return | v38 | 22.25 | 1.72 s renderer stage; both renderers remain available in cache cycle |
| male22 / red black shirt, next FASHN return | v38 | 29.31 | 3.42 s renderer stage; second complete model cycle |

The 12-step female hem regression led to the source-based 15-step rule. It is
not counted as a successful quality result simply because HTTP 200 and the
technical validators passed. The rule is based on garment sleeve metadata,
source clothing segmentation and elbow coverage, not filenames or photo hashes.
An older 30-step female comparison also has a neckline compositing imperfection;
these changes do not claim to fix that existing appearance issue.

The cached and cold v36 DeCo previews have zero maximum JPEG pixel difference.
The v38 DeCo result is also pixel-identical to v37, and the v38 FASHN result
is pixel-identical to its earlier 12-step probe, supporting exact weight reuse
for both named comparisons. The first file-backed FASHN return still took
31.88 seconds, so even with this cache the 20–30 second target is not a hard
upper bound. No successful run is omitted because it exceeded that target.
That supports weight-transfer correctness for this named probe; it does not
prove equivalence for every input or exact fidelity to the garment reference.
Private JPEGs, case identifiers, loaded service fingerprints, step counts and
timing JSON files are in `instance/dual-speed-2026-11-20/`. v35/v36 records are
exploratory history, kept separate from v37/v38 coverage. v37 also exposed RAM
pressure dropping the inactive cache on a later switch; v38's file-backed
FASHN parking addresses the avoidable dirty-copy allocation. Memory limits can
still cause cold reloads rather than unbounded cache growth.

## Verification and operation

127 GPU-service tests and 9 audit tests pass. Added regressions cover bounded
CPU model reuse, CUDA exclusivity, RAM reserve, failed activation rollback,
profile-dependent steps, preservation of long/retry settings, source-based
sampling, exact BF16 file-backed weight restoration/eval state and
profile-sensitive audit fingerprints. JavaScript syntax checked.

For the conservative profile, set `$env:AIFIT_SPEED_PROFILE='balanced'` before
starting/restarting the GPU service. Use the same profile environment for the
audit CLI as the running service. The runtime fingerprint includes the profile;
old manifests must not be resumed under this new code/configuration. The full
matrix audit remains checkpointed to prioritize interactive latency. These
named probes are not a full-matrix accuracy or future-photo guarantee.
