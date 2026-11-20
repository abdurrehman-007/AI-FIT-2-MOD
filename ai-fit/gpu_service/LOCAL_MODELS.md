# Local inference engines

The Windows service selects FASHN for upper garments when its pinned source,
dependencies and weights are installed. Otherwise `auto` selects DeCo. Lower
and full outfits retain DeCo when installed; FASHN's upper repairs do not
silently replace their established renderer. `AIFIT_ENGINE` accepts
`auto`, `fashn`, `deco` or `catvton` for an explicit selection.

From the application directory:

```powershell
.\.gpu-runtime\Scripts\python.exe gpu_service\setup_fashn.py
.\.gpu-runtime\Scripts\python.exe gpu_service\setup_fashn.py --check
.\.gpu-runtime\Scripts\python.exe -u scripts\run_local.py
```

The setup script pins the [upstream FASHN source](https://github.com/fashn-AI/fashn-vton-1.5)
at `7c0f10af3f91ad4048fe9729c470a13ef905d25a`, checks SHA256 hashes for all
weights, and installs `fashn-human-parser==0.1.1` and `einops==0.8.1` into the
ignored `.runtime-deps/fashn` directory. It uses the existing CUDA, PyTorch,
CPU ONNX, transformers and image-processing environment. Model files remain
under ignored `.models/fashn`; upstream code remains under `.external`.

FASHN runs guidance branches sequentially, with DWPose, human parsing and the
safety checker on CPU. A local RTX 3060 Laptop GPU (6 GB) completed the reference
probes with approximately 2.3 GB peak PyTorch allocations; that metric excludes
CUDA/desktop overhead. CUDA bfloat16 support is required by this adapter.

FASHN prefers native cuDNN SDPA for supported CUDA attention shapes, retaining
the original dispatcher for unsupported/allocation-failed cases. Set
`AIFIT_FASHN_ATTENTION=original` before restart to disable that preference.
The model, precision, resolution and step count stay unchanged. This setting
is included in the runtime fingerprint; floating-point ordering can differ.

The launcher now defaults to `AIFIT_SPEED_PROFILE=balanced`, retaining 20
FASHN steps and 50 DeCo steps for standard upper candidates. Opt-in `fast`
uses 12 FASHN steps, or 15 when an existing short top must become long-sleeved
(unknown elbow coverage is conservative). These local fast settings are below upstream's recommended
20-step fast profile and can soften fabric detail. Long tops, other categories
and quality retries retain 30 FASHN steps. DeCo uses 35 steps for the first
standard-upper fast candidate and 50 otherwise. FASHN uses guidance 2.0,
seed 73 and the mask-free path. A quality retry uses seed 42, masked
conditioning and guidance 3.0.
Fast sampling skips guidance in the last three steps using the upstream
sampling option; other sampling retains the original one-step setting.
The sequential CFG adapter avoids the discarded unconditional forward in
those final steps. All source protection and output validators remain active.
When source parsing confirms both a coat and inner shirt, masked conditioning
with guidance 3.0 is selected from the first candidate. This helps replace the
source layers instead of copying their open-front style onto a closed reference.
Failed sleeve length can trigger one bounded regional correction. The service
still composites through source protection masks and checks output pose,
silhouette, sleeve length and applicable closed-reference contrast constraints.
Safety-blocked or quality-rejected images are not returned as successful previews.

In automatic mode, persistent FASHN sleeve failure can use DeCo for the second
candidate. The same validation and safety checks apply. The old renderer is
moved off CUDA first, keeping one renderer on the 6 GB GPU. At most one
inactive renderer is retained on CPU when available RAM exceeds the estimated
additional parking allocation plus a 1.5 GiB reserve; otherwise it is discarded.
FASHN rebinds its unchanged original BF16 weights through file-backed CPU
tensors instead of allocating a second dirty host copy. DeCo moves its weights
to CPU. Warm switching
reuses those weights. Nonblocking transfers are synchronized before moved
weights can be used. First use still adds loading time. Explicit engine selection
disables this automatic fallback.

Optional local calibration lives in ignored `instance/garment-backends.json`.
Its `garments` map uses SHA256 of the EXIF-normalised RGBA reference's size and
pixels, with an `engine` value of `deco` or `fashn`. A different reference image
does not inherit the choice. Calibration selects an engine before generation;
safety rejection never triggers an alternate-engine retry.

Small invented contrasting emblems can be repaired on confidently plain white
or dark neutral references. Reference contrast disables this repair except for
the centre closure; protected pixels and centre zips/buttons remain excluded.
This conservative correction does not certify arbitrary printed designs.

Upper-only cropped portraits are supported when the face/shoulders are strong,
enough torso is visible, and person foreground continues through the lower
image boundary. Geometry checks use that visible boundary, never extrapolated
hip coordinates. Lower/full outfits still require their visible body anchors.
For full sleeves, recognised source arms reaching the cropped boundary can
change without fabricating wrist locations. Editable new sleeves are excluded
from old arm-skin restoration, preventing skin-coloured holes when dark cloth
is mislabelled as skin. Actual hands and neck retain source protection.

Worn short-sleeved references retain an explicit sleeve hint when substantial
exposed donor arms contradict the cropped shirt silhouette. Hair at the head
and long-hair cores stay protected; a narrow fringe beside clothing below the
shoulders may be inpainted to avoid copying old-shirt colour at the hair edge.

The private audit checks `/health`'s loaded-code/configuration fingerprint
against its manifest before every inference request. Changing source files,
engine configuration or local garment calibration requires a matching service
restart/new manifest. Diagonal audit rounds cover different people/styles
early, then complete every inventory pair; technical success still requires
separate visual review.

Retailer inventory supports `upper_length: standard | long`, including the
optional CSV column and the Top length selector. Long upper garments receive
a bounded waist-to-knee overlay using observed foreground. Source hands,
scene and the lower part of the trousers stay protected. Long FASHN references
use masked one-piece conditioning from the first candidate, while output
composition/validation retain the upper category. Generated trousers below
the new garment are replaced with the source layers. A length check refuses
a long reference that was shortened to the waist.

Colour-aware closed-front checks permit lighter fabric with the reference's
hue; a brightness change alone is not proof of an invented neutral underlayer.
For cropped portraits, small shoulder corrections anchor to the visible lower
frame rather than unavailable hips; the final neck/pose validator is unchanged.

Background audits yield to waiting interactive requests. Up to two shoppers
can wait for a slot for 90 seconds; an active inference cannot be interrupted.
The web timeout includes this waiting time. `/health` exposes `busy` and
`queued` for local diagnostics; these do not certify generation quality.

Use `/health` to check the active engine and resolution. `CATVTON_URL` and
`CATVTON_API_KEY` remain the web application's transport settings for this
private service, including when the selected engine is FASHN.

The release weights were not trained or fine-tuned by these changes. Generative
output is approximate: printed details, closures, fit and occluded anatomy can
still be imperfect. A structure pass is not a garment-fidelity certificate.
The complete person-by-inventory audit is separate from targeted verification.

In automatic mode, standard plain coloured long-sleeve references can use DeCo
when segmentation confirms clothing near both source elbows. Short/bare source
arms, neutral or contrasting references and long tops keep the prior policy.
Explicit engine settings and per-garment calibration take precedence. This
conservative choice follows a visually reviewed Forest Green Twill Shirt
regression; it is not proof of exact reference design preservation.

Segmentation results are reused only within one admitted request, keyed by
pixels and bounded to eight entries, then cleared on exit. Private inference
stage measurements are returned in `X-AIFIT-Timings`; queue waiting is separate.
The v34 targeted standard-shirt checks took approximately 38–40 seconds with
no queue. Long garments, model switches and retries can take longer. Production
remains BF16: the tested native INT8 path was slower on this GPU. See
`reports/SPEED_OPTIMIZATION_2026-11-19.md` for measured scope and limitations.

The subsequent dual-model fast work is in
`reports/DUAL_MODEL_SPEED_2026-11-20.md`. Change profile before starting or
restarting the GPU process, for example `$env:AIFIT_SPEED_PROFILE='balanced'`.
Both direct service execution and the launcher default to balanced following
the recent appearance review. The audit CLI must use the same profile environment as the running
service when preparing and generating a new manifest; the fingerprint includes
this setting. Old matrix manifests cannot be resumed under the new code/profile.

The fitting room's processing timeline uses authenticated, owner-scoped polling
and actual sampler-step callbacks. Percentage and ETA are explicitly estimated;
queues and extra refinement can extend them. Only a verified, saved result
reaches 100%. See `reports/PROCESSING_TIMELINE_2026-11-20.md` for live browser
measurements, negative cases and the preserved-quality latency limits.
