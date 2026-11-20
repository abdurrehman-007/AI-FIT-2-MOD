# Remaining upper try-on work — in progress

**Current status:** v29 remains checkpointed after four pairs, with three
technical passes and one generation rejection. Its shirtless-source collar
cutout is a historical visual failure. v30 expands the verified bare collar
mask while protecting the original face/upper neck. The focused live retest
passed in 70.05 seconds and was inspected at full resolution; the skin wedges
over the new collar are gone. Fresh v30 preflight completed all 72 photos and
32 products; its matching-code matrix is now running under
`instance/all-photo-audit-v30/`. Read that directory's progress/run-state for
actual completion; generation and visual review are not finished.

The earlier 80% figure was a rough opinion, not measured accuracy or completed
generation coverage. This follow-up tracks actual defects and evidence.

## Fixes implemented

- Upper-only cropped portraits use strong visible face/shoulder evidence,
  enough visible torso and foreground at the lower image boundary. Geometry
  uses the frame boundary rather than extrapolated hip locations. Lower and
  full outfits retain their complete-pose requirements.
- Cropped visible arms can receive full sleeves even when wrists are outside
  the picture. Sleeve checks inspect those source arm regions too.
- Dark new cloth misclassified as skin no longer causes the old bare arm to
  be pasted into an editable long sleeve. Actual protected hands/neck remain.
- Worn T-shirt/polo references use substantial exposed donor arms as sleeve
  evidence. The worn `blue shirt`, `body black` and `SHIRT` references now
  classify as short sleeves in preflight; flat hoodies remain long-sleeved.
- Confirmed coat-plus-shirt sources begin with masked FASHN conditioning to
  replace the old layered/open style. Closed coloured references also reject
  darker grey underlayers, not only bright white ones.
- Hair protection permits a small old-shirt fringe to change while preserving
  head hair/core. A conservative brown-hair colour rule addresses light coat
  contamination; fine hair boundaries still need visual review.
- A cropped-pose quality failure can try DeCo as its second candidate. All
  structure/safety checks remain; safety rejection never triggers this fallback.
- Generated-pose failures now describe the generated image, instead of falsely
  blaming a source face that passed input validation.
- Cropped shoulder alignment anchors to the visible lower frame; its bounded
  correction handles the seated portrait without guessing hip coordinates.
  Final neck/pose checks retain their original strict thresholds.
- Inventory/API/CSV now support `upper_length: standard | long`. Four inspected
  long references (Modern Kurta, Sherwani, Chapan and Trench Coat) were tagged;
  original records are backed up privately. Long tops can cover the waist and
  upper trousers through a bounded observed-foreground region instead of
  stopping at the old shirt hem. Shoppers' hands remain source pixels.
- Long references use masked one-piece conditioning with upper-only composition.
  Shortened long garments are rejected. A second independent human parser
  confirms trousers before restoring source lower layers: pale floral cloth
  misclassified by SCHP cannot be erased by that single prediction.
- Closed-front colour checks compare hue as well as brightness/saturation.
  Lighter blue folds are permitted while neutral invented underlayers remain
  checked. Ordinary colour/lighting variation alone does not trigger rejection.
- Interactive shoppers have bounded waiting and priority over the background
  audit. Two shoppers can wait up to 90 seconds; extra/expired requests get a
  busy response. The app allows 360 seconds for waiting plus generation, and
  displays the longer possible wait.
- Every audit request checks the running service's loaded-code/configuration
  fingerprint. Diagonal rounds expose different people and garment styles
  early, then cover every pair. CSV/JSON retain backend and refinement details.

## Evidence and outstanding cases

| Photo/reference | Actual observation |
|---|---|
| `M.model.jpg` + Forest Green | v21 visually reviewed: full sleeves without the previous skin holes; 60.09 s |
| `F.model.jpg` + Forest Green | v21 closed front improved; hair/coat fringe still imperfect. v24 final fringe probe saved for review |
| `male/male_person_19.png` + Forest Green | v21 collar/full sleeves control passed and was visually reviewed; 59.97 s |
| Same shirtless-source collar at full resolution | v30: source face/upper neck retained, collar cutout corrected; 70.05 s; visual review passed |
| `female/female_person_02.png` + Forest Green | v21 sleeves complete; precise collar/style fidelity remains approximate |
| `Class/Ali Ammar.jpeg` + `SHIRT` | v21 short sleeves corrected and donor removed; visually reviewed; 59.61 s |
| `Class/Addan 2.jpeg` + red black shirt | v21 check pattern and full sleeves retained; visually reviewed; 61.36 s |
| WhatsApp seated portrait + Forest Green | v25 fallback + cropped shoulder correction passed, visually reviewed; 94.44 s |
| `ABD.jpeg` + `body black` | v19 safety-rejected; no blocked output returned or retained. Still a named unsupported generation result |
| `Class/Ali Ammar.jpeg` + Modern Kurta | v29: untucked long hem, standing collar and source trousers retained; visually reviewed; 60.64 s |
| `male/male_person_02.png` + Sherwani | v29: closed patterned coat and complete sleeves; source lower trousers retained; 61.33 s |
| `female/female_person_02.png` + Trench Coat | v29: belted coat extends below hips; source shorts/legs retained; 61.72 s |
| `male/male_person_01.png` + Chapan | v29: patterned body extends below hips and source shorts/legs retained; 60.69 s; precise sleeve cut remains approximate |

Private evidence: `instance/remaining-repairs-v21/`,
`instance/remaining-repairs-v24/`, `instance/remaining-repairs-v25/`,
`instance/remaining-repairs-v27/`, `instance/remaining-repairs-v28/` and
`instance/remaining-repairs-v29/`.
These are named tests, separate from the
full-matrix journal. Source files changed during development probes; the
v21 service context and matching manifest identify the code actually loaded.
v24 probes additionally record and verify the service fingerprint per request.

## Verification and full coverage

121 service tests, 52 application/analytics/sizing tests and 9 audit tests
passed. JavaScript syntax checked successfully. Final live checks and the
full-matrix sweep remain in progress.

Live priority probe observed an interactive request queued and a competing
audit request receiving HTTP 429. That probe's long-garment result was quality
rejected, so it verifies queue behavior and error handling, not a successful
preview. Its temporary upload/failure event was removed from live analytics;
private evidence remains. A successful post-fix web-route probe is pending.

Preflight checked all 72 person images: 71 ready, one two-person rejection
(`Class/duo.jpeg`). Thirty real upper references are ready; two demo records
have no product image. There are 2,304 total inventory pairs, including 2,130
eligible GPU pairs and 174 preflight-rejected pairs.

At approximately one minute per generation, a full sweep requires roughly
35.5 GPU hours before retries or manual review. The full sweep and arbitrary
future-photo correctness are not certified. Successful HTTP responses remain
technical passes until visual review, including print, hem and collar details.

The checkpointed v29 matrix is in `instance/all-photo-audit-v29/`. It explicitly
reuses v25 preflight classifications because subsequent changes concern
output processing; every file hash, person/product scope and garment-length
field was rechecked. Its inference fingerprint identifies the new code,
separately from the recorded preflight provenance. Full generation is not
marked complete by preparing a manifest.

The four final long-reference probes completed on their first candidate and
were visually inspected. Long-mode validation requires visible extension
beyond the hips; it does not assume every long top must reach the knees.
Exact hem length, sleeve cut and print arrangement remain approximate rather
than physical fit measurements. Full matrix outputs still need visual review.

The CLI audit requests idle-sleep prevention only for its own lifetime and
restores the previous thread execution state on exit. It checkpoints before
stopping, including while yielding to interactive requests. Source/reference/
result thumbnails and per-case error names are available in its local HTML.

## Reported latency and appearance regression, v31

The v30 full audit was checkpoint-stopped to give interactive generation the
GPU. A reported Forest Green Twill Shirt preview took 133.05 seconds including
waiting; queue and compute durations were not separately persisted, so their
exact split is unknown. Visual inspection found darkened fabric, excessive
looseness, an open front/hem and cuff artifacts despite passing technical gates.
Those gates do not certify product fidelity.

On the original dataset photo `male/male_person_22.png`, an app-equivalent JPEG
and the same reference rendered with DeCo in 45.12 seconds including renderer
loading. Visual comparison showed closer green colour, a closed front and
closer fit, with a minor cuff artifact still visible. This is an inference and
selection change, not training or a universal accuracy claim.

v31 conservatively selects DeCo for standard-length, plain coloured long-sleeve
references when both source arms have observed clothing near the elbows.
Explicit engine settings, per-reference calibration, long tops, printed or
ambiguous references and bare/short-sleeved arms retain their existing policy.
Actual photo probes select the alternate for male22 and retain FASHN for
shirtless male19 and short-sleeved female02. New unit regressions cover the
selection boundaries. Full v30 results remain tied to their original code;
they must not be resumed as evidence for v31 without a new manifest.

Private comparison evidence: `instance/user-quality-2026-11-19/`.

The restarted v31 automatic endpoint reproduced this selection in a live
HTTP 200 run: 54.66 seconds, DeCo, one candidate and no sleeve refinement.
Full-resolution inspection confirms improved colour/front/fit; a grey edge
remains beside one forearm, so the preview is improved rather than flawless.
The earlier 45.12-second explicit-renderer comparison and this automatic run
are separate measurements, not a guaranteed generation duration. The website
is running on port 5050; the full batch remains checkpointed (24 finished
pairs in the v30 journal, including rejected cases, not full coverage).

The subsequent v34 speed work is documented in
`SPEED_OPTIMIZATION_2026-11-19.md`: request-level segmentation reuse and fast
standard-upper sampling produced targeted live measurements around 38–40 s.
Those measurements do not close the appearance issues or certify the full
matrix. The attempted native INT8 path was slower and is not enabled.

The subsequent dual-model work is in `DUAL_MODEL_SPEED_2026-11-20.md`.
v38 retains both renderers, adds bounded/file-backed model parking and a
fast profile with source-dependent sampling. Named measurements include
22.25 s for a warm DeCo return and 31.88 s for the first mapped FASHN return;
20–30 s is a target rather than a hard bound. 127 service tests pass. Existing
appearance defects and full-matrix coverage remain explicitly incomplete.

Following the recent detail complaint, balanced was restored as the launcher's
default. `QUALITY_REVIEW_2026-11-20.md` records the matched 20-step latest-photo
check (38.33 s), improved fabric detail and remaining sleeve-edge defects.
