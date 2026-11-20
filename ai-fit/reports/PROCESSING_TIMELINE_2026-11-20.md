# Processing timeline and realistic time estimates

The fitting room now displays photo validation, outfit preparation, generation,
quality checking and completion. While processing it shows elapsed time,
estimated remaining time, estimated total time and an explicitly estimated
percentage. The completed result retains a compact 100% / elapsed-time summary.

## Where progress comes from

- The browser sends a random 128-bit request identifier with `/api/tryon`.
  The app atomically reserves it with the authenticated owner, forwards it to
  the private GPU service and rejects duplicate submissions.
- `/api/tryon-progress/<id>` requires the same owner. It proxies authenticated
  GPU progress without exposing the GPU key, model internals or image bytes.
  Job metadata expires after one hour and is purged during subsequent requests.
- GPU progress has a bounded, locked registry (128 entries, 20-minute TTL).
  FASHN reports completed conditional/unconditional step work. DeCo reports
  after each iteration of its existing sampling progress iterator. No vendored
  model source, weights, seed or image resolution was changed for this feature.
- Progress updates are throttled to about once per second and synchronize CUDA
  when reporting, so queued GPU work is not described as already completed.
  ETA updates from the observed sampling rate, excluding fixed preprocessing
  from its initial rate estimate. Remaining quality-check time is an estimate.
- Percentages weight processing stages; they are not an exact fraction of
  total mathematical work. The label says **Estimated progress**. Refinement
  can increase the time estimate without moving the percentage backwards.
- GPU completion is capped at 99% in the app until the returned image has been
  verified and saved. Only confirmed success reaches 100%. Failed/rejected
  previews never receive a successful completion bar.
- Queueing, unavailable tracking and overrunning estimates have honest fallback
  states. The browser counts elapsed time locally but never advances percentage
  just because time passed. It polls without overlapping requests and stops
  timers/polling after success, failure or sign-out.
- Photo uploading disables generation until the new upload is ready. Upload
  tokens and owner checks prevent older uploads/late generation responses from
  attaching to a new selection or a signed-out session. A previous browser test
  exposed the old upload race; it was corrected rather than counted as a pass.

## Quality and latency boundaries

Both renderers and existing automatic selection remain enabled. Balanced stays
the default: standard FASHN 20 steps / DeCo 50, with existing long-garment,
regional sleeve refinement and quality retry settings retained. Source masks,
output safety, neck/pose/body/sleeve checks and model/segmentation cache
optimizations remain active. The timeline does not reduce steps or stop quality
checks to fit a countdown.

The user's 20–30 second target is not a hard latency guarantee on this GPU.
Existing balanced measurements include approximately 38–55 seconds, with longer
first loads or retries. The UI must show actual estimates rather than a fixed
20–30 seconds. Fast sampling remains optional with its documented detail trade;
it was not silently re-enabled to meet this request.

## Verification

137 GPU-service tests and 55 app/analytics/sizing tests pass, plus the Node
progress presentation checks. Coverage includes progress stage/ETA accounting,
completed-step callbacks, callback cleanup on failure, authenticated ownership,
duplicate submission prevention, expiry, monotonic estimated percentage,
no premature 100%, delayed estimates and HTML escaping.

The disposable browser fixture checks desktop/mobile rendering, upload-before-
generate, result completion and error/retry behavior. Its metadata and screenshots
live under ignored `instance/progress-ui-check/`. Mocked screenshots test UI
behavior only and make no generated-image quality claim. Live GPU browser
verification uses disposable app data and existing test model pictures, leaving
production inventory and analytics untouched. A service disconnect during an
earlier live attempt was not counted as a successful live verification.

The completed live browser runs passed on both actual renderers, with no browser
errors, intermediate generation percentages and 100% only after saving the
result. The final v40 run retained FASHN 20 / DeCo 50 steps:

| Case | End-to-end browser time |
|---|---:|
| FASHN, first bare-source run | 37.77 s |
| FASHN, repeated run | 36.26 s |
| DeCo, first covered-source run | 37.14 s |
| DeCo, repeated run | 28.74 s |

The earlier v39 browser runs took approximately 45 s for FASHN and 41 s for
DeCo. These are local named measurements under varying desktop load, not a
controlled model-accuracy benchmark or a fixed SLA. The 20–30-second target is
met by the warm DeCo probe, but not all preserved-quality FASHN cases.

## Native attention optimization without reducing steps

A local BF16 SDPA kernel experiment at shape (1,10,6912,128) measured about
18.97 ms for the default dispatcher versus 8.88 ms for native cuDNN, with mean
absolute output difference approximately 0.00000433. This is a kernel benchmark,
not a claim that the entire application becomes twice as fast.

v40 prefers the native cuDNN attention path for eligible FASHN CUDA inputs,
using contiguous tensors. Unsupported shapes or allocation failure fall back
to the established dispatcher; unexpected runtime errors are not hidden.
No model weights, resolution, sampler seeds, guidance policy or step counts
were reduced. cuDNN changes floating-point operation order, so whole-image
bit-for-bit equivalence is not claimed. Full-resolution browser previews were
inspected and existing safety/pose/body/sleeve/reference checks remained active.

The two final FASHN runs recorded 1,036 native attention calls each, zero
fallback calls and zero disabled shapes. Private health/timing records and
desktop/mobile active/completed screenshots are in
`instance/progress-ui-check/real-results.json` and neighboring PNGs. The mock
browser test additionally passed upload blocking, failure/retry, sign-out and
late-response cleanup. GPU output is mocked only in that negative UI fixture.

Set `AIFIT_FASHN_ATTENTION=original` before restarting the GPU service to use
the previous dispatcher. This configuration is included in the runtime/audit
fingerprint. Automatic model selection and bounded CPU/file-backed parking
remain unchanged. The timeline was verified against v39 and v40; the final
website runs v40 in balanced mode.
