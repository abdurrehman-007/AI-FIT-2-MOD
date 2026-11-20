# Upper try-on speed optimization, v34

The local service remains on CUDA with its original weights and resolution.
Request-level segmentation reuse and the fast FASHN sampling profile bring
the targeted standard-shirt probes into roughly 38–40 seconds. These are local
service measurements with no queue; they are not a universal latency SLA.
Long garments, renderer loading, sleeve repair and quality retries can exceed
40 seconds. The full matrix audit remains checkpointed to prioritize shoppers.

## Changes

- Cache SCHP results by image pixels, dimensions and mode within the admitted
  request. Reuse identical pixels, invalidate edited pixels, limit to eight
  entries and clear on every exit. No personal-photo cache survives the request.
- Standard upper FASHN first candidates use the upstream 20-step fast setting;
  long tops, other categories and retries retain 30 steps. DeCo stays at 50.
- Fast FASHN uses the supported last-three-step conditional-only option.
  Sequential guidance avoids computing unconditional outputs that the sampler
  discards. Other profiles retain the original last-one-step behavior.
- Preserve output safety, source compositing, neck, pose, body outline, sleeve
  and applicable garment checks. No validator was disabled for speed.
- Expose private stage timings in `X-AIFIT-Timings`: input, renderer selection/
  loading, mask preparation, generation, total and segmentation call/cache data.
  Generation includes backend preprocessing and output safety checks; it is not
  denoising alone. Total excludes queue waiting, which has its existing header.
- Update the waiting message to describe typical standard-top timing and the
  slower first-use/long-outfit/extra-check cases.

## Measured results

RTX 3060 Laptop GPU, 6 GB; torch 2.5.1+cu118, BF16 renderers. App-equivalent
JPEG inputs, direct authenticated local `/infer`, one candidate, no sleeve
refinement, queue wait zero:

| Source / reference | Seconds | Notes |
|---|---:|---|
| male22 / Forest Green, old v31 | 54.66 | DeCo, 50 steps, before request cache |
| male22 / Forest Green, cached v32 | 38.27 | Includes 10.0 s renderer switch; output pixels identical to v31 JPEG |
| male19 / Forest Green, v33 | 42.16 | 20 steps, original last-one-step setting |
| male19 / Forest Green, final v34 | 39.80 | 20 steps, last-three-step setting |
| female02 / Forest Green, final v34 | 38.69 | Short-sleeved source; full sleeves generated |
| male22 / red black shirt, final v34 | 37.89 | Contrasting check reference, kept on FASHN |
| male22 / Forest Green, final v34 | 36.47 | DeCo, unchanged 50 steps; includes 9.02 s renderer switch |

Full-resolution review found the male19 collar/neck and sleeves usable and the
red/black checks recognizable. Female02 still has an imperfect neckline edge
also present in the older 30-step comparison. Some cuff boundaries and the
DeCo forearm edge remain imperfect. Printed arrangement, fit and closures are
approximate; HTTP 200 and pose passes do not certify exact product fidelity.

Private images, JSON measurements and the kernel benchmark are in
`instance/speed-v32/` (folder name retained across these benchmark revisions).
Earlier records predate the final service fingerprint and must not be merged
into a current full-matrix accuracy score.

## Quantization experiment and rejected alternatives

Native CUDA INT8 matrix multiplication works in this installed PyTorch runtime.
A local dynamic per-token activation/per-output-channel weight INT8 benchmark
on three representative FASHN linear shapes included quantize/dequantize cost:

| Matrix shape M/K/N | BF16 ms | INT8 ms | Relative RMSE |
|---|---:|---:|---:|
| 6912 / 1280 / 1280 | 1.23 | 3.72 | 1.15% |
| 6912 / 1280 / 5120 | 4.62 | 8.51 | 1.16% |
| 6912 / 5120 / 1280 | 4.02 | 10.34 | 1.26% |

This is a native-kernel experiment, **not a quantized full-model quality test**
and not proof that all INT8 implementations are slower. This path was rejected
because it missed the speed objective and introduced numerical error. A pinned
bitsandbytes wheel download was too slow to complete and was stopped; no base
PyTorch/CUDA upgrade or partial quantization was enabled. Production remains
BF16. GPU quantization depends on workload, kernels and hardware; see the
[official PyTorch GPU quantization tutorial](https://docs.pytorch.org/tutorials/unstable/gpu_quantization_torchao_tutorial.html)
and [bitsandbytes installation requirements](https://huggingface.co/docs/bitsandbytes/v0.49.0/installation).

Upstream batched guidance was also tested: male19 took 70.44 s including a
13.38 s renderer switch, with 52.73 s backend generation. It did not improve
this GPU's sampling throughput and was removed; low-memory sequential guidance
remains active. No unvalidated quantization or batched setting is left enabled.

## Validation

121 GPU-service regression tests and 9 audit tests pass. New checks cover cache
reuse, edited pixels, request cleanup, bounded memory, sampling-step selection,
quality retry/long-garment preservation and discarded guidance-pass elimination.
JavaScript syntax and whitespace checks pass. These checks plus the named live
probes establish targeted behavior, not perfection for every future input.
