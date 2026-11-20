# Local Windows startup verified — 13 November 2026

Open http://127.0.0.1:5050. Restart both services with `D:\ai-fit\Open-AI-FIT.cmd`.
The complete application is in `D:\ai-fit\ai-fit`.

The isolated `.gpu-runtime` environment uses Python 3.11, PyTorch 2.5.1+cu118
and torchvision 0.20.1+cu118. The signed Microsoft Visual C++ x64 runtime was
installed. CUDA calculation passed on the NVIDIA GeForce RTX 3060 Laptop GPU
(6 GB, existing driver 517.70). No NVIDIA driver replacement was needed.

DeCo-VTON uses the existing local UNet, downloaded VAE, SCHP and official FP16
safety-checker weights. Auxiliary weights are cached under `.models/hf-cache`.
The maintained Stable Diffusion inpainting repository supplies the scheduler
and safety checker. Upstream parser source is in `.external/CatVTON`.
The GPU service listens on 127.0.0.1:8001. Both services bind to loopback.

Verification:
- `pip check`: no broken requirements.
- App, sizing and analytics: 51 tests passed.
- GPU service regression tests: 38 tests passed.
- Live GPU health: ready, DeCo backend, pose validation and segmentation enabled.
- Live app bootstrap: `tryon_ready=true`.
- Actual authenticated local `/infer`: HTTP 200 in 23.75 seconds using the
  existing Module 1 test inputs. Generated JPEG validated with Pillow.
- Private output: `instance/startup-smoke-2026-11-13.jpg`.

Logs are in `instance/app-output.log`, `instance/app-error.log`,
`instance/gpu-output.log`, and `instance/gpu-error.log`.
Generation success verifies runtime operation; it does not establish garment
reconstruction or fitting accuracy.
