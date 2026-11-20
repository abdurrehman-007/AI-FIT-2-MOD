## Local upper try-on repair - 19 November 2026

The local GPU service can select the pinned FASHN backend with sequential guidance and CPU preprocessing on the 6 GB GPU. See [local engine setup](gpu_service/LOCAL_MODELS.md). Full photo-by-inventory testing is checkpointed while targeted repairs are verified. Earlier dated notes describe their respective runtime snapshots.

## Development resumed - 29 September 2026

See [current verification and remaining work](reports/RESUME_2026-09-29.md). CUDA memory error handling improved; empty real-fitting collection templates added. Production model resolution, chart fallback and unavailable confidence are preserved.

## UI polish - 28 September 2026

Controls, focus states, mobile wrapping and result actions have been refined. See [validation](reports/UI_POLISH.md). Current archives: `deliverables/AI-FIT-source.zip` and `deliverables/AI-FIT-datasets.zip`. Duplicate legacy Module2 archives were removed. Hard-refresh the browser (Ctrl+F5) to load updated styles.

# AI-FIT

Latest full-outfit correction (24 September): [OUTFIT_FIX.md](reports/OUTFIT_FIX.md). Old trouser-colour leakage and missed garment edges are addressed, with hand and footwear protection retained. 87 Python regression tests pass. Exact product fidelity remains unproven; see the visual findings rather than treating generation success as a quality score. Restart the GPU service to load the updated mask.

Latest Module 1 update: [pose-preservation fix and exact-case evidence](reports/POSE_FIX.md). MediaPipe landmarks now protect hands during inpainting; full-outfit masks preserve exposed skin, donor-model body cues are removed, and excessive output pose changes are rejected. Restart the GPU service and generate again to use the fix. This improves pose preservation, not guaranteed garment reconstruction. Earlier dated statements that GPU source was unchanged refer to the prior release.

## Current delivery — 23 September 2026

See [final verification and external dependencies](reports/FINAL_DELIVERY.md). The approved DeCo-VTON runtime is preserved. Module 2 exposes Upper/Lower only, with inches, feet/inches and kg. Module 3 includes category popularity, aggregate customer preferences and scoped offline evaluation reporting with unavailable states.

Inventory → Edit size chart offers **Load womenswear reference for review**, from EshalPreet's general Pakistani womenswear body chart. Loading fills the form; Save applies it. Check applicability first: it is not a menswear, product-specific fit or national-standard chart. Existing charts are not changed automatically. CSVs and unit/basis decisions: [Pakistan chart audit](datasets/pakistan_charts/DATA_AUDIT.md).

No suitable independent Pakistani fitting-labelled dataset was established. Chart fallback and unavailable fitting confidence are preserved. RF/KNN training/loading supports future reviewed data; adjacent `.metrics.json` reports are shown only when retailer/category/chart/features/units match. No model is unpickled for analytics, and an offline report does not prove deployment or provenance. Chart changes invalidate its scope.

Preference summaries use successful try-ons collected after this update, with store-specific keyed account pseudonyms and aggregate counts only in analytics responses. Older identities are not guessed. Body-shape segmentation is unavailable without all necessary measurements. Returns are period events, not a matched sales cohort; before/after comparisons are observational. Trend dates are UTC.

Current archives: `deliverables/AI-FIT-source.zip` and `deliverables/AI-FIT-datasets.zip`; `manifest.json` records verified file hashes. Compatibility-named Module2 ZIPs are also refreshed. These archives exclude credentials, private photos/databases, installed dependencies and model weights. Your existing `instance` data stays locally. This is source delivery, not a clean-machine GPU installer or private-data backup.

In this folder, run `.\start-gpu.ps1` in PowerShell for GPU plus app, or `.\start.ps1` for app only; open `http://127.0.0.1:5000`. The GPU launcher reuses the existing Python 3.11.5 environment and caches documented below. Audit test services are temporary.

Verification commands (use the existing GPU Python environment):

```powershell
python -B -m unittest test_app test_sizing test_analytics
python -B -m unittest discover -s gpu_service -p 'test_*.py'
node test_catalog.cjs
node test_generate.cjs
node test_full_ui.cjs
python -B scripts/package_module2.py
```

Browser tests use Playwright/Edge paths in their scripts; adjust these on another computer. They use a disposable database and a mocked GPU response to check application flow. Real GPU output was tested separately; native OS sharing remains unverified.


A runnable Flask/Jinja2 application based on the AI-FIT proposal: virtual fitting room, store-aware sizing, retailer analytics, inventory management and QR/tablet access.

Module 2 adds inches/feet sizing and separate category charts. See [evaluation and dataset findings](reports/MODULE2_EVALUATION.md).

## Module 1 verification (September 10, 2026)

Photo preprocessing and Windows service integration were verified with the existing Anaconda Python 3.11.5 environment, ONNX Runtime 1.20.1 from `.preprocess-deps`, MediaPipe 0.10.14, and the RTX 3060 Laptop GPU. No existing-environment packages were replaced. The launcher validates full preprocessing initialization before starting the service; its ONNX repair installs only into the project-local dependency folder. MediaPipe loads the local task as bytes, and failed segmentation initialization closes the pose runtime.

PNG transparency now survives upload, media download and the GPU request. Existing cutouts retain their alpha; opaque garments undergo CPU foreground segmentation. Health reports unavailable CUDA and incomplete initialization without crashing. Regression coverage includes pose rejection, cutout preservation, blank garments, service readiness, authorization and upload round trips.

Actual `/infer` runs with the saved customer photo returned blue, black and grey T-shirts at original 961 x 1280 dimensions in 13.94, 13.11 and 13.67 seconds respectively (384 x 512 internal generation, 50 steps). Visual inspection confirmed the garment colours and removal of the grey product's checkerboard background. The grey result introduced a necklace/detail absent from the product; these outputs remain previews, not exact garment reconstruction. A lower-body request correctly returned 422 because the customer's ankles were cropped out. Successful lower/overall generation is not established by this check.

Local private evidence: `instance/module1-results.jpg`, `instance/module1-results.json`, and `instance/module1-check.log`. These contain customer photos or local runtime data and stay outside the portable deliverable.

Run regression tests with the same existing interpreter:

```powershell
& 'E:\Comsats\8th Semester\Fyp 2\Virtual-Try-On-main\.venv\Scripts\python.exe' -B test_app.py
& 'E:\Comsats\8th Semester\Fyp 2\Virtual-Try-On-main\.venv\Scripts\python.exe' -B -m unittest discover -s gpu_service -p 'test_*.py'
```

## Verified local GPU setup (September 10, 2026)

The Windows launcher now defaults to DeCo-VTON using the existing Python 3.11.5 environment and RTX 3060 Laptop GPU (6 GB). Production-route tests on the customer's exact photo and uploaded blue/black T-shirts returned correctly coloured crew-neck shirts in 15–16 seconds per request after model loading. Inference uses 384 × 512, 50 DDIM steps, and a local safety checker. Output is restored to the original photo dimensions; this does not increase the generated detail resolution.

The SCHP clothing parser runs on CPU. The mask protects the face, forearms/hands, trousers and background, while allowing the shirt hem and sleeves to change. These are visual previews, not exact garment or fit reconstructions: the verified black result added a necklace and fine garment details can still differ. Bottoms and full outfits have not been visually validated with this checkpoint.

To start both services:

```powershell
.\start-gpu.ps1
```

The launcher reuses the original project without modifying it. Pass `-ExistingProject 'your project path'` if it is elsewhere. It downloads the pinned 3.4 GB DeCo-VTON checkpoint on first setup, resumes interrupted downloads and verifies its SHA-256. This computer already has the checkpoint. Cached Stable Diffusion VAE, scheduler, safety checker and SCHP files plus the existing project's Python dependencies are also required. Missing cached components fail explicitly; this is an existing-environment launcher, not a clean-machine GPU installer.

The launcher waits for the GPU service, loads the private shared key from `instance/gpu.key`, and starts http://127.0.0.1:5000. Keep the terminal open when launching manually. Stop an existing instance on port 5000 before launching another.

The older CatVTON engine is available only with `$env:AIFIT_ENGINE='catvton'`; it reproduced incorrect garment colours in the exact-photo regression and is not the default. There is no silent fallback to that engine. The separate Linux service below remains an alternative, unverified deployment path. Firebase and trained size-model setup remain outstanding.

The portable ZIP excludes model weights, installed dependencies, credentials and private runtime photos. DeCo-VTON source and its licence are included under `gpu_service/vendor`; the model downloader is `gpu_service/download_deco.py`.

## Run on this computer

Open PowerShell in this folder and run:

```powershell
.\start.ps1
```

Open **http://127.0.0.1:5000**. The bundled dependencies have already been installed under `.vendor` for this computer. If another copy is running on port 5000, use that copy or stop it first.

On another machine with Python 3.12:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python app.py
```

For macOS/Linux, use `python3 -m venv .venv`, `.venv/bin/python` and the same Python commands. The portable ZIP excludes installed dependencies and private runtime data.

## First use

1. Explore the sample collection or open **Find my size**. Sample garments are clearly marked illustrations, not actual catalog photos.
2. Create an account with **Sign in → Create an account**. There are no default admin credentials.

For local accounts, **Sign in > Forgot password?** emails a link valid for 30 minutes. Set `SMTP_HOST`, `SMTP_PORT` (default 465), `SMTP_FROM`, and the mail provider credentials `SMTP_USER` and `SMTP_PASSWORD` in the server environment. SMTP must support TLS on connection. Set `PUBLIC_URL` to the AI-FIT address recipients can open and restart the server. Firebase accounts use Firebase password reset email.
3. Open **Inventory** and create your store. Your account owns this store; another account cannot modify it or read its analytics.
4. Add garments or download and import the CSV template. Add a real garment photograph through **Edit** for each product you want to use for try-on.
5. Edit your store's body-size chart. The initial chart is illustrative and must be replaced with your store's verified measurements.
6. Switch the store selector to your store, upload a person photo with consent, and generate after starting the GPU service.
7. Record sales/returns in **Store insights**. A sale linked to a completed try-on ID contributes to conversion; a generation alone never counts as a sale.
8. Set `PUBLIC_URL` to the deployed HTTPS address, then download the store QR code or open tablet kiosk mode.

## What runs now and what needs setup

| Capability | Delivery status |
|---|---|
| Responsive fitting room, catalog, search/category filters | Implemented |
| Account registration/login, sessions, retailer ownership | Works locally; Firebase option implemented |
| JPEG/PNG validation, consent, private photos, removal | Implemented |
| Inventory CRUD, CSV upserts, size charts | Implemented |
| Size/chart comparison, fit tags, notes | Works without a trained model |
| Random Forest + KNN | Offline training/calibration/evaluation supplied; no suitable labelled fitting data established |
| Local Windows try-on | DeCo-VTON verified on two uploaded T-shirts; detail fidelity remains imperfect |
| Alternative Linux CatVTON / MediaPipe / background segmentation | Implementation supplied; deployment unverified |
| Save/share generated result | Implemented; native share depends on browser support |
| Popularity, conversion, size/body-shape distributions, returns and latency | Uses actual recorded events; no fabricated statistics |
| QR and kiosk | Implemented; mobile QR requires a reachable deployment URL |
| Public deployment | Not performed; Flask requires a Python host, and Sites scaffolding failed locally |

The missing GPU/model configuration is **not** replaced with a fake rendered image. Without labelled data the size result explicitly uses chart matching and provides no invented confidence score. This is a functional application with external AI setup remaining, not a validated production AI system.

## Configuration

`.env.example` documents the environment variables. Export them in your shell or hosting dashboard; the application does not automatically load `.env` files.

| Variable | Purpose |
|---|---|
| `SECRET_KEY` | Persistent random session-signing secret; local development generates one in `instance/session.key` |
| `DATA_DIR` | Writable persistent directory for SQLite and image files |
| `STORAGE_BACKEND` | `sqlite` (default) or `firestore` |
| `AUTH_BACKEND` | `local` (default) or `firebase` |
| `GOOGLE_APPLICATION_CREDENTIALS` | Path to Firebase service-account credential outside source control |
| `FIREBASE_WEB_API_KEY` | Firebase project's web API key for email/password sign-in |
| `CATVTON_URL` | Private GPU service base URL |
| `CATVTON_API_KEY` | Shared secret for application → GPU authentication |
| `SIZE_MODEL_PATH` | Legacy setting; Module 2 chart estimates do not load unvalidated model artifacts |
| `PUBLIC_URL` | Canonical public HTTPS origin used in QR codes |
| `COOKIE_SECURE` | Set `1` for HTTPS deployments; leave `0` for local HTTP |

## Firebase setup

1. Create a Firebase project, enable **Authentication → Email/Password**, and create a Firestore database.
2. Create a server service account with access to that project's Auth and Firestore. Keep its JSON file outside this repository. Set `GOOGLE_APPLICATION_CREDENTIALS` to the file location.
3. Set `AUTH_BACKEND=firebase`, `STORAGE_BACKEND=firestore`, and `FIREBASE_WEB_API_KEY` from project settings.
4. Restart the app. The same email/password form uses Firebase's REST sign-in, and the server verifies the returned ID token with the Admin SDK before creating an application session. The API also accepts an already-issued Firebase ID token at `/api/auth/firebase`.
5. Use server-only Firestore access. Since clients do not access Firestore directly, deploy rules denying browser reads and writes:

```text
rules_version = '2';
service cloud.firestore {
  match /databases/{database}/documents {
    match /{document=**} { allow read, write: if false; }
  }
}
```

Admin SDK access uses service-account IAM and does not rely on these client rules. Images remain in `DATA_DIR`; mount persistent storage even when using Firestore. Local accounts/data do not automatically migrate when switching backends. Firebase credentials, remote availability and end-to-end behavior were not tested in this delivery because no project was supplied.

## CatVTON GPU setup

Use a separate Linux CUDA environment compatible with the [official CatVTON repository](https://github.com/Zheng-Chong/CatVTON). Follow its environment instructions and retain a tested repository revision. The adapter uses the `CatVTONPipeline` / `AutoMasker` API from that repository.

1. Clone and install CatVTON following its README, including its CUDA/PyTorch and DensePose dependencies. Do this in a separate environment from the web application.
2. Install the service extras in that environment: `pip install Flask mediapipe "rembg[cpu]"`.
3. Download a compatible Pose Landmarker `.task` model from the [MediaPipe Python guide](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/python). Set `POSE_MODEL_PATH` to its absolute path.
4. Copy `gpu_service/server.py` into the CatVTON repository as `aifit_server.py` so its `model` and `utils` imports resolve.
5. Set a random `CATVTON_API_KEY`, then run `python aifit_server.py`. Startup downloads CatVTON checkpoint assets and the U2NetP segmentation model. It fails clearly if CUDA or the pose model is missing.
6. The server listens on loopback port 8001. Use a private HTTPS reverse proxy when the web app runs on another machine. Configure that proxy's body limit and request timeout for image inference. Do not expose an unauthenticated GPU service.
7. On the web server, set `CATVTON_URL` (for the same host, `http://127.0.0.1:8001`) and the same key; restart the application.
8. Upload actual garment and person images, and verify output quality with your own test set before a demonstration.

The adapter validates 33 pose landmarks and shoulder/hip visibility, removes the garment background, uses CatVTON's body garment mask, and returns a generated JPEG. Pose landmarks are a quality gate; CatVTON's own masker constructs the inpainting mask. Only one GPU inference runs at a time; a concurrent request gets a busy response. The web request timeout is 240 seconds. Production throughput will need a job queue and capacity planning.

**Model licensing:** The upstream CatVTON project identifies its weights/demo as noncommercial. Review its [license](https://github.com/Zheng-Chong/CatVTON/blob/main/LICENCE) and obtain appropriate rights before commercial SME deployment. This delivery does not download or redistribute its weights. GPU output and segmentation quality remain unverified without the GPU environment.

## Module 2 sizing and research

Body measurements and category charts use **inches**. Height has separate **feet / inches** fields; weight stays **kg**. Choose a garment to see the relevant fields. Upper garments use chest/bust and waist; lower garments use waist and hips; full outfits use all three. Retailers can add the supported optional shoulder/sleeve/inseam/thigh columns in Inventory > Edit size chart. Optional columns must be complete for all sizes.

Each category has its own body chart. Starter values are illustrative, converted from the original centimetre chart; they are not Pakistani sizing standards. Existing stored centimetre charts are converted on read without overwriting them. New chart submissions require `chart_units: "in"`; size requests require `units: "in"`, `height_feet`, `height_inches` and kg `weight`. Flat garment widths must not be entered as body circumferences.

Recommendations use relevant measurements and available sizes. They explicitly flag poor chart matches and do not claim a perfect fit. No raw measurements are retained. Height/weight are collected as context and do not drive chart matching.

Two permitted Pakistan research datasets are delivered in `datasets/module2`, with source/licence details and original files. Neither contains the required confirmed garment-size/fit labels. No production RF/KNN model, independent fitting accuracy or calibrated fitting confidence is claimed. Existing uncalibrated model files are not loaded by the sizing endpoint.

See [Module 2 evaluation](reports/MODULE2_EVALUATION.md), [data audit](reports/module2-data-audit.json) and [sources/licences](datasets/module2/SOURCES_AND_LICENCES.md).

The offline training script requires independently confirmed fitting labels, separate participant-disjoint training/calibration/test files and a provenance manifest. It reports RF/KNN and raw/calibrated ensemble metrics; its research artifacts are not auto-enabled in the app. Use:

```powershell
python train_size.py train.csv --calibration calibration.csv --test external-test.csv --manifest provenance.json --output models/size-research.joblib
```

Required CSV columns: `person_id,fitting_id,category,label_origin,units,height_feet,height_inches,weight,size`, plus the required dimensions for the category. `label_origin` must be `independent_physical_fitting`, `units` must be `in`, and labels are XS�XXL. See `datasets/module2/provenance.example.json` for metadata requirements; it is an incomplete template and cannot train a model as supplied. Never derive labels from the chart used for evaluation. The script checks leakage and declared provenance, but declarations still require a human audit.

## Analytics definitions

- **Completed try-ons:** successful GPU responses that decode to a valid image.
- **Conversion:** distinct completed try-ons linked to a recorded sale, divided by completed try-ons, within the selected reporting window.
- **Return rate:** return events divided by sale events recorded in the window. This is period activity, not a matched order-cohort estimate; returns from earlier sales can exceed current-period sales.
- **Baseline comparison:** retailer-entered pre-AI orders/returns versus current-period activity. It does not establish causality.
- **Segments:** coarse chest-versus-hip categories; no raw measurements are retained.
- **Latency and failures:** actual server-observed request duration and failed service attempts.

Sales/returns are manually recorded because checkout, payments and ERP integration are excluded by the proposal. Order references prevent duplicate sale or return records. Popularity/segmentation charts stay empty until real activity exists. Model evaluation metrics are available in the generated training report; the dashboard does not invent model accuracy.

## Privacy and operation

Photos are re-encoded to remove metadata and stored under random identifiers. Person/result media requires the owning account. The server rejects expired personal photos after 24 hours. Run physical cleanup at least hourly:

```powershell
python -m flask --app app:create_app cleanup-photos
```

When using the preinstalled `.vendor` dependencies instead of a virtual environment, set `PYTHONPATH` to this folder's `.vendor` before using `python -m flask`.

Kiosk mode clears previews and signs out after two minutes without keyboard/pointer activity. Use a dedicated kiosk browser profile and end a customer's session before handing over the device. Keep local session keys, images, database files and service-account credentials out of source control and backups shared with others.

For deployment, use the included Dockerfile with a persistent `/data` volume, HTTPS proxy, secret environment variables, and scheduled photo cleanup. Example:

```text
docker build -t aifit .
docker run --env-file .env -p 8080:8080 -v aifit-data:/data aifit
```

The example `.env` must contain real environment assignments, including `DATA_DIR=/data`, a stable random `SECRET_KEY`, `PUBLIC_URL` and `COOKIE_SECURE=1` behind HTTPS. The Docker image runs Waitress as a non-root user. The Flask development server is for local preview only. The current persistence adapter is suitable for a capstone/small installation; it reads collections for analytics and should be replaced with indexed, paginated queries and pre-aggregation for large stores.

## Validation

```powershell
python test_app.py
python test_sizing.py
python -m unittest discover -s gpu_service -p "test_*.py"
python -m compileall -q app.py storage.py sizing.py train_size.py gpu_service
node --check static/app.js
```

The integration suite checks account flows, CSRF, retailer isolation, atomic CSV validation, persistence after restart, invalid images, photo ownership/expiry/removal, chart sizing, consent, unavailable AI, inference success/failure contracts and sales conversion. GPU HTTP responses are mocked in tests: a passing test does not establish real model performance. The Windows production GPU route was separately executed and visually checked with the exact customer photo and two garments. Real Firebase and Docker deployment have not been validated. Optional WebMCP catalog tools are feature-detected; a supported validation context was unavailable.

## Source layout

`app.py` provides routes, validation and access checks. `storage.py` implements SQLite/Firestore persistence. `sizing.py` provides recommendations; `train_size.py` trains models. `templates/` and `static/` contain the responsive UI. `gpu_service/server.py` is the separate inference adapter. `test_app.py` contains the integration suite.

Technical references used: [Firebase Auth REST API](https://firebase.google.com/docs/reference/rest/auth), [CatVTON](https://github.com/Zheng-Chong/CatVTON), [MediaPipe Pose Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/python), and [rembg](https://github.com/danielgatis/rembg). The provided proposal determined application requirements; its administrative instructions and project schedule were not treated as instructions to execute.


## Module 2 current integration

Sizing exposes Upper Body and Lower Body only; Module 1 categories remain unchanged. Height is feet/inches, body measurements inches, and weight kg. Tight/Perfect/Loose labels are chart-estimated. RF/KNN schema-v2 loading is integrated with retailer/category/chart validation and safe fallback. No suitable fitting-labelled cohort was verified, so no production model is enabled and confidence remains unavailable. See [current evaluation and model installation](reports/MODULE2_EVALUATION.md).
