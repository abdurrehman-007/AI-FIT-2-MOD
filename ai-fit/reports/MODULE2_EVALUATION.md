# AI-FIT Module 2 evaluation

Updated: 23 September 2026 (historical tests and research below retain their dates)

## Current completion update

Upper/Lower and inches, feet/inches, kg are preserved. A verified **unit/basis** womenswear body reference is now available as an opt-in chart-editor preset; it is not independently fit-validated and is not applied to existing stores. Two CSVs contain 12 rows each. Details and original source values are in [the current Pakistan audit](../datasets/pakistan_charts/DATA_AUDIT.md). Earlier statements about no brand integration refer to the historical implementation, not this opt-in addition.

Actual RF/KNN evaluation reporting is now supported in Store insights using adjacent `.metrics.json` files. It checks retailer, category, exact chart hash, features, units, six classes, split hashes, test count and finite metric ranges. It never unpickles a model for analytics. Missing/incompatible reports show unavailable, not zero accuracy. Reports describe offline evaluation, not a deployed-model guarantee. No suitable real labelled cohort was established in the renewed source review; chart fallback and unavailable fitting confidence remain correct.

Current delivery and cross-module tests are recorded in [FINAL_DELIVERY.md](FINAL_DELIVERY.md). Historical full-outfit sizing screenshots are excluded from the current source archive; Module 1 retains its full-outfit category.

## Result

The application now accepts body measurements in inches, height as separate whole-feet and fractional-inches fields, and weight in kg. Upper and lower recommendations use independently editable category charts and only their relevant dimensions. Module 1 preprocessing, GPU inference and image-transparency behaviour are preserved.

**No production RF/KNN fitting model was trained or enabled. Independent fitting accuracy and confidence calibration are unavailable.** The downloaded permitted Pakistan datasets lack the required confirmed garment-size/fit labels. No chart-derived labels, synthetic training dataset, inferred bust measurements, or invented confidence percentages are delivered.

## Measurements and chart behaviour

| Category | Required body dimensions | Optional retailer-chart dimensions |
|---|---|---|
| Upper | Chest/bust, waist | Shoulder width, arm/sleeve length |
| Lower | Waist, hips | Inseam, thigh circumference |

All chart dimensions are inches. Optional dimensions appear in the shopper form only when the retailer supplies that column for every size. Height and weight are validated but do not alter a chart-only estimate. No body dimension is estimated from BMI. The UI tells users to enter full circumferences rather than flat garment widths.

The shopper form and chart editor expose only `upper` and `lower`, with XS–XXL rows. Stored legacy `overall` charts remain intact for data compatibility; the size API rejects full-outfit products. Module 1 keeps all its original categories. Default numbers are the original illustrative chart converted from centimetres, not Pakistani population standards. A retailer must replace them with applicable body charts. They are not copied from unlicensed brand charts. Retailer charts validate finite numeric values, category relevance, complete columns and nondecreasing size progression. Saves are atomic and ownership-restricted.

Old stored `chart` values remain unchanged. They are interpreted as centimetres and converted on read with the exact 2.54 cm/in factor. New category charts are persisted only when settings are saved. A legacy centimetre request is rejected with an explicit units message instead of silently being read as inches. Existing products, photos and retailer records were not bulk-modified.

The estimator filters to available sizes, prefers those accommodating the required circumferences with a 0.2-inch chart tolerance, then minimizes squared distance. Lengths have half the per-inch weighting of circumferences. The tolerance and weights are implementation heuristics, not empirically calibrated fit probabilities. Larger gaps or incompatible dimensions are flagged as outside the chart range or requiring a length check. The app labels Tight, Perfect and Loose explicitly as chart-estimated. Perfect denotes proximity to chart dimensions, not guaranteed physical fit. Out-of-range warnings remain separate. Missing, legacy, corrupt, disabled or incompatible models return chart estimates with `confidence: null`.

## Dataset audit and licensing

| Delivered source | Local inspection | Suitability for supervised fitting |
|---|---|---|
| Pakistan school-study S1 workbook, Tanveer et al. (2025), CC BY 4.0 | 1,200 nonempty rows, 47 columns, 1,200 unique IDs; height/weight and pre/post study variables | No garment categories, confirmed size/fit labels, chest, waist or hips; schoolchildren cannot establish adult shopper performance |
| Pakistan DHS national anthropometry, Open Data Pakistan / DHS Program, source-labelled Creative Commons Attribution | 65 Pakistan indicator rows, surveys 1991, 2012 and 2017; original HXL metadata row retained | Aggregate health/nutrition indicators, not individual fitting observations |

The school workbook's publisher MD5 was verified. Source files are unchanged. Exact SHA-256 values, missing counts and numeric ranges are in [the machine-readable audit](module2-data-audit.json). `Age` and `agemean` have different ranges in the workbook, so their coding was not guessed or repurposed as shopper ages. No medical conclusions are drawn from either dataset.

Full attribution, download endpoints, source licence links, access outcomes and redistribution notes are supplied in [SOURCES_AND_LICENCES.md](../datasets/module2/SOURCES_AND_LICENCES.md). The [Figshare record](https://plos.figshare.com/articles/dataset/S1_File_-_Effectiveness_of_a_school-based_physical_activity_intervention_on_overweight_and_obesity_among_children_and_adolescents_in_Pakistan/28475215) and [Open Data Pakistan resource](https://opendata.com.pk/dataset/pakistan-demographic-and-health-data/resource/e647ddde-5ff8-4698-b208-408f87665ebf) identify the delivered files and licences.

Other candidates were reviewed: Mendeley Pakistan youth anthropometry (CC BY 4.0, raw access unavailable during this run, metadata delivered); original ModCloth/RentTheRunway research (self-reported fit but mismatched geography/features/target and insufficiently established redistribution permission); SIZER (author-request access). None was converted into invented Pakistani XS–XXL labels. This bounded search did not establish a suitable labelled dataset; it does not prove that none exists.

## Independent accuracy and confidence calibration

| Metric | Result | Reason |
|---|---|---|
| RF independent test-set size accuracy | Not available | No suitable fitting-labelled training and independent test cohorts |
| KNN independent test-set size accuracy | Not available | Same evidence gap |
| RF/KNN ensemble accuracy | Not available | No deployed supervised model |
| Confidence calibration / reliability curve | Not available | No independent calibration and test fitting outcomes |
| Population or retailer fit guarantee | Not established | Chart matching is a deterministic estimate |

These are missing measurements, not zero-percent performance. Passing software tests does not measure garment fit or Pakistani generalisation.

`train_size.py` now requires separately supplied train, calibration and test CSVs plus source/licence/population/retailer/category/collection-protocol provenance. Every row requires a participant ID, unique fitting ID and independently confirmed physical-fitting label. It rejects synthetic/chart-derived origin declarations, duplicate fitting IDs and participant overlap across splits. Provenance declarations must still be audited by a person; code cannot prove that an attestation is truthful.

The future evaluation workflow fits RF and scaled KNN only on training data, selects temperature scaling using calibration log loss, and evaluates untouched test data once. It reports per-class precision/recall, balanced accuracy, confusion matrices, a training-majority baseline, raw/calibrated log loss, multiclass Brier score and ten-bin top-label reliability/ECE. Brier/log loss assess more than calibration alone; reliability bins must also be considered. This separation follows the principles in [scikit-learn calibration documentation](https://scikit-learn.org/stable/modules/calibration.html) and [grouped evaluation guidance](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data).

New schema-v2 artifacts are supported by the size endpoint but remain disabled after training until an administrator reviews and enables them. The deployed adapter is bound to category, retailer ID, inches/kg units, feature order and the exact chart SHA-256. It accepts the original model top choice only when available and chart-compatible; otherwise it falls back without carrying confidence over to the adjusted choice. Even a calibrated size-label probability is not automatically a probability of satisfaction or of a chart-adjusted choice. Before deployment, collect consented adult Pakistani fittings for the intended retailer and garment category, cover real cut/fabric variations, reserve new participants and a later collection period for external testing, assess subgroup/sample uncertainty, and evaluate the final deployed decision rule. Do not reuse the final test set to tune the model.

## Validation

Verified with the existing Anaconda Python **3.11.5** environment; installed packages were not replaced. Runtime versions: NumPy 2.4.6, scikit-learn 1.7.1, joblib 1.5.1, openpyxl 3.1.5. This records the tested local environment; no clean-machine Docker deployment was performed.

- 16 application tests passed, including authentication, ownership, CSV inventory, photo privacy, transparent PNG round trips and Module 1 inference contracts.
- 27 Module 2 tests passed: units, separate category inputs, legacy chart conversion, chart atomicity/persistence, optional fields, available-size constraints, out-of-range handling, no raw-measurement retention, provenance rejection and calibration math.
- 22 GPU/Module 1 regression tests passed.
- Existing catalog and generation-state JavaScript checks passed.
- Headless Edge exercised both category forms and live size API responses, a 390px mobile viewport, chart saving/reopening and zero browser JavaScript errors. Screenshots were visually inspected. A stale chart-editor state issue found during testing was fixed.

No new GPU weights were downloaded and no new real-garment inference run was needed for these sizing-only changes. Module 1's prior real-garment verification remains documented in README; its regression coverage passed here.

## Delivery

`AI-FIT-Module2-source.zip` contains the updated application, Module 1 service/source, tests, scripts, dataset sources and reports. `AI-FIT-Module2-datasets.zip` contains raw data, source metadata, attribution and the data audit. Private instance data, credentials, customer photos, Python environments and multi-GB model weights are excluded. Archive integrity and file hashes are verified separately in the delivery manifest.

## Integration follow-up and current limitations

The current follow-up passed 43 application/sizing tests (including model inference with artificial **software-test fixtures**, corrupt/legacy files, scope mismatch, absent calibration evidence and unavailable predicted sizes). The browser checked both category forms, live API results, chart persistence and 390px mobile layout. These fixtures are not a fitting dataset and their probabilities are not reported as real accuracy.

Additional source review: [UCSD clothing-fit datasets](https://cseweb.ucsd.edu/~jmcauley/datasets.html#clothing_fit) describe ModCloth/RentTheRunway feedback. They are not evidence of Pakistani retailer-specific physical fittings with this application's required measurements. [SIZER](https://virtualhumans.mpi-inf.mpg.de/sizer/) concerns size-sensitive 3D clothing; it was not established as a suitable drop-in dataset for this feature schema and population. No new suitable training cohort was verified or substituted. Existing permitted datasets remain reference-only. Consequently the delivered app continues to use the chart fallback with unavailable confidence.

## Installing a future reviewed model

1. Prepare separate independently labelled upper/lower train, calibration and test CSVs using the existing training schema. Retain participant separation and evidence for data permissions.
2. The provenance manifest's `retailer` must be the application's exact store ID. Add `chart_sha256`, computed by `sizing.chart_fingerprint(category_chart)` from that store's persisted chart, alongside the existing source/licence/population/collection fields. Feature values are inches and kg.
3. Run `python train_size.py train.csv --calibration calibration.csv --test test.csv --manifest provenance.json --output models/upper.joblib` (or lower). Review the adjacent metrics JSON, limitations and actual population/category coverage.
4. Administrator-installed, trusted artifacts only: after review, explicitly set the saved artifact's `production_enabled` to true using local joblib tooling. Never load downloaded/untrusted pickle/joblib files or accept model paths from user requests. Training does not automatically approve deployment.
5. Set `SIZE_MODEL_PATH` to the absolute directory containing `upper.joblib` and `lower.joblib`, then restart the app. A single artifact path is also supported for its matching category. Chart changes invalidate the matching hash and cause fallback until revalidation.

Confidence is temperature-scaled size-label model support, shown only with independent-test metadata. It is never renormalized across stock sizes or relabelled as fit satisfaction. A report's presence is not proof that its provenance is true: administrator review remains required. No trained production model or real accuracy percentage is included in this delivery.
