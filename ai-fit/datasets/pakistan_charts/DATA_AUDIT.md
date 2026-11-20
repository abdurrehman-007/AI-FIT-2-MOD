# Current Pakistan chart audit — 23 September 2026

The active optional reference is `verified_presets.json`, built from the official [EshalPreet size guide](https://eshalpreet.com/pages/size-guide). `verified_upper_charts.csv` and `verified_lower_charts.csv` each contain 12 measurement rows: six sizes and two dimensions. `SOURCE_EVIDENCE.json` records attribution and the scope decision. The historic failed downloads below are preserved for traceability and never imported.

## Compatibility decision

The source explicitly identifies **body circumferences**, provides inches and rounded centimetres, and distinguishes general womenswear targets from finished garment measurements. Bust maps to chest. Upper uses chest/waist; Lower uses waist/hips. Original inch values are used; original cm values are retained, with rounding checked against 2.54 cm/in. No width-to-body or hem-to-waist conversion was invented.

This is a general brand-published womenswear reference, **not product-specific fitting validation, a menswear chart, or a Pakistani national standard**. It is exposed only by an explicit retailer action in Inventory → Edit size chart → Load womenswear reference for review → Save all charts. Loading does not persist until saved. Existing stores, product images and charts are unchanged. Retailers must check applicability to their products, fabric and intended ease. No fitting-confidence percentage comes from this chart.

The source does not establish an open dataset licence. Only numerical chart facts, attribution and brief evidence are redistributed, not copied page prose or images. See `SOURCE_EVIDENCE.json`. No source endorsement is implied.

[ZURI](https://zuriapparel.com/size-guide) was also reviewed: its page describes a baseline needing pattern-block checks before launch. That is insufficient evidence for verified retailer sizing; it was not integrated. Existing Sapphire half-width and failed lower pages below remain unsuitable. The new compatible body reference replaces their role as an integration candidate; the historic CSVs are not silently overwritten.

## Real fitting-labelled data search

- [UCSD ModCloth/RentTheRunway](https://cseweb.ucsd.edu/~jmcauley/datasets.html#clothing_fit) and the [dataset authors](https://github.com/rishabhmisra/Product-Catalog-Size-Recommendation-Framework) describe catalog fit feedback. This does not establish independent Pakistani physical fittings tied to the current retailer/category chart and required body features. No labels were remapped into invented XS–XXL outcomes.
- The [Pakistani fashion dataset card](https://huggingface.co/datasets/mohummadmahad/pakistani_fashion_dataset) exposes image/text pairs, useful for image tasks, without a confirmed size-fit target. It was not downloaded as a fitting dataset.
- Existing permitted anthropometric datasets in `../module2` remain reference-only; neither provides suitable garment fitting labels.

This bounded review did not establish suitable train/calibration/independent-test cohorts. RF/KNN code and loading support remain implemented, but no real production model or calibrated confidence is claimed. Source metadata and rejection decisions are retained in `sources_current.csv`.

---

## Historical downloaded-page audit (14 September 2026)

# Pakistan downloaded size-chart audit

Audited 2026-09-14. Scope: all three downloaded HTML files listed in `sources.csv`. Findings concern the saved HTML, not current live pages or HTTP response status (response headers were not saved).

## Source decisions

| File / source | Actual content and evidence | Extraction | Body-chart import |
| --- | --- | --- | --- |
| `sapphire_upper_waistcoat.html` — [source](https://pk.sapphireonline.pk/pages/waistcoats.html) | Title “Waist Coats”; `Page-Show` at line 592. Actual “Waist Coat” size table at lines 2002–2055, with 5 sizes and 4 measurement rows. This is a real chart, not a challenge/error page. | Usable with the basis limitations below: 20 upper rows. Generic category chart; no product-specific SKU supplied. | Unsuitable: half chest width is not a body circumference; no waist values or XXL. |
| `sapphire_lower_straight_trousers.html` — [requested source](https://pk.sapphireonline.pk/collections/woman/products/3PBEW24V420T-XSM-999.html) | Generic “Sapphire Online Store” title; `Search-Show`, `cgid=undefined` at line 614; “We are sorry, but no results were found for:” at line 2172. No table or actual product size chart. | Unusable no-results/search fallback page, not evidence of product dimensions. 0 rows. | Unsuitable. |
| `sapphire_lower_pull_on_trousers.html` — [requested source](https://pk.sapphireonline.pk/collections/pyjamas/products/SLBTMS25V207_999.html) | Generic title; `Search-Show`, `cgid=pyjamas` at line 614; same no-results message at line 2172. No table or actual product size chart. | Unusable no-results/search fallback page. 0 rows. | Unsuitable. |

No challenge-page content was identified. Generic CSS `.size-chart-collapsible`, cart-error elements and JavaScript error handlers do not establish either a size chart or a failed page. Lower classifications follow the main page action and no-results content. Product identifiers appearing only in requested URLs do not verify product content or measurements.

## Values, units and basis

The upper table explicitly marks its active tab **INCHES** (line 2005) and table class `active-inches` (line 2011). A CM toggle exists, but no separate original CM value table is present. The inactive toggle's contradictory `aria-selected=true` does not outweigh both matching active CSS indicators. All 20 source units are verified inches. There are **0 verified original cm values**, therefore no cm-to-inch conversions were performed. For a future verified cm source, divide by exactly 2.54 while retaining the original value, unit and measurement basis; unit conversion never changes width into circumference.

| Original measurement | XS / S / M / L / XL (inches) | Basis decision |
| --- | --- | --- |
| Length | 27.5 / 28.25 / 28.75 / 29.25 / 30 | Values and units transcribed; exact endpoints/method unspecified, so basis and overall verification are `unverified`. |
| Chest Half | 18.5 / 19.5 / 20.5 / 21.5 / 23 | Explicit half-chest width. Verified as a half width only; flat-lay method is not separately described. Never a full/body circumference. |
| Hem | 19.5 / 20.5 / 21.5 / 22.5 / 24 | Flat width versus full circumference unspecified: `unverified`. Do not equate hem to waist. |
| Shoulders | 16 / 16.5 / 17 / 17.5 / 18.25 | Exact endpoints and measurement method unspecified: `unverified`. |

`upper_charts.csv` contains one row per observed size/measurement, preserving original labels and numeric text. `lower_charts.csv` is intentionally header-only; no dummy rows, missing sizes or measurements are fabricated. Empty SKU means absent from the generic source, not a guessed SKU. `value_inches` preserves the source inch value even where basis is unverified; it does not confer body-import eligibility. `verification_status` is verified only for the explicit half-chest interpretation (5 rows); the other 15 rows remain unverified. All rows have `body_import_eligible=false`.

No relaxed or extended waist values occur in any usable source. The `waist_state` field is blank throughout. Future observed relaxed and extended values must have separate rows and distinct states, never be averaged or substituted for body waist.

## Integration decision and preservation

`sizing.py` defines upper required fields as chest/waist, lower as waist/hips, and requires XS through XXL with the same fields per size. Its public chart values are body measurements in inches. These downloads supply no suitable verified body chart for either category. Doubling a garment half width would still produce a garment circumference and would not establish body dimensions or wearing ease.

No app integration, chart replacement, model training, or category mapping was made. Module 1 and all existing application code/data remain untouched. No fitting accuracy or confidence claim is supported by these sources. Integration tests were not applicable because no integration was performed.

## Reproduction and checks

Run `python scripts/extract_pakistan_charts.py` from the project root. The extractor is deliberately limited to these audited sources and fails on unexpected table structure rather than guessing. It checks actual table labels, size alignment, inch indicators, positive numeric values, and the two no-results pages, then round-trips both CSVs. Expected totals: 20 upper observations, 0 lower observations, 0 body-import-eligible rows. Source URLs are copied from the local manifest; row-level source file and line evidence are included.
