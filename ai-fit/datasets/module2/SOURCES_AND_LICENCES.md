# Module 2 dataset sources and permitted use

Downloaded 10 September 2026. Raw source files are unchanged. These are research reference data, not training labels or Pakistani clothing-size standards. Attribution below must travel with redistributed copies. Source licences do not imply endorsement of AI-FIT.

## Pakistan school-study workbook

File: `pone.0317534.s001.xlsx` (264,013 bytes).

Authors: Moazzam Tanveer, Ejaz Asghar, Georgian Badicu, Alexios Batrakoulis, Umar Tanveer, Serge Brand, Matheus Santos de Sousa Fernandes, Luca Paolo Ardigò (2025).

Title: S1 File — Effectiveness of a school-based physical activity intervention on overweight and obesity among children and adolescents in Pakistan.

Source: https://plos.figshare.com/articles/dataset/28475215
Metadata: https://api.figshare.com/v2/articles/28475215
Download: https://ndownloader.figshare.com/files/52573738
Licence: **CC BY 4.0**, https://creativecommons.org/licenses/by/4.0/
Legal text: https://creativecommons.org/licenses/by/4.0/legalcode
Metadata evidence: `pakistan-school-metadata.json` includes author list, licence, file size and publisher MD5. Download MD5 matched `aba332b1c1cc839b708e08c7422aa463`.

CC BY 4.0 permits sharing and adaptation, including commercial use, subject to attribution, linking the licence and indicating changes. No changes were made to the workbook. Our audit is separate and does not add size labels. No endorsement is claimed.

Inspection: one sheet, 1,200 nonempty data rows, 47 columns and 1,200 distinct source IDs. `Height_M` is in metres; `Weight_Kg` is kg. `Age` ranges 1–9 while `agemean` ranges 9–17, so age coding must not be assumed from column names. The source relates to schoolchildren; it does not establish adult shopper coverage. Pre/post measurement columns exist. No chest/bust, waist, hip, garment category or independent size/fit labels exist. It is unsuitable for the requested supervised sizing target. We did not infer missing circumferences or use BMI as a fit label.

## Pakistan DHS national anthropometry

File: `national-anthropometry.csv` (11,202 bytes).

Provider: Open Data Pakistan, National Center in Big Data and Cloud Computing / LUMS; underlying aggregated indicators credited to The DHS Program.

Source: https://opendata.com.pk/dataset/pakistan-demographic-and-health-data/resource/e647ddde-5ff8-4698-b208-408f87665ebf
Download: https://opendata.com.pk/dataset/ec728c14-2591-4dee-81e2-2459f28930dc/resource/e647ddde-5ff8-4698-b208-408f87665ebf/download/national-anthropometry.csv
Source licence label: **Creative Commons Attribution** (version not specified on the resource page).
Source licence link: https://opendefinition.org/licenses/cc-by/
That link describes CC Attribution and links to its versions, with 4.0 in the overview. We preserve the source's unspecified-version attribution rather than inventing a version grant. Attribute Open Data Pakistan and The DHS Program and retain the source licence link when sharing this unchanged file.

Inspection: 65 Pakistan indicator rows spanning PK1991DHS, PK2012DHS and PK2017DHS. One HXL metadata row and blank lines are excluded from that count. These are aggregate indicators about children and women, not individual observations. Their Value column mixes percentages, counts and standardised indices; it is not a body-measurement training feature. No fitting labels exist. The CSV's original line endings and HXL row are retained.

## Additional candidates researched, not included as raw data

- Muhammad Aslam (2020), Anthropometric-Data-Aged-2-19-Pak, version 1, DOI 10.17632/sxgymx5xjm.1. https://data.mendeley.com/datasets/sxgymx5xjm/1. CC BY 4.0 confirmed in DOI metadata (`mendeley-metadata.json`, obtained from https://api.datacite.org/dois/10.17632/sxgymx5xjm.1). Described population is 10,782 children/adolescents, ages 2–19, from four Pakistani cities. No clothing fit labels. Raw download was unavailable through attempted public endpoints (403 challenge / timeout); the archive contains metadata only, not its raw dataset. Article: https://pmc.ncbi.nlm.nih.gov/articles/PMC7749368/.
- Misra, Wan and McAuley, ModCloth / RentTheRunway fit feedback: https://rishabhmisra.github.io/Would-This-Clothing-Product-Fit-Me/ and https://cseweb.ucsd.edu/~jmcauley/datasets.html#clothing_fit. Real self-reported fit feedback exists, but it is not a Pakistani XS–XXL, retailer/category-labelled dataset with the required body measurements. Mirrored licence claims differ (Apache / CC BY); a sufficiently clear original redistribution grant was not established in this review. Not downloaded or relabelled as Pakistani data.
- SIZER: https://github.com/garvita-tiwari/sizer_dataset. Author repository requires a form and an emailed password for access. No request was sent and no gated data downloaded. Garment size annotations alone do not establish the confirmed best-fitting size for Pakistani shoppers.
- Generic recommended-size Kaggle datasets were not accepted merely because they expose a size column: independent physical-fitting provenance and Pakistan coverage were not established.

Research conclusion: no suitable permitted adult Pakistani fitting-labelled dataset was established in this search. This is a bounded search result, not a claim that no such dataset exists.

## Reproducibility

`scripts/download_module2_data.py` downloads the two supplied raw files from public source endpoints and verifies the workbook's publisher MD5. `scripts/audit_module2_data.py` checks raw row counts, fields, ranges, missing values and SHA-256 hashes without modifying data. The corresponding audit is in `reports/module2-data-audit.json`.
