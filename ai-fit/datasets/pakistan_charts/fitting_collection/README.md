# Real fitting collection protocol

These are empty templates, not a dataset. No fitting accuracy or calibrated confidence is established.

Recruit consenting adult participants for a specific retailer and category. Use pseudonymous person/fitting IDs; keep consent and identity records outside distributable datasets. Record measurements in inches, height in feet/inches and weight in kg. Upper needs chest/waist; Lower needs waist/hips. Record the best physically tried XS-XXL size after trying adjacent available sizes, with a documented assessor/wearer protocol; never infer labels from the chart. Keep garment SKU, fabric/cut, tried sizes, wearer feedback and assessor decision in a controlled study log keyed by fitting_id. A fit tag alone is not the confirmed size target used by this trainer.

Record provenance and chart fingerprint for the exact retailer/category. Do not set label_origin to independent_physical_fitting until that is true and audited. Blank templates intentionally fail training. Save collected personal measurements outside the source/dataset ZIP allowlist, for example in an access-controlled study location, not this templates directory.

Assign participants to separate train, calibration and independent test cohorts before tuning. Keep repeated fittings from one person within one cohort. Reserve later/new participants for the test cohort. Do not expand rows synthetically and report them as new people. The trainer execution minimum (five examples per size in every split) is not a power calculation or evidence of representativeness.

Use train_size.py with separate CSV files, --calibration, --test and --manifest, and an output path outside the distributable datasets directory. Review RF/KNN baselines, class metrics, reliability, sample uncertainty and chart fallback behavior. The model remains production-disabled until administrator review. Do not tune against the final test set or claim national Pakistani accuracy from a convenience sample.
