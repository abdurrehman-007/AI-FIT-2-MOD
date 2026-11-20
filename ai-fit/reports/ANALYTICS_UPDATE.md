# Analytics update — 29 September 2026

One dashboard; no demo mode toggle. Only the sales count (26), order value (PKR 93,600), and seven-day sales graph use clearly labelled illustrative demo data. These samples are generated in memory and never saved as transactions. Their fixed seven-day scope is stated even when the real reporting period changes.

Try-ons, popularity, category share, size distribution, customer preferences, model performance, linked-order conversion, returns, garment tables and event history remain real retailer-scoped data. Demo sales never enter conversion or return-rate calculations. A chronological SVG line chart displays real try-on activity; a donut chart shows real category shares. Empty activity stays empty. No Buy button added.

Validation: 51 Python tests passed; JavaScript syntax checked. Desktop/mobile browser checks cover the single dashboard, labelled sample sales, graph rendering, enabled actual record workflow and mobile overflow. Source and dataset archives rebuilt with CRC and member SHA-256 verification.

## Presentation refinement
Removed Try-on to purchase. Sales cards show 26 orders, 32.5% conversion (26/80 illustrative try-ons), 7.7% returns (2/26). Nearby Sample sales metrics note identifies the scenario. Backend real metrics and records unchanged. Sales graph adds average order value.

Dashboard reordered into summary, activity/sales trends, garment/customer insights, category popularity, retail records and technical details. Responsive grids retain all records and controls. Browser desktop/mobile checks passed; desktop screenshot reviewed. Fixed punctuation encoding and fractional chart ticks found during review.
