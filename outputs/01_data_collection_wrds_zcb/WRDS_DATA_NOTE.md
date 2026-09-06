# WRDS CRSP Treasury exports

Raw WRDS/CRSP daily and monthly Treasury exports are **not** included in this
repository (licensing + size). On the original VM they lived as:

- `Yield/qz4vtnncykho8sgm.csv` (daily)
- `Yield/h3ktxzxmldfnwnbs.csv` (monthly)

Re-download from WRDS under your own license, then place them beside the
rebuild scripts or update paths in `scripts/01_data_collection_wrds_zcb/`.

Model-ready DNS panels under `dns_rebuild_output/` are included when available.
