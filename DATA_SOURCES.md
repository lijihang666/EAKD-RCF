# Data Sources

Raw datasets are not redistributed in this repository. The experiment loader expects the exact local filenames listed below under `data/`. Verify the license and citation requirements on each source page before downloading.

| Experiment name | Required local filename | Public source | DOI / identifier | Status and preparation note |
|---|---|---|---|---|
| BCWO | `bcwo.csv` | [UCI Breast Cancer Wisconsin (Original)](https://archive.ics.uci.edu/dataset/15/breast+cancer+wisconsin+original) | `10.24432/C5HP4Z` | Source confirmed. The release loader expects a header-bearing CSV and the formal manifest contains 698 usable rows, while the UCI raw file contains 699 rows. Exact source-to-`bcwo.csv` preparation is **TO VERIFY**. |
| Haberman | `Haberman.data` | [UCI Haberman's Survival](https://archive.ics.uci.edu/dataset/43/haberman+s+survival) | `10.24432/C5XK51` | Source and filename confirmed. |
| WDBC | `WDBC.data` | [UCI Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) | `10.24432/C5DW2B` | Source confirmed. Rename `wdbc.data` to the case-sensitive local filename shown above when needed. |
| WPBC | `wpbc.data` | [UCI Breast Cancer Wisconsin (Prognostic)](https://archive.ics.uci.edu/dataset/16/breast+cancer+wisconsin+prognostic) | `10.24432/C5GK50` | Source and filename confirmed. |
| Pima | `pima.csv` | Common Pima Indians Diabetes Database, originally from NIDDK | **TO VERIFY** | The exact copy used by the formal run has 768 rows, eight predictors, and one binary label, with no header. The precise download URL and redistribution terms cannot be established from the current project files, so no URL or DOI is asserted. |
| Sick | `sick.data` | [UCI Thyroid Disease](https://archive.ics.uci.edu/dataset/102/thyroid+disease) | `10.24432/C5D010` | Source confirmed. Use the 2,800-row Sick training file with approximately 29 attributes; do not substitute the separate test file. |
| Mammographic | `mammographic_masses.data` | [UCI Mammographic Mass](https://archive.ics.uci.edu/dataset/161/mammographic+mass) | `10.24432/C53K6Z` | Source and filename confirmed. |
| Heart | `heart_disease_cleveland.data` | [UCI Heart Disease](https://archive.ics.uci.edu/dataset/45/heart+disease) | `10.24432/C52P4X` | Use the processed Cleveland subset with 303 rows and 13 predictors. Rename the downloaded file to the required local filename. |
| Hepatitis | `hepatitis.data` | [UCI Hepatitis](https://archive.ics.uci.edu/dataset/46/hepatitis) | `10.24432/C5Q59J` | Source and filename confirmed. |

## Formal Dataset Manifest

The exact sample counts, feature counts, class counts, imbalance ratios, and encoded labels observed by the frozen loader are preserved in `results/dataset_manifest.csv`. After preparing local files, compare loader output against that manifest before starting the full formal run.

No raw data file should be committed without separately reviewing its source license and the journal's data-availability requirements.

