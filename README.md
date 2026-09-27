# EAKD-RCF

Official implementation of **EAKD-RCF: Entropy-driven Adaptive Knowledge Distillation with Reliable Candidate Fusion** for imbalanced binary classification.

EAKD-RCF combines entropy-driven adaptive knowledge distillation with validation-driven reliable candidate selection. The frozen experiment code retains the historical internal identifier `ESTD-KD-RCF` in CSV files and command-line method names so that released results remain directly traceable to the original formal runs.

## Environment

The formal experiments were run with Python 3.13.4 and the package versions pinned in `requirements.txt`. The main dependencies are PyTorch, scikit-learn, imbalanced-learn, XGBoost, pandas, NumPy, SciPy, and Matplotlib.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements.txt
```

The recorded CPU environment and exact package versions are also preserved in `configs/environment_versions.txt`.

## Data

Raw public datasets are not redistributed. Download and prepare the nine datasets described in `DATA_SOURCES.md`, then place the required files in `data/`:

```text
data/
  bcwo.csv
  Haberman.data
  WDBC.data
  wpbc.data
  pima.csv
  sick.data
  mammographic_masses.data
  heart_disease_cleveland.data
  hepatitis.data
```

The nine datasets are BCWO, Haberman, WDBC, WPBC, Pima, Sick, Mammographic, Heart, and Hepatitis.

## Formal Protocol

- nine public imbalanced binary classification datasets
- repeated stratified five-fold cross-validation
- five repetitions
- 25 outer-test evaluations per dataset and method
- fixed repetition seeds 42, 43, 44, 45, and 46
- internal validation for candidate, calibration, and threshold selection
- outer test folds used only for final evaluation
- principal metrics: MCC, F1-score, G-Mean, ECE, and Brier score

The frozen main configuration is in `configs/formal_main_experiment.json`. The formal EAKD-RCF equal-dataset means are MCC 0.619641, F1-score 0.725517, G-Mean 0.790733, ECE 0.079730, and Brier 0.102995.

## Run

Run the complete main comparison with the exact formal harness:

```bash
python scripts/run_formal.py --phase main --data-dir data --output-root outputs
```

Run all formal suites, including stage contribution, ablation, architecture, alpha-bound, one-factor parameter, and joint gamma-tau studies:

```bash
python scripts/run_formal.py --phase all --data-dir data --output-root outputs
```

The equivalent direct command used by the main suite is:

```bash
python src/eakd_rcf_final.py --data_dir data --output_dir outputs/02_main_comparison --datasets all --formal --n_splits 5 --resume --with_balanced_rf --with_easy_ensemble --with_public_kd_methods --with_rcf_methods --methods "LR,RF,XGBoost,BalancedRF,EasyEnsemble,KD,Yang-ATD-adapted,BKD,CBD,CalibKD,CCKD,SRW-KD,UMKD-UDD-adapted,CB-DistillGrad,ESTD-KD-RCF"
```

Do not use outer-test scores for model, threshold, calibration, or candidate selection.

## Tables and Statistics

After the formal suites finish, generate paper-ready tables, Friedman tests, Wilcoxon signed-rank tests, and Holm corrections with:

```bash
set EAKD_RCF_OUTPUT_ROOT=outputs
python scripts/generate_tables_and_statistics.py
```

Reference paper-ready CSV files are included under `results/paper_ready/`. Statistical tests treat the nine dataset means as independent blocks; the 225 outer folds are not treated as independent inferential units.

## Figures

The current Matplotlib scripts for the finalized result figures are included:

```bash
set EAKD_RCF_OUTPUT_ROOT=outputs
set EAKD_RCF_FIGURE_DIR=figures
python scripts/generate_paper_figures.py
python scripts/generate_supplementary_s2.py
```

These scripts require the raw formal outputs under `outputs/`. Small source tables used by the confirmed layouts are included in `results/figure_source_tables/`. SVG output keeps text as text and contains no embedded raster image.

## Repository Layout

```text
src/       Frozen final experiment implementation
scripts/   Formal runner, table/statistics builder, and figure scripts
configs/   Frozen configuration, environment, and provenance
results/   Paper-ready summaries and compact figure source tables
```

## Reproducibility Notes

- The upstream frozen source SHA-256 is recorded in `configs/upstream_final_code_manifest.txt`.
- The release copy changes only machine-specific default paths to repository-relative defaults. Algorithm logic and experimental parameters are unchanged.
- Raw datasets, full prediction files, caches, checkpoints, partial outputs, logs, and historical experiment versions are intentionally excluded.
- No project license was present in the source workspace. A license should be selected before broader reuse.

