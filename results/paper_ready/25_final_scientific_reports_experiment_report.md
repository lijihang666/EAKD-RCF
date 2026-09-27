# Scientific Reports: FINAL Formal Experiment Report

## A. Completeness and protocol

All paper results in this report originate from the frozen Scientific Reports formal experiment directory represented by this release. Nine fixed binary datasets, repeated stratified 5-fold CV, five repeats, and seeds 42-46 were used. Each dataset-method/configuration has 25 outer-test results. All 19 formal run directories passed the row-count, exact method, dataset, fold-key, seed, finite-metric, frozen-config and log checks: **True**. The equivalent Full methods pass cross-suite metric, threshold and test-probability checks: **True**. Smoke results are excluded.

The frozen method uses teacher predictive entropy for sample-wise temperature and training-fold reliability only for a shared `lambda_bar`. Diagnostic per-sample reliability labels are not student training weights. No outer-test score was used for method or threshold selection.

## B. Main comparison against 15 methods

The frozen ESTD-KD-RCF nine-dataset equal-weight means are:

| method | MCC | F1 | GMean | PRAUC | AUC | ECE | Brier |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ESTD-KD-RCF | 0.619641 | 0.725517 | 0.790733 | 0.797931 | 0.879973 | 0.079730 | 0.102995 |

Average ranks calculated on unrounded dataset means:

| metric | method | average_rank | n_datasets |
| --- | --- | --- | --- |
| MCC | ESTD-KD-RCF | 4.777778 | 9 |
| F1 | ESTD-KD-RCF | 3.555556 | 9 |
| GMean | ESTD-KD-RCF | 3.777778 | 9 |
| ECE | ESTD-KD-RCF | 7.222222 | 9 |
| Brier | ESTD-KD-RCF | 7.333333 | 9 |

Per-dataset rank among the 15 methods (unrounded values were ranked before formatting):

| dataset | MCC_rank | F1_rank | GMean_rank | ECE_rank | Brier_rank |
| --- | --- | --- | --- | --- | --- |
| BCWO | 6.000000 | 6.000000 | 7.000000 | 6.000000 | 6.000000 |
| Haberman | 5.000000 | 3.000000 | 1.000000 | 9.000000 | 12.000000 |
| WDBC | 1.000000 | 1.000000 | 2.000000 | 2.000000 | 5.000000 |
| WPBC | 9.000000 | 7.000000 | 5.000000 | 12.000000 | 2.000000 |
| Pima | 5.000000 | 1.000000 | 1.000000 | 10.000000 | 9.000000 |
| Sick | 2.000000 | 2.000000 | 6.000000 | 3.000000 | 3.000000 |
| Mammographic | 2.000000 | 2.000000 | 2.000000 | 10.000000 | 9.000000 |
| Heart | 5.000000 | 2.000000 | 1.000000 | 4.000000 | 11.000000 |
| Hepatitis | 8.000000 | 8.000000 | 9.000000 | 9.000000 | 9.000000 |

Complete baseline-by-baseline Win/Tie/Loss values are in `main_win_tie_loss_vs_ours.csv`. A rank or mean advantage is not asserted where the data do not show one.

## C. Stage contribution

Oriented gains below are target minus source for MCC/F1/GMean and the reverse for ECE/Brier; positive means improvement.

| transition | MCC_oriented_gain | F1_oriented_gain | GMean_oriented_gain | ECE_oriented_gain | Brier_oriented_gain |
| --- | --- | --- | --- | --- | --- |
| Delta_KD | 0.018756 | 0.005107 | 0.006886 | 0.007165 | 0.007335 |
| Delta_Fixed | 0.023781 | 0.018651 | 0.016243 | -0.002523 | 0.002503 |
| Delta_RCF | 0.019763 | 0.018892 | 0.017446 | 0.000780 | 0.001546 |

## D. Final component ablation

Full-minus-ablation oriented gains; negative means the ablated variant is better on that aggregate metric:

| ablation | MCC_oriented_gain | F1_oriented_gain | GMean_oriented_gain | ECE_oriented_gain | Brier_oriented_gain |
| --- | --- | --- | --- | --- | --- |
| ESTD-KD-RCF (w/o adaptive temperature) | 0.003963 | 0.004408 | 0.004608 | -0.000413 | 0.000222 |
| ESTD-KD-RCF (w/o global reliability KD strength) | 0.003968 | 0.006022 | 0.007455 | 0.000898 | 0.000269 |
| ESTD-KD-RCF (w/o Brier) | 0.004631 | 0.001808 | 0.000208 | -0.000171 | 0.000406 |
| ESTD-KD-RCF (fixed fusion) | 0.019763 | 0.018892 | 0.017446 | 0.000780 | 0.001546 |
| ESTD-KD-RCF (w/o candidate pool) | 0.013333 | 0.013033 | 0.013472 | -0.000255 | 0.000256 |

Win/Tie/Loss across nine dataset means:

| comparison | metric | win | tie | loss |
| --- | --- | --- | --- | --- |
| ESTD-KD-RCF (w/o adaptive temperature) | MCC | 7 | 0 | 2 |
| ESTD-KD-RCF (w/o adaptive temperature) | F1 | 6 | 0 | 3 |
| ESTD-KD-RCF (w/o adaptive temperature) | GMean | 6 | 0 | 3 |
| ESTD-KD-RCF (w/o adaptive temperature) | ECE | 4 | 0 | 5 |
| ESTD-KD-RCF (w/o adaptive temperature) | Brier | 5 | 0 | 4 |
| ESTD-KD-RCF (w/o global reliability KD strength) | MCC | 6 | 0 | 3 |
| ESTD-KD-RCF (w/o global reliability KD strength) | F1 | 6 | 0 | 3 |
| ESTD-KD-RCF (w/o global reliability KD strength) | GMean | 6 | 0 | 3 |
| ESTD-KD-RCF (w/o global reliability KD strength) | ECE | 7 | 0 | 2 |
| ESTD-KD-RCF (w/o global reliability KD strength) | Brier | 4 | 0 | 5 |
| ESTD-KD-RCF (w/o Brier) | MCC | 7 | 0 | 2 |
| ESTD-KD-RCF (w/o Brier) | F1 | 8 | 0 | 1 |
| ESTD-KD-RCF (w/o Brier) | GMean | 6 | 0 | 3 |
| ESTD-KD-RCF (w/o Brier) | ECE | 3 | 0 | 6 |
| ESTD-KD-RCF (w/o Brier) | Brier | 5 | 0 | 4 |
| ESTD-KD-RCF (fixed fusion) | MCC | 6 | 2 | 1 |
| ESTD-KD-RCF (fixed fusion) | F1 | 6 | 2 | 1 |
| ESTD-KD-RCF (fixed fusion) | GMean | 6 | 2 | 1 |
| ESTD-KD-RCF (fixed fusion) | ECE | 3 | 2 | 4 |
| ESTD-KD-RCF (fixed fusion) | Brier | 4 | 2 | 3 |
| ESTD-KD-RCF (w/o candidate pool) | MCC | 6 | 2 | 1 |
| ESTD-KD-RCF (w/o candidate pool) | F1 | 6 | 2 | 1 |
| ESTD-KD-RCF (w/o candidate pool) | GMean | 6 | 2 | 1 |
| ESTD-KD-RCF (w/o candidate pool) | ECE | 2 | 2 | 5 |
| ESTD-KD-RCF (w/o candidate pool) | Brier | 3 | 2 | 4 |

This table reports the evidence for adaptive temperature, global reliability-guided KD strength, student Brier regularization, RCF fusion and the candidate pool separately. The global-reliability ablation uses the prespecified midpoint `(lambda_min + lambda_max)/2`; no test-set tuning was performed.

## E. Yang and UMKD adaptations

`Yang-ATD-adapted` and `UMKD-UDD-adapted` implement transferable mechanisms for tabular binary classification; they are not claimed to reproduce every image-model component. Dataset and overall means appear in `12_yang_umkd_direct_comparison.csv`.

| comparison | metric | win | tie | loss |
| --- | --- | --- | --- | --- |
| Ours vs Yang-ATD-adapted | MCC | 6 | 0 | 3 |
| Ours vs Yang-ATD-adapted | F1 | 7 | 0 | 2 |
| Ours vs Yang-ATD-adapted | GMean | 7 | 0 | 2 |
| Ours vs Yang-ATD-adapted | ECE | 3 | 0 | 6 |
| Ours vs Yang-ATD-adapted | Brier | 3 | 0 | 6 |
| Ours vs UMKD-UDD-adapted | MCC | 7 | 0 | 2 |
| Ours vs UMKD-UDD-adapted | F1 | 8 | 0 | 1 |
| Ours vs UMKD-UDD-adapted | GMean | 7 | 0 | 2 |
| Ours vs UMKD-UDD-adapted | ECE | 4 | 0 | 5 |
| Ours vs UMKD-UDD-adapted | Brier | 6 | 0 | 3 |

## F. Entropy versus reliability

Low-entropy yet wrong teacher cases: **843**, or **1.2446%** of selected extreme-entropy OOF training observations, appearing in: BCWO, Haberman, Heart, Hepatitis, Mammographic, Pima, WPBC. High-entropy yet correct cases: **26434**, or **39.0273%**, appearing in: BCWO, Haberman, Heart, Hepatitis, Mammographic, Pima, Sick, WDBC, WPBC. These real cases support that low entropy does not guarantee correctness and high entropy does not imply incorrectness. Case IDs and probabilities are in `15_teacher_state_case_examples.csv`.

The final method does not apply a sample-wise reliability KD coefficient; reliability is aggregated only within the outer-training fold for `lambda_bar`.

## G. Student architecture

| method | MCC | F1 | GMean | ECE | Brier |
| --- | --- | --- | --- | --- | --- |
| ESTD-KD-RCF [Linear] | 0.614427 | 0.722477 | 0.787602 | 0.081701 | 0.103606 |
| ESTD-KD-RCF [MLP-32] | 0.620917 | 0.727884 | 0.792339 | 0.078197 | 0.103215 |
| ESTD-KD-RCF [MLP-64-32] | 0.619641 | 0.725517 | 0.790733 | 0.079730 | 0.102995 |
| ESTD-KD-RCF [MLP-128-64] | 0.613201 | 0.720136 | 0.786541 | 0.079726 | 0.103104 |

All four variants are reported. MLP-64-32 remains the frozen default regardless of other architectures' outer-test means; the table is a sensitivity analysis, not a model-selection step.

## H. RCF alpha bounds

All nine prespecified combinations are retained. Absolute max-minus-min spreads across nine-dataset equal-weight means:

| metric | minimum | maximum | absolute_range |
| --- | --- | --- | --- |
| MCC | 0.618509 | 0.619760 | 0.001251 |
| F1 | 0.724637 | 0.725588 | 0.000951 |
| GMean | 0.790390 | 0.790803 | 0.000413 |
| ECE | 0.079730 | 0.080614 | 0.000883 |
| Brier | 0.102995 | 0.103276 | 0.000281 |

The default remains alpha_min=0.50 and alpha_max=0.95.

## I. Core parameter stability

Only the historical one-factor grids were run. The tau=0.50 default reference comes from the frozen main run; it was not added as a new grid experiment.

| parameter | n_values | MCC_absolute_range | F1_absolute_range | GMean_absolute_range | ECE_absolute_range | Brier_absolute_range |
| --- | --- | --- | --- | --- | --- | --- |
| gamma | 4 | 0.010087 | 0.009875 | 0.009213 | 0.002325 | 0.000578 |
| lambda_max | 3 | 0.012525 | 0.011089 | 0.010576 | 0.001074 | 0.000622 |
| mu_brier | 3 | 0.004631 | 0.001914 | 0.002959 | 0.000981 | 0.000581 |
| reliability_tau | 3 | 0.008399 | 0.004466 | 0.003781 | 0.001012 | 0.000697 |

The six gamma-by-tau combinations are retained in `19_gamma_tau_sensitivity.csv`. No outer-test result changed the defaults.

## J. Statistical tests

Tests use only the 15 main-comparison methods, first averaging 25 outer tests within each dataset-method and then treating the nine datasets as blocks. Friedman results:

| metric | statistic | p_value | n_datasets | n_methods |
| --- | --- | --- | --- | --- |
| MCC | 37.300000 | 0.000664 | 9 | 15 |
| F1 | 35.033333 | 0.001453 | 9 | 15 |
| GMean | 30.355556 | 0.006816 | 9 | 15 |
| ECE | 10.455556 | 0.728177 | 9 | 15 |
| Brier | 53.866667 | 0.000001 | 9 | 15 |

Wilcoxon signed-rank versus each baseline with Holm correction is in `21_wilcoxon_holm.csv`. **0** adjusted comparisons reach p<0.05; all others **do not reach the 0.05 significance level**. The 225 folds are not treated as independent inferential units.

## K. Placement

Main text: dataset description, principal MCC/F1/GMean/ECE/Brier comparison, stage contribution, key ablation effects, concise entropy counterexamples, and corrected statistical conclusions. Supplementary Information: complete AUC/PRAUC results, all dataset-level ablations, architecture and alpha grids, core parameters, full statistical tests, runtimes, and teacher-state cases. The nine prepared supplementary tables and placement notes are in `11_supplementary_ready`.

## L. Manuscript readiness

`READY_FOR_MANUSCRIPT_REVISION = YES`

Readiness is based on experimental completeness and code/result consistency, **not** on achieving a desired performance ranking. No manuscript file was edited.
