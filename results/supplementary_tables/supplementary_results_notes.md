# Supplementary Results Placement Notes

- Main text: compact dataset table, five principal metrics, stage contribution and the decisive component ablations.
- S1: nine dataset sizes, feature counts, class counts and imbalance ratios.
- S2: unabridged 25-fold mean/std for all 15 methods and seven metrics; rank labels use unrounded values.
- S3: six component-ablation conditions by dataset.
- S4: four student architectures; MLP-64-32 remains the frozen main-method default.
- S5: nine RCF alpha-bound combinations and full metrics.
- S6: all historical single-factor parameter values. The tau=0.50 row is a main-run reference, not an extra parameter experiment.
- S7: Friedman and Wilcoxon-Holm results from the main comparison only, using nine dataset means as independent blocks.
- S8: runtime mean/std by dataset-method and equal-dataset overall means.
- S9: deterministic real OOF teacher-state examples. Diagnostic reliability is not a per-sample KD training weight in the final method.
