from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from run_scirep_formal import ABLATION, ALPHA, ARCH, CORE, DATASETS, MAIN, METRICS, ROOT, RUNS, STAGE, audit


PAPER = ROOT / "10_paper_ready"
SUPP = ROOT / "11_supplementary_ready"
CORE_METRICS = ["MCC", "F1", "GMean", "ECE", "Brier"]
LOWER = {"ECE", "Brier"}
TIE = 1e-6
TOL = 1e-12
KEYS = ["dataset", "repeat_id", "fold_id", "seed"]
STATE_ORDER = [
    "low_entropy_correct", "low_entropy_wrong",
    "high_entropy_correct", "high_entropy_wrong",
]


def read_metrics(directory: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / directory / "metrics" / "all_metrics.csv")


def save(df: pd.DataFrame, directory: Path, filename: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    df.to_csv(directory / filename, index=False, encoding="utf-8-sig")


def by_dataset(df: pd.DataFrame) -> pd.DataFrame:
    result = df.groupby(["dataset", "method"], sort=False)[METRICS].agg(["mean", "std"])
    result.columns = [f"{metric}_{stat}" for metric, stat in result.columns]
    result = result.reset_index()
    result["n_outer_folds"] = df.groupby(["dataset", "method"], sort=False).size().to_numpy()
    for metric in METRICS:
        result[f"{metric}_mean_std"] = result.apply(
            lambda row: f"{row[f'{metric}_mean']:.4f} +/- {row[f'{metric}_std']:.4f}",
            axis=1,
        )
    return result


def dataset_means(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(["dataset", "method"], as_index=False, sort=False)[METRICS].mean()


def overall(df: pd.DataFrame) -> pd.DataFrame:
    means = dataset_means(df)
    result = means.groupby("method", as_index=False, sort=False)[METRICS].mean()
    result["n_datasets"] = means.groupby("method", sort=False).size().to_numpy()
    return result


def rank_by_dataset(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    for metric in METRICS:
        result[f"{metric}_rank"] = result.groupby("dataset")[f"{metric}_mean"].rank(
            ascending=metric in LOWER, method="average"
        )
        result[f"{metric}_placement"] = result[f"{metric}_rank"].map(
            lambda value: "best" if value == 1 else "second" if value == 2 else "other"
        )
    return result


def oriented(new: float, old: float, metric: str) -> float:
    raw = new - old
    return -raw if metric in LOWER else raw


def comparisons(
    means: pd.DataFrame,
    specs: list[tuple[str, str, str]],
    comparison_column: str,
) -> pd.DataFrame:
    rows = []
    for dataset in [*DATASETS, "ALL_9_EQUAL_WEIGHT"]:
        block = (
            means[means["dataset"] == dataset].set_index("method")
            if dataset != "ALL_9_EQUAL_WEIGHT"
            else means.groupby("method", sort=False)[METRICS].mean()
        )
        for name, old, new in specs:
            row = {
                "dataset": dataset,
                comparison_column: name,
                "old_method": old,
                "new_method": new,
            }
            for metric in CORE_METRICS:
                raw = float(block.loc[new, metric] - block.loc[old, metric])
                row[f"{metric}_raw_delta"] = raw
                row[f"{metric}_oriented_gain"] = -raw if metric in LOWER else raw
            rows.append(row)
    return pd.DataFrame(rows)


def win_tie_loss(
    means: pd.DataFrame, specs: list[tuple[str, str, str]]
) -> pd.DataFrame:
    block = means.set_index(["dataset", "method"])
    rows = []
    for label, old, new in specs:
        for metric in CORE_METRICS:
            gains = []
            for dataset in DATASETS:
                gains.append(oriented(
                    float(block.loc[(dataset, new), metric]),
                    float(block.loc[(dataset, old), metric]),
                    metric,
                ))
            arr = np.asarray(gains)
            rows.append({
                "comparison": label,
                "metric": metric,
                "win": int((arr > TIE).sum()),
                "tie": int((np.abs(arr) <= TIE).sum()),
                "loss": int((arr < -TIE).sum()),
                "tie_tolerance": TIE,
            })
    return pd.DataFrame(rows)


def write_main() -> tuple[pd.DataFrame, pd.DataFrame]:
    main = read_metrics("02_main_comparison")
    per_dataset = rank_by_dataset(by_dataset(main))
    main_overall = overall(main)
    save(per_dataset, PAPER, "02_main_comparison_by_dataset.csv")
    save(main_overall, PAPER, "03_main_comparison_overall.csv")
    source = ROOT / "02_main_comparison" / "metrics"
    shutil.copyfile(source / "statistical_average_ranks.csv", PAPER / "04_average_ranks.csv")
    return main, main_overall


def write_stage() -> pd.DataFrame:
    stage = read_metrics("03_stage_contribution")
    save(by_dataset(stage), PAPER, "05_stage_contribution_by_dataset.csv")
    save(overall(stage), PAPER, "06_stage_contribution_overall.csv")
    specs = [
        ("Delta_KD", "ESTD-Student", "ESTD-KD-S"),
        ("Delta_Fixed", "ESTD-KD-S", "ESTD-KD-RCF (fixed fusion)"),
        ("Delta_RCF", "ESTD-KD-RCF (fixed fusion)", "ESTD-KD-RCF"),
    ]
    delta = comparisons(dataset_means(stage), specs, "transition")
    save(delta, PAPER, "07_stage_contribution_delta.csv")
    return delta


def write_ablation() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = read_metrics("04_final_ablation")
    ablation_by_dataset = by_dataset(df)
    save(ablation_by_dataset, PAPER, "08_final_ablation_by_dataset.csv")
    save(overall(df), PAPER, "09_final_ablation_overall.csv")
    specs = [
        (method, method, "ESTD-KD-RCF") for method in ABLATION if method != "ESTD-KD-RCF"
    ]
    means = dataset_means(df)
    delta = comparisons(means, specs, "ablation")
    wtl = win_tie_loss(means, specs)
    save(delta, PAPER, "10_ablation_delta_vs_full.csv")
    save(wtl, PAPER, "11_ablation_win_tie_loss.csv")
    source = ROOT / "04_final_ablation" / "metrics"
    save(ablation_by_dataset, source, "final_ablation_by_dataset.csv")
    save(overall(df), source, "final_ablation_overall.csv")
    save(delta, source, "final_ablation_delta_vs_full.csv")
    save(wtl, source, "final_ablation_win_tie_loss.csv")
    return delta, wtl


def write_yang_umkd(main: pd.DataFrame) -> pd.DataFrame:
    keep = ["KD", "Yang-ATD-adapted", "UMKD-UDD-adapted", "ESTD-KD-RCF"]
    subset = main[main["method"].isin(keep)]
    means = dataset_means(subset)
    overall_rows = overall(subset).assign(dataset="ALL_9_EQUAL_WEIGHT")
    direct = pd.concat([means, overall_rows], ignore_index=True, sort=False)
    specs = [
        ("Ours vs Yang-ATD-adapted", "Yang-ATD-adapted", "ESTD-KD-RCF"),
        ("Ours vs UMKD-UDD-adapted", "UMKD-UDD-adapted", "ESTD-KD-RCF"),
    ]
    wtl = win_tie_loss(means, specs)
    save(direct, PAPER, "12_yang_umkd_direct_comparison.csv")
    save(wtl, PAPER, "13_yang_umkd_win_tie_loss.csv")
    source = ROOT / "02_main_comparison" / "metrics"
    save(means, source, "yang_umkd_by_dataset.csv")
    save(overall(subset), source, "yang_umkd_overall.csv")
    save(wtl, source, "yang_umkd_win_tie_loss.csv")
    return wtl


def write_teacher() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    source = ROOT / "02_main_comparison" / "metrics"
    samples = pd.read_csv(source / "teacher_state_samples.csv")
    selected = samples[samples["state"].isin(STATE_ORDER)].copy()
    if selected.empty:
        raise RuntimeError("No teacher extreme-state records")
    total = len(selected)
    aggregate = {
        "count": ("state", "size"),
        "minority_ratio": ("is_minority", "mean"),
        "mean_entropy": ("entropy", "mean"),
        "mean_reliability": ("reliability_true_class", "mean"),
        "mean_temperature": ("temperature_estd", "mean"),
        "mean_teacher_disagreement": ("teacher_disagreement_std", "mean"),
    }
    table = selected.groupby("state", as_index=False).agg(**aggregate)
    table["percentage"] = table["count"] / total * 100
    table["state"] = pd.Categorical(table["state"], STATE_ORDER, ordered=True)
    table = table.sort_values("state").reset_index(drop=True)
    per_dataset = selected.groupby(["dataset", "state"], as_index=False).agg(**aggregate)
    totals = selected.groupby("dataset").size()
    per_dataset["percentage_within_dataset_extremes"] = per_dataset.apply(
        lambda row: row["count"] / totals[row["dataset"]] * 100, axis=1
    )
    cases = (
        selected[selected["state"].isin(["low_entropy_wrong", "high_entropy_correct"])]
        .sort_values(["state", "dataset", "repeat_id", "fold_id", "original_row_id"])
        .groupby(["state", "dataset"], sort=False)
        .head(1)
    )
    case_columns = [
        "dataset", "repeat_id", "fold_id", "original_row_id", "y_true",
        "teacher_prob", "teacher_pred", "entropy", "reliability_true_class",
        "temperature_estd", "teacher_disagreement_std", "state",
    ]
    cases = cases[case_columns].rename(columns={
        "reliability_true_class": "diagnostic_reliability",
        "temperature_estd": "sample_temperature",
        "teacher_disagreement_std": "teacher_disagreement",
    })
    save(table, PAPER, "14_teacher_entropy_reliability_table.csv")
    save(cases, PAPER, "15_teacher_state_case_examples.csv")
    save(per_dataset, source, "teacher_entropy_reliability_by_dataset.csv")
    return table, per_dataset, cases


def write_architecture() -> pd.DataFrame:
    df = read_metrics("05_student_architecture")
    by = by_dataset(df).assign(scope="dataset")
    over = overall(df).assign(dataset="ALL_9_EQUAL_WEIGHT", scope="overall_equal_dataset")
    result = pd.concat([by, over], ignore_index=True, sort=False)
    result["is_frozen_default"] = result["method"].eq("ESTD-KD-RCF [MLP-64-32]")
    save(result, PAPER, "16_student_architecture_sensitivity.csv")
    return over


def write_alpha() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = read_metrics("06_alpha_sensitivity")
    by = by_dataset(df).assign(scope="dataset")
    over = overall(df).assign(dataset="ALL_9_EQUAL_WEIGHT", scope="overall_equal_dataset")
    result = pd.concat([by, over], ignore_index=True, sort=False)
    result["is_frozen_default"] = result["method"].eq("ESTD-KD-RCF [a0.50-0.95]")
    save(result, PAPER, "17_alpha_sensitivity.csv")
    ranges = []
    for metric in CORE_METRICS:
        values = over[metric]
        ranges.append({
            "metric": metric,
            "minimum": float(values.min()),
            "maximum": float(values.max()),
            "absolute_range": float(values.max() - values.min()),
        })
    range_df = pd.DataFrame(ranges)
    save(range_df, ROOT / "06_alpha_sensitivity" / "metrics", "alpha_bounds_range_summary.csv")
    return over, range_df


def write_core(main: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    blocks = []
    for name, flag, value in CORE:
        df = read_metrics(f"07_core_parameter_sensitivity/{name}")
        parameter = flag.lstrip("-")
        by = by_dataset(df).assign(scope="dataset")
        over = overall(df).assign(dataset="ALL_9_EQUAL_WEIGHT", scope="overall_equal_dataset")
        combined = pd.concat([by, over], ignore_index=True, sort=False)
        combined.insert(0, "parameter", parameter)
        combined.insert(1, "value", float(value))
        combined.insert(2, "source", "dedicated_parameter_run")
        blocks.append(combined)

    tau_default = overall(main[main["method"] == "ESTD-KD-RCF"]).assign(
        dataset="ALL_9_EQUAL_WEIGHT", scope="overall_equal_dataset",
        parameter="reliability_tau", value=0.5, source="main_frozen_default_reference"
    )
    blocks.append(tau_default)
    result = pd.concat(blocks, ignore_index=True, sort=False)
    save(result, PAPER, "18_core_parameter_sensitivity.csv")
    ranges = []
    grid_overall = result[
        (result["scope"] == "overall_equal_dataset")
        & (result["source"] == "dedicated_parameter_run")
    ]
    for parameter, block in grid_overall.groupby("parameter", sort=False):
        row = {"parameter": parameter, "n_values": len(block)}
        for metric in CORE_METRICS:
            row[f"{metric}_absolute_range"] = float(block[metric].max() - block[metric].min())
        ranges.append(row)
    return result, pd.DataFrame(ranges)


def write_gamma_tau() -> pd.DataFrame:
    df = read_metrics("08_gamma_tau_check")
    by = by_dataset(df).assign(scope="dataset")
    over = overall(df).assign(dataset="ALL_9_EQUAL_WEIGHT", scope="overall_equal_dataset")
    result = pd.concat([by, over], ignore_index=True, sort=False)
    parsed = result["method"].str.extract(r"g(?P<gamma>\d+p\d+)-t(?P<tau>\d+p\d+)")
    result.insert(2, "gamma", parsed["gamma"].str.replace("p", ".", regex=False).astype(float))
    result.insert(3, "reliability_tau", parsed["tau"].str.replace("p", ".", regex=False).astype(float))
    save(result, PAPER, "19_gamma_tau_sensitivity.csv")
    return over


def write_stats_runtime() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    source = ROOT / "02_main_comparison" / "metrics"
    friedman = pd.read_csv(source / "statistical_friedman.csv")
    wilcoxon = pd.read_csv(source / "statistical_wilcoxon_holm_vs_ours.csv")
    save(friedman, PAPER, "20_friedman_test.csv")
    save(wilcoxon, PAPER, "21_wilcoxon_holm.csv")
    runtime = pd.read_csv(source / "runtime_summary.csv")
    by = runtime.groupby(["dataset", "method"], as_index=False, sort=False).agg(
        runtime_mean_sec=("time_sec", "mean"),
        runtime_std_sec=("time_sec", "std"),
        n_outer_folds=("time_sec", "size"),
    ).assign(scope="dataset")
    over = by.groupby("method", as_index=False, sort=False).agg(
        runtime_mean_sec=("runtime_mean_sec", "mean"),
        runtime_std_across_dataset_means_sec=("runtime_mean_sec", "std"),
        n_datasets=("dataset", "nunique"),
    ).assign(dataset="ALL_9_EQUAL_WEIGHT", scope="overall_equal_dataset")
    paper = pd.concat([by, over], ignore_index=True, sort=False)
    save(paper, PAPER, "22_runtime_summary.csv")
    return friedman, wilcoxon, paper


def completion() -> tuple[pd.DataFrame, bool]:
    rows = []
    for spec in RUNS:
        result = audit(spec)
        checks = result["checks"]
        rows.append({
            "suite": spec.name,
            "expected_rows": result["expected_rows"],
            "actual_rows": result["actual_rows"],
            "expected_methods": len(spec.methods),
            "checks_passed": sum(checks.values()),
            "checks_total": len(checks),
            "status": result["status"],
            "details": "; ".join(f"{key}={value}" for key, value in checks.items()),
        })
    result = pd.DataFrame(rows)
    save(result, PAPER, "23_experiment_completion_check.csv")
    return result, bool((result["status"] == "PASS").all())


def cross_suite() -> tuple[pd.DataFrame, bool]:
    reference_suite = "03_stage_contribution"
    reference_method = "ESTD-KD-RCF"
    reference = read_metrics(reference_suite)
    reference = reference[reference["method"] == reference_method].set_index(KEYS).sort_index()
    ref_preds = pd.read_csv(
        ROOT / reference_suite / "preds" / "all_predictions.csv",
        usecols=[*KEYS, "method", "original_row_id", "y_true", "y_prob"],
    )
    ref_preds = ref_preds[ref_preds["method"] == reference_method].set_index(
        [*KEYS, "original_row_id", "y_true"]
    ).sort_index()
    comparisons_to_make = [
        ("04_final_ablation", "ESTD-KD-RCF"),
        ("05_student_architecture", "ESTD-KD-RCF [MLP-64-32]"),
        ("06_alpha_sensitivity", "ESTD-KD-RCF [a0.50-0.95]"),
    ]
    rows = []
    for suite, method in comparisons_to_make:
        cand = read_metrics(suite)
        cand = cand[cand["method"] == method].set_index(KEYS).sort_index()
        metric_keys_match = reference.index.equals(cand.index)
        preds = pd.read_csv(
            ROOT / suite / "preds" / "all_predictions.csv",
            usecols=[*KEYS, "method", "original_row_id", "y_true", "y_prob"],
        )
        preds = preds[preds["method"] == method].set_index(
            [*KEYS, "original_row_id", "y_true"]
        ).sort_index()
        pred_keys_match = ref_preds.index.equals(preds.index)
        for metric in [*CORE_METRICS, "threshold", "y_prob"]:
            if metric == "y_prob":
                same_keys = pred_keys_match
                diff = (ref_preds[metric] - preds[metric]).abs() if same_keys else pd.Series([np.inf])
            else:
                same_keys = metric_keys_match
                diff = (reference[metric] - cand[metric]).abs() if same_keys else pd.Series([np.inf])
            max_abs_diff = float(diff.max())
            rows.append({
                "reference_suite": reference_suite,
                "reference_method": reference_method,
                "comparison_suite": suite,
                "comparison_method": method,
                "metric": metric,
                "n_compared": len(diff) if same_keys else 0,
                "same_keys": same_keys,
                "max_abs_diff": max_abs_diff,
                "tolerance": TOL,
                "status": "PASS" if same_keys and max_abs_diff <= TOL else "FAIL",
            })
    result = pd.DataFrame(rows)
    save(result, PAPER, "24_cross_suite_consistency_check.csv")
    return result, bool((result["status"] == "PASS").all())


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in df[columns].iterrows():
        cells = []
        for column in columns:
            value = row[column]
            cells.append(f"{value:.6f}" if isinstance(value, (float, np.floating)) else str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_report(
    complete: bool,
    consistent: bool,
    main: pd.DataFrame,
    main_overall: pd.DataFrame,
    stage_delta: pd.DataFrame,
    ablation_delta: pd.DataFrame,
    ablation_wtl: pd.DataFrame,
    yang_wtl: pd.DataFrame,
    teacher_table: pd.DataFrame,
    teacher_by_dataset: pd.DataFrame,
    arch_overall: pd.DataFrame,
    alpha_ranges: pd.DataFrame,
    core_ranges: pd.DataFrame,
    friedman: pd.DataFrame,
    wilcoxon: pd.DataFrame,
) -> bool:
    ours = main_overall[main_overall["method"] == "ESTD-KD-RCF"]
    ranks = pd.read_csv(PAPER / "04_average_ranks.csv")
    ours_ranks = ranks[ranks["method"] == "ESTD-KD-RCF"]
    main_by_dataset = pd.read_csv(PAPER / "02_main_comparison_by_dataset.csv")
    ours_dataset = main_by_dataset[main_by_dataset["method"] == "ESTD-KD-RCF"]
    stage_overall = stage_delta[stage_delta["dataset"] == "ALL_9_EQUAL_WEIGHT"]
    ablation_overall = ablation_delta[ablation_delta["dataset"] == "ALL_9_EQUAL_WEIGHT"]
    state_counts = teacher_table.set_index("state")
    low_wrong = state_counts.loc["low_entropy_wrong"]
    high_correct = state_counts.loc["high_entropy_correct"]
    low_wrong_datasets = teacher_by_dataset[
        (teacher_by_dataset["state"] == "low_entropy_wrong")
        & (teacher_by_dataset["count"] > 0)
    ]["dataset"].tolist()
    high_correct_datasets = teacher_by_dataset[
        (teacher_by_dataset["state"] == "high_entropy_correct")
        & (teacher_by_dataset["count"] > 0)
    ]["dataset"].tolist()
    arch = arch_overall[["method", *CORE_METRICS]].copy()
    significant = wilcoxon[wilcoxon["p_holm"] < 0.05]
    ready = complete and consistent
    main_wtl = win_tie_loss(
        dataset_means(main),
        [(f"Ours vs {method}", method, "ESTD-KD-RCF") for method in MAIN if method != "ESTD-KD-RCF"],
    )
    save(main_wtl, PAPER, "main_win_tie_loss_vs_ours.csv")
    report = f"""# Scientific Reports: FINAL Formal Experiment Report

## A. Completeness and protocol

All paper results in this report originate from `{ROOT}`. Nine fixed binary datasets, repeated stratified 5-fold CV, five repeats, and seeds 42-46 were used. Each dataset-method/configuration has 25 outer-test results. All {len(RUNS)} formal run directories passed the row-count, exact method, dataset, fold-key, seed, finite-metric, frozen-config and log checks: **{complete}**. The equivalent Full methods pass cross-suite metric, threshold and test-probability checks: **{consistent}**. Smoke results are excluded.

The frozen method uses teacher predictive entropy for sample-wise temperature and training-fold reliability only for a shared `lambda_bar`. Diagnostic per-sample reliability labels are not student training weights. No outer-test score was used for method or threshold selection.

## B. Main comparison against 15 methods

The frozen ESTD-KD-RCF nine-dataset equal-weight means are:

{markdown_table(ours, ['method', *METRICS])}

Average ranks calculated on unrounded dataset means:

{markdown_table(ours_ranks, ['metric', 'method', 'average_rank', 'n_datasets'])}

Per-dataset rank among the 15 methods (unrounded values were ranked before formatting):

{markdown_table(ours_dataset, ['dataset', 'MCC_rank', 'F1_rank', 'GMean_rank', 'ECE_rank', 'Brier_rank'])}

Complete baseline-by-baseline Win/Tie/Loss values are in `main_win_tie_loss_vs_ours.csv`. A rank or mean advantage is not asserted where the data do not show one.

## C. Stage contribution

Oriented gains below are target minus source for MCC/F1/GMean and the reverse for ECE/Brier; positive means improvement.

{markdown_table(stage_overall, ['transition', *[f'{m}_oriented_gain' for m in CORE_METRICS]])}

## D. Final component ablation

Full-minus-ablation oriented gains; negative means the ablated variant is better on that aggregate metric:

{markdown_table(ablation_overall, ['ablation', *[f'{m}_oriented_gain' for m in CORE_METRICS]])}

Win/Tie/Loss across nine dataset means:

{markdown_table(ablation_wtl, ['comparison', 'metric', 'win', 'tie', 'loss'])}

This table reports the evidence for adaptive temperature, global reliability-guided KD strength, student Brier regularization, RCF fusion and the candidate pool separately. The global-reliability ablation uses the prespecified midpoint `(lambda_min + lambda_max)/2`; no test-set tuning was performed.

## E. Yang and UMKD adaptations

`Yang-ATD-adapted` and `UMKD-UDD-adapted` implement transferable mechanisms for tabular binary classification; they are not claimed to reproduce every image-model component. Dataset and overall means appear in `12_yang_umkd_direct_comparison.csv`.

{markdown_table(yang_wtl, ['comparison', 'metric', 'win', 'tie', 'loss'])}

## F. Entropy versus reliability

Low-entropy yet wrong teacher cases: **{int(low_wrong['count'])}**, or **{low_wrong['percentage']:.4f}%** of selected extreme-entropy OOF training observations, appearing in: {', '.join(low_wrong_datasets)}. High-entropy yet correct cases: **{int(high_correct['count'])}**, or **{high_correct['percentage']:.4f}%**, appearing in: {', '.join(high_correct_datasets)}. These real cases support that low entropy does not guarantee correctness and high entropy does not imply incorrectness. Case IDs and probabilities are in `15_teacher_state_case_examples.csv`.

The final method does not apply a sample-wise reliability KD coefficient; reliability is aggregated only within the outer-training fold for `lambda_bar`.

## G. Student architecture

{markdown_table(arch, ['method', *CORE_METRICS])}

All four variants are reported. MLP-64-32 remains the frozen default regardless of other architectures' outer-test means; the table is a sensitivity analysis, not a model-selection step.

## H. RCF alpha bounds

All nine prespecified combinations are retained. Absolute max-minus-min spreads across nine-dataset equal-weight means:

{markdown_table(alpha_ranges, ['metric', 'minimum', 'maximum', 'absolute_range'])}

The default remains alpha_min=0.50 and alpha_max=0.95.

## I. Core parameter stability

Only the historical one-factor grids were run. The tau=0.50 default reference comes from the frozen main run; it was not added as a new grid experiment.

{markdown_table(core_ranges, ['parameter', 'n_values', *[f'{m}_absolute_range' for m in CORE_METRICS]])}

The six gamma-by-tau combinations are retained in `19_gamma_tau_sensitivity.csv`. No outer-test result changed the defaults.

## J. Statistical tests

Tests use only the 15 main-comparison methods, first averaging 25 outer tests within each dataset-method and then treating the nine datasets as blocks. Friedman results:

{markdown_table(friedman, ['metric', 'statistic', 'p_value', 'n_datasets', 'n_methods'])}

Wilcoxon signed-rank versus each baseline with Holm correction is in `21_wilcoxon_holm.csv`. **{len(significant)}** adjusted comparisons reach p<0.05; all others **do not reach the 0.05 significance level**. The 225 folds are not treated as independent inferential units.

## K. Placement

Main text: dataset description, principal MCC/F1/GMean/ECE/Brier comparison, stage contribution, key ablation effects, concise entropy counterexamples, and corrected statistical conclusions. Supplementary Information: complete AUC/PRAUC results, all dataset-level ablations, architecture and alpha grids, core parameters, full statistical tests, runtimes, and teacher-state cases. The nine prepared supplementary tables and placement notes are in `11_supplementary_ready`.

## L. Manuscript readiness

`READY_FOR_MANUSCRIPT_REVISION = {'YES' if ready else 'NO'}`

Readiness is based on experimental completeness and code/result consistency, **not** on achieving a desired performance ranking. No manuscript file was edited.
"""
    (PAPER / "25_final_scientific_reports_experiment_report.md").write_text(report, encoding="utf-8")
    return ready


def write_supplementary() -> None:
    source = ROOT / "02_main_comparison" / "metrics"
    mappings = [
        (ROOT / "00_environment" / "dataset_manifest.csv", "Supplementary_Table_S1_dataset_details.csv"),
        (PAPER / "02_main_comparison_by_dataset.csv", "Supplementary_Table_S2_full_mean_std_all_metrics.csv"),
        (PAPER / "08_final_ablation_by_dataset.csv", "Supplementary_Table_S3_ablation_by_dataset.csv"),
        (PAPER / "16_student_architecture_sensitivity.csv", "Supplementary_Table_S4_student_architecture.csv"),
        (PAPER / "17_alpha_sensitivity.csv", "Supplementary_Table_S5_alpha_grid.csv"),
        (PAPER / "18_core_parameter_sensitivity.csv", "Supplementary_Table_S6_core_parameters.csv"),
        (PAPER / "22_runtime_summary.csv", "Supplementary_Table_S8_runtime.csv"),
        (PAPER / "15_teacher_state_case_examples.csv", "Supplementary_Table_S9_teacher_state_cases.csv"),
    ]
    SUPP.mkdir(parents=True, exist_ok=True)
    for src, name in mappings:
        shutil.copyfile(src, SUPP / name)
    friedman = pd.read_csv(PAPER / "20_friedman_test.csv").assign(test="Friedman")
    wilcoxon = pd.read_csv(PAPER / "21_wilcoxon_holm.csv").assign(test="Wilcoxon_Holm")
    save(pd.concat([friedman, wilcoxon], ignore_index=True, sort=False), SUPP, "Supplementary_Table_S7_statistical_tests.csv")
    notes = """# Supplementary Results Placement Notes

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
"""
    (SUPP / "supplementary_results_notes.md").write_text(notes, encoding="utf-8")


def verify_artifacts() -> None:
    paper_expected = [
        "01_dataset_summary.csv", "02_main_comparison_by_dataset.csv",
        "03_main_comparison_overall.csv", "04_average_ranks.csv",
        "05_stage_contribution_by_dataset.csv", "06_stage_contribution_overall.csv",
        "07_stage_contribution_delta.csv", "08_final_ablation_by_dataset.csv",
        "09_final_ablation_overall.csv", "10_ablation_delta_vs_full.csv",
        "11_ablation_win_tie_loss.csv", "12_yang_umkd_direct_comparison.csv",
        "13_yang_umkd_win_tie_loss.csv", "14_teacher_entropy_reliability_table.csv",
        "15_teacher_state_case_examples.csv", "16_student_architecture_sensitivity.csv",
        "17_alpha_sensitivity.csv", "18_core_parameter_sensitivity.csv",
        "19_gamma_tau_sensitivity.csv", "20_friedman_test.csv",
        "21_wilcoxon_holm.csv", "22_runtime_summary.csv",
        "23_experiment_completion_check.csv", "24_cross_suite_consistency_check.csv",
        "25_final_scientific_reports_experiment_report.md",
    ]
    supp_expected = [
        *(f"Supplementary_Table_S{i}_{name}.csv" for i, name in enumerate([
            "dataset_details", "full_mean_std_all_metrics", "ablation_by_dataset",
            "student_architecture", "alpha_grid", "core_parameters",
            "statistical_tests", "runtime", "teacher_state_cases",
        ], start=1)),
        "supplementary_results_notes.md",
    ]
    missing = [
        str(path) for folder, names in ((PAPER, paper_expected), (SUPP, supp_expected))
        for name in names
        if not (path := folder / name).is_file() or path.stat().st_size == 0
    ]
    if missing:
        raise RuntimeError(f"Missing or empty paper/supplementary artifacts: {missing}")


def main() -> None:
    PAPER.mkdir(parents=True, exist_ok=True)
    SUPP.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / "00_environment" / "dataset_manifest.csv", PAPER / "01_dataset_summary.csv")
    complete_df, complete = completion()
    main_df, main_overall = write_main()
    stage_delta = write_stage()
    ablation_delta, ablation_wtl = write_ablation()
    yang_wtl = write_yang_umkd(main_df)
    teacher_table, teacher_by_dataset, _ = write_teacher()
    arch_overall = write_architecture()
    _, alpha_ranges = write_alpha()
    _, core_ranges = write_core(main_df)
    write_gamma_tau()
    friedman, wilcoxon, _ = write_stats_runtime()
    cross, consistent = cross_suite()
    write_supplementary()
    ready = write_report(
        complete, consistent, main_df, main_overall, stage_delta, ablation_delta,
        ablation_wtl, yang_wtl, teacher_table, teacher_by_dataset, arch_overall,
        alpha_ranges, core_ranges, friedman, wilcoxon,
    )
    verify_artifacts()
    if not ready:
        raise RuntimeError(f"Not ready: complete={complete}; consistent={consistent}")
    print("SCIENTIFIC REPORTS FINAL FORMAL EXPERIMENTS COMPLETED")
    print(f"PAPER_READY={PAPER}")
    print(f"SUPPLEMENTARY_READY={SUPP}")
    print(f"FORMAL_SUITES={len(complete_df)}; CROSS_CHECK_ROWS={len(cross)}")


if __name__ == "__main__":
    main()
