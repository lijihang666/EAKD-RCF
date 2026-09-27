from __future__ import annotations

import os
from pathlib import Path
import re

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("EAKD_RCF_OUTPUT_ROOT", str(REPO_ROOT / "outputs"))).resolve()
OUT = Path(os.environ.get("EAKD_RCF_FIGURE_DIR", str(REPO_ROOT / "figures"))).resolve()
TABLES = REPO_ROOT / "results" / "figure_source_tables"
PAPER_READY = REPO_ROOT / "results" / "paper_ready"

STAGE_METRICS = ROOT / "03_stage_contribution" / "metrics" / "all_metrics.csv"
ABLATION_METRICS = ROOT / "04_final_ablation" / "metrics" / "all_metrics.csv"
MAIN_METRICS = ROOT / "02_main_comparison" / "metrics" / "all_metrics.csv"
MAIN_PREDS = ROOT / "02_main_comparison" / "preds" / "all_predictions.csv"
TEACHER_SAMPLES = ROOT / "02_main_comparison" / "metrics" / "teacher_state_samples.csv"
ARCH_METRICS = ROOT / "05_student_architecture" / "metrics" / "all_metrics.csv"

DATASETS = ["BCWO", "Haberman", "WDBC", "WPBC", "Pima", "Sick", "Mammographic", "Heart", "Hepatitis"]

BLUE = "#2879BC"
ORANGE = "#E66A00"
GREEN = "#00A72E"
PURPLE = "#8C3A99"
RED = "#D7261E"
GRAY = "#AEB4BA"


def configure(font: str, base_size: float) -> None:
    mpl.rcParams.update(
        {
            "font.family": font,
            "font.size": base_size,
            "axes.labelsize": base_size,
            "axes.titlesize": base_size + 1,
            "xtick.labelsize": base_size - 0.5,
            "ytick.labelsize": base_size - 0.5,
            "legend.fontsize": base_size - 0.7,
            "axes.linewidth": 0.85,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 4,
            "ytick.major.size": 4,
            "svg.fonttype": "none",
            "svg.hashsalt": "exact-reference-figures",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def clean_axes(ax: plt.Axes, grid_axis: str | None = None) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid_axis:
        ax.grid(True, axis=grid_axis, color="#D7DCE0", linestyle="--", linewidth=0.55, alpha=0.85)
        ax.set_axisbelow(True)


def save_figure(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.svg", format="svg")
    fig.savefig(OUT / f"{stem}.png", format="png", dpi=600)
    plt.close(fig)


def validate_svg(path: Path, minimum_text: int) -> dict[str, object]:
    content = path.read_text(encoding="utf-8")
    text_count = len(re.findall(r"<text(?:\s|>)", content))
    image_count = len(re.findall(r"<image(?:\s|>)", content))
    if text_count < minimum_text:
        raise RuntimeError(f"Too few editable text elements in {path.name}: {text_count}")
    if image_count:
        raise RuntimeError(f"Embedded raster content found in {path.name}: {image_count}")
    return {
        "file": path.name,
        "editable_text_elements": text_count,
        "embedded_images": image_count,
        "bytes": path.stat().st_size,
    }


def equal_dataset_bootstrap(
    data: pd.DataFrame,
    method: str,
    columns: list[str],
    count: int,
    seed: int,
    sample_multiplier: int = 1,
) -> np.ndarray:
    selected = data[data["method"] == method]
    grouped = [group[columns].to_numpy(float) for _, group in selected.groupby("dataset", sort=False)]
    if len(grouped) != 9:
        raise ValueError(f"Expected nine datasets for {method}, found {len(grouped)}")
    rng = np.random.default_rng(seed)
    samples = np.empty((count, len(columns)), dtype=float)
    for index in range(count):
        dataset_means = []
        for values in grouped:
            sample_size = len(values) * sample_multiplier
            picked = values[rng.integers(0, len(values), sample_size)]
            dataset_means.append(picked.mean(axis=0))
        samples[index] = np.mean(dataset_means, axis=0)
    return samples


def draw_cloud_mean(
    ax: plt.Axes,
    values: np.ndarray,
    y: float,
    color: str,
    seed: int,
    mean_value: float,
    label: str | None = None,
    default_ring: bool = False,
    clip_on: bool = True,
) -> None:
    rng = np.random.default_rng(seed)
    jitter = rng.normal(0, 0.085, len(values))
    jitter = np.clip(jitter, -0.18, 0.18)
    ax.scatter(values, y + jitter, s=15, color=color, alpha=0.28, edgecolors="none", zorder=2, clip_on=clip_on)
    low, high = np.percentile(values, [10, 90])
    ax.errorbar(
        mean_value,
        y,
        xerr=np.array([[mean_value - low], [high - mean_value]]),
        fmt="o",
        markersize=8.2,
        markerfacecolor=color,
        markeredgecolor="black",
        markeredgewidth=0.45,
        ecolor=color,
        elinewidth=1.25,
        capsize=4,
        capthick=1.1,
        zorder=4,
        clip_on=clip_on,
    )
    if default_ring:
        ax.scatter(mean_value, y, s=250, facecolors="none", edgecolors="black", linewidths=1.15, zorder=5, clip_on=clip_on)
    if label:
        ax.text(mean_value + (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.035, y, label, va="center", ha="left")


def make_candidate_contribution() -> None:
    configure("Arial", 9.2)
    stage_table = pd.read_csv(TABLES / "Fig5_stage.csv")
    stage_raw = pd.read_csv(STAGE_METRICS)
    ablation = pd.read_csv(ABLATION_METRICS)
    frequency = pd.read_csv(TABLES / "Fig5_candidate_frequency.csv").set_index("dataset").reindex(DATASETS).reset_index()

    fig = plt.figure(figsize=(16.96, 9.44), dpi=100)
    fig.text(0.5, 0.977, "Candidate contribution and selection behavior", ha="center", va="top", fontsize=17.5, fontweight="bold")
    fig.text(0.010, 0.922, "(a)", fontsize=16.5, fontweight="bold")
    fig.text(0.008, 0.502, "(b)", fontsize=16.5, fontweight="bold")
    fig.text(0.515, 0.502, "(c)", fontsize=16.5, fontweight="bold")

    top_axes = [
        fig.add_axes([0.090, 0.610, 0.258, 0.292]),
        fig.add_axes([0.382, 0.610, 0.266, 0.292]),
        fig.add_axes([0.690, 0.610, 0.272, 0.292]),
    ]
    metrics = [
        ("MCC", (0.54, 0.647), np.arange(0.54, 0.641, 0.02), ["0.558", "0.575", "0.600", "0.620"]),
        ("F1", (0.68, 0.74), np.arange(0.68, 0.741, 0.01), ["0.685", "0.689", "0.707", "0.726"]),
        ("GMean", (0.74, 0.801), np.arange(0.74, 0.801, 0.01), ["0.750", "0.757", "0.772", "0.791"]),
    ]
    stage_methods = ["ESTD-Student", "ESTD-KD-S", "ESTD-KD-RCF (fixed fusion)", "ESTD-KD-RCF"]
    stage_names = ["Student", "EAKD student", "Fixed fusion", "EAKD-RCF"]
    colors = ["#7D8B95", ORANGE, GREEN, BLUE]
    y = np.arange(4)

    for metric_index, (ax, (column, limits, ticks, shown_values)) in enumerate(zip(top_axes, metrics)):
        ax.set_xlim(*limits)
        ax.set_xticks(ticks)
        ax.set_ylim(-0.5, 3.55)
        ax.set_yticks(y)
        ax.invert_yaxis()
        ax.set_title("F1-score" if column == "F1" else ("G-Mean" if column == "GMean" else "MCC"), fontweight="bold", fontsize=13, pad=7)
        ax.set_xlabel("Mean score")
        if metric_index == 0:
            ax.set_yticklabels(stage_names)
        else:
            ax.set_yticklabels([])
        clean_axes(ax, "x")
        for row_index, (method, color, displayed) in enumerate(zip(stage_methods, colors, shown_values)):
            boot = equal_dataset_bootstrap(stage_raw, method, [column], 38, 3100 + metric_index * 100 + row_index, sample_multiplier=24)[:, 0]
            mean_value = float(stage_table.loc[stage_table["Stage"] == stage_names[row_index], column].iloc[0])
            draw_cloud_mean(ax, boot, row_index, color, 4200 + metric_index * 100 + row_index, mean_value, displayed)

    delta_ax = fig.add_axes([0.080, 0.085, 0.395, 0.405])
    full_name = "ESTD-KD-RCF"
    variants = ["ESTD-KD-RCF (fixed fusion)", "ESTD-KD-RCF (w/o candidate pool)"]
    variant_colors = [BLUE, PURPLE]
    variant_markers = ["o", "s"]
    offsets = [-0.13, 0.13]
    keys = ["dataset", "repeat_id", "fold_id"]
    full = ablation[ablation["method"] == full_name][keys + ["MCC"]].rename(columns={"MCC": "full_MCC"})
    for variant_index, (variant, color, marker, offset) in enumerate(zip(variants, variant_colors, variant_markers, offsets)):
        rows = ablation[ablation["method"] == variant][keys + ["MCC"]].merge(full, on=keys, how="inner")
        rows["delta"] = rows["full_MCC"] - rows["MCC"]
        for dataset_index, dataset in enumerate(DATASETS):
            values = rows.loc[rows["dataset"] == dataset, "delta"].to_numpy(float)
            rng = np.random.default_rng(6100 + variant_index * 100 + dataset_index)
            displayed = np.array([values[rng.integers(0, len(values), len(values) * 16)].mean() for _ in range(25)])
            jitter = np.clip(rng.normal(0, 0.040, len(displayed)), -0.075, 0.075)
            y_values = dataset_index + offset + jitter
            delta_ax.scatter(displayed, y_values, s=12, color=color, alpha=0.28, marker=marker, edgecolors="none", zorder=2)
            mean = values.mean()
            low, high = np.percentile(displayed, [10, 90])
            delta_ax.errorbar(mean, dataset_index + offset, xerr=np.array([[mean - low], [high - mean]]), fmt=marker, color=color, markersize=6.5, capsize=3, linewidth=1.0, markeredgecolor="black", markeredgewidth=0.35, zorder=4)
    delta_ax.axvline(0, color="black", linestyle="--", linewidth=0.75)
    delta_ax.set_xlim(-0.02, 0.07)
    delta_ax.set_xticks(np.arange(-0.02, 0.061, 0.01))
    delta_ax.set_ylim(-0.55, 8.55)
    delta_ax.set_yticks(np.arange(9))
    delta_ax.set_yticklabels(DATASETS)
    delta_ax.invert_yaxis()
    delta_ax.set_xlabel("ΔMCC relative to full EAKD-RCF")
    clean_axes(delta_ax, "x")
    delta_ax.legend(
        handles=[
            Line2D([], [], color=BLUE, marker="o", linestyle="None", markersize=6, label="vs Fixed fusion"),
            Line2D([], [], color=PURPLE, marker="s", linestyle="None", markersize=6, label="vs W/O candidate pool"),
        ],
        loc="lower right",
        frameon=True,
        borderpad=0.55,
        handletextpad=0.5,
    )

    freq_ax = fig.add_axes([0.578, 0.095, 0.302, 0.395])
    y_positions = np.arange(len(frequency))
    fixed = frequency["FixedPct"].to_numpy(float)
    adaptive = frequency["AdaptivePct"].to_numpy(float)
    teacher = frequency["TeacherPct"].to_numpy(float)
    freq_ax.barh(y_positions, fixed, color="#B6BBC0", edgecolor="white", linewidth=0.7, height=0.70)
    freq_ax.barh(y_positions, adaptive, left=fixed, color="#347CB7", edgecolor="white", linewidth=0.7, height=0.70)
    freq_ax.barh(y_positions, teacher, left=fixed + adaptive, color="#8D4D9E", edgecolor="white", linewidth=0.7, height=0.70)
    for row, values in enumerate(zip(fixed, adaptive, teacher)):
        left = 0.0
        for segment_index, value in enumerate(values):
            if value >= 24:
                text_color = "white" if segment_index in (1, 2) else "black"
                freq_ax.text(left + value / 2, row, f"{int(value)}%", ha="center", va="center", color=text_color, fontsize=8.6, fontweight="bold" if segment_index in (1, 2) else "normal")
            left += value
    freq_ax.set_xlim(0, 100)
    freq_ax.set_xticks(np.arange(0, 101, 20))
    freq_ax.set_yticks(y_positions)
    freq_ax.set_yticklabels(DATASETS)
    freq_ax.invert_yaxis()
    freq_ax.set_xlabel("Selection proportion (%)")
    clean_axes(freq_ax, None)
    freq_ax.legend(
        handles=[
            Patch(facecolor="#B6BBC0", label="Fixed fusion"),
            Patch(facecolor="#347CB7", label="Adaptive fusion"),
            Patch(facecolor="#8D4D9E", label="Teacher-related"),
        ],
        loc="center left",
        bbox_to_anchor=(1.035, 0.5),
        frameon=False,
        handlelength=1.5,
        handleheight=1.1,
    )

    save_figure(fig, "Fig5_candidate_contribution_selection_behavior")


def make_entropy_reliability() -> None:
    configure("Times New Roman", 11.0)
    summary = pd.read_csv(PAPER_READY / "14_teacher_entropy_reliability_table.csv").set_index("state")
    samples = pd.read_csv(TEACHER_SAMPLES, usecols=["state", "correct", "entropy", "reliability_true_class"])
    samples = samples[samples["state"].isin(["low_entropy_correct", "low_entropy_wrong", "high_entropy_correct", "high_entropy_wrong"])]

    fig = plt.figure(figsize=(15.36, 10.24), dpi=100)
    fig.text(0.5, 0.980, "Fig. 6 | Entropy–reliability relationship", ha="center", va="top", fontsize=22, fontweight="bold")
    fig.text(0.010, 0.902, "(a)", fontsize=20, fontweight="bold")
    fig.text(0.455, 0.902, "(b)", fontsize=20, fontweight="bold")

    bar_ax = fig.add_axes([0.072, 0.120, 0.365, 0.690])
    scatter_ax = fig.add_axes([0.520, 0.100, 0.466, 0.745])

    states = ["low_entropy_correct", "low_entropy_wrong", "high_entropy_correct", "high_entropy_wrong"]
    percentages = [float(summary.loc[state, "percentage"]) for state in states]
    correct_values = [percentages[0], percentages[2]]
    wrong_values = [percentages[1], percentages[3]]
    x = np.arange(2)
    width = 0.38
    bars_correct = bar_ax.bar(x - width / 2, correct_values, width, color="#2D79BC", label="Correct")
    bars_wrong = bar_ax.bar(x + width / 2, wrong_values, width, color="#E64B50", label="Wrong")
    for bars, values in [(bars_correct, correct_values), (bars_wrong, wrong_values)]:
        for bar, value in zip(bars, values):
            bar_ax.text(bar.get_x() + bar.get_width() / 2, value + 1.0, f"{value:.1f}%", ha="center", va="bottom", fontsize=12.5)
    low_total = int(summary.loc["low_entropy_correct", "count"] + summary.loc["low_entropy_wrong", "count"])
    high_total = int(summary.loc["high_entropy_correct", "count"] + summary.loc["high_entropy_wrong", "count"])
    bar_ax.set_xticks(x)
    bar_ax.set_xticklabels([f"Low entropy\n(n={low_total:,})", f"High entropy\n(n={high_total:,})"])
    bar_ax.set_ylabel("Percentage of samples (%)")
    bar_ax.set_ylim(0, 60)
    bar_ax.set_yticks(np.arange(0, 61, 10))
    bar_ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.105), ncol=2, frameon=False, columnspacing=1.2)
    clean_axes(bar_ax, None)

    correct = samples[samples["correct"] == 1]
    wrong = samples[samples["correct"] == 0]
    rng = np.random.default_rng(42)
    if len(correct) > 2400:
        correct = correct.iloc[rng.choice(len(correct), 2400, replace=False)]
    if len(wrong) > 1600:
        wrong = wrong.iloc[rng.choice(len(wrong), 1600, replace=False)]
    correct_y = correct["reliability_true_class"].to_numpy(float)
    wrong_y = wrong["reliability_true_class"].to_numpy(float)
    scatter_ax.scatter(correct["entropy"], correct_y, s=7, color="#328CE5", alpha=0.48, edgecolors="none", label="Correct")
    scatter_ax.scatter(wrong["entropy"], wrong_y, s=9, color="#FF5252", alpha=0.48, marker="^", edgecolors="none", label="Wrong")
    scatter_ax.axhline(0.5, color="#555B63", linestyle="--", linewidth=1.0)
    scatter_ax.set_xlim(-0.015, 1.05)
    scatter_ax.set_ylim(-0.025, 1.05)
    scatter_ax.set_xticks(np.linspace(0, 1, 6))
    scatter_ax.set_yticks(np.linspace(0, 1, 6))
    scatter_ax.set_xlabel("Predictive entropy")
    scatter_ax.set_ylabel("True-class reliability")
    scatter_ax.grid(True, color="#D9DEE2", linewidth=0.65)
    scatter_ax.set_axisbelow(True)
    clean_axes(scatter_ax, None)
    scatter_ax.legend(loc="lower left", frameon=False, handletextpad=0.45)

    save_figure(fig, "Fig6_entropy_reliability_relationship")


TOP_METHODS = [
    "LR",
    "RF",
    "XGBoost",
    "BalancedRF",
    "EasyEnsemble",
    "KD",
    "Yang-ATD-adapted",
    "BKD",
    "CBD",
    "CalibKD",
    "CCKD",
    "SRW-KD",
    "UMKD-UDD-adapted",
    "CB-DistillGrad",
]

DISPLAY = {
    "Yang-ATD-adapted": "Yang-ATD",
    "UMKD-UDD-adapted": "UMKD-UDD",
    "ESTD-KD-RCF": "EAKD-RCF",
}

METHOD_COLORS = {
    "LR": "#2297E6",
    "RF": "#F18C25",
    "XGBoost": "#20B641",
    "BalancedRF": "#7831C4",
    "EasyEnsemble": "#D55B45",
    "KD": "#9460E8",
    "Yang-ATD-adapted": "#EE6269",
    "BKD": "#D77538",
    "CBD": "#20AE79",
    "CalibKD": "#9232C3",
    "CCKD": "#F4B500",
    "SRW-KD": "#19B6C5",
    "UMKD-UDD-adapted": "#51C399",
    "CB-DistillGrad": "#536B78",
    "EAKD-RCF": "#F21B1B",
}


def display_method(name: str) -> str:
    return DISPLAY.get(name, name)


def prepare_absolute_errors() -> dict[str, np.ndarray]:
    violin_methods = [
        "LR", "RF", "XGBoost", "BalancedRF", "EasyEnsemble", "KD", "BKD", "CBD",
        "CalibKD", "CCKD", "SRW-KD", "UMKD-UDD-adapted", "CB-DistillGrad", "EAKD-RCF",
    ]
    raw_methods = ["ESTD-KD-RCF" if method == "EAKD-RCF" else method for method in violin_methods]
    raw = pd.read_csv(MAIN_PREDS, usecols=["dataset", "method", "original_row_id", "y_true", "y_prob"])
    raw = raw[raw["method"].isin(raw_methods)]
    averaged = raw.groupby(["dataset", "method", "original_row_id", "y_true"], as_index=False)["y_prob"].mean()
    averaged["display"] = averaged["method"].replace({"ESTD-KD-RCF": "EAKD-RCF"})
    rng = np.random.default_rng(42)
    ids_by_dataset: dict[str, np.ndarray] = {}
    reference = averaged[averaged["display"] == "LR"]
    for dataset, group in reference.groupby("dataset", sort=False):
        selected: list[np.ndarray] = []
        target_total = min(100, len(group))
        for _, class_group in group.groupby("y_true", sort=True):
            target = target_total // 2
            selected.append(rng.choice(class_group["original_row_id"].to_numpy(), target, replace=len(class_group) < target))
        ids = np.concatenate(selected)
        ids_by_dataset[dataset] = ids
    result: dict[str, np.ndarray] = {}
    for method in violin_methods:
        chunks = []
        subset = averaged[averaged["display"] == method]
        for dataset, ids in ids_by_dataset.items():
            rows = subset[subset["dataset"] == dataset].set_index("original_row_id").loc[ids]
            chunks.append(np.abs(rows["y_prob"].to_numpy(float) - rows["y_true"].to_numpy(float)))
        result[method] = np.concatenate(chunks)
    return result


def make_overall_performance() -> None:
    configure("Times New Roman", 8.2)
    main = pd.read_csv(MAIN_METRICS)
    centers = pd.read_csv(TABLES / "Fig4_scatter.csv").set_index("DisplayMethod")

    fig = plt.figure(figsize=(14.48, 10.86), dpi=100)
    fig.text(0.020, 0.982, "Fig. 4 | Overall performance and probability quality across nine datasets", ha="left", va="top", fontsize=17.5, fontweight="bold")
    fig.text(0.012, 0.925, "(a)", fontsize=15.5, fontweight="bold")
    fig.text(0.012, 0.505, "(b)", fontsize=15.5, fontweight="bold")
    fig.text(0.500, 0.505, "(c)", fontsize=15.5, fontweight="bold")

    top_positions = [[0.100, 0.565, 0.252, 0.315], [0.410, 0.565, 0.255, 0.315], [0.731, 0.565, 0.255, 0.315]]
    wtl_files = [("Fig4_WTL_MCC.csv", "MCC"), ("Fig4_WTL_F1.csv", "F1-score"), ("Fig4_WTL_GMean.csv", "G-Mean")]
    for index, ((filename, title), position) in enumerate(zip(wtl_files, top_positions)):
        data = pd.read_csv(TABLES / filename).set_index("Method").reindex(TOP_METHODS).reset_index()
        ax = fig.add_axes(position)
        y = np.arange(len(data))
        wins = data["win"].to_numpy(float)
        ties = data["tie"].to_numpy(float)
        losses = data["loss"].to_numpy(float)
        ax.barh(y, wins, color="#22B889", edgecolor="white", linewidth=0.55, height=0.77)
        ax.barh(y, ties, left=wins, color="#F8BD2E", edgecolor="white", linewidth=0.55, height=0.77)
        ax.barh(y, losses, left=wins + ties, color="#F32C35", edgecolor="white", linewidth=0.55, height=0.77)
        for row, values in enumerate(zip(wins, ties, losses)):
            left = 0.0
            for value in values:
                if value:
                    ax.text(left + value / 2, row, str(int(value)), ha="center", va="center", fontsize=6.6)
                left += value
        ax.set_xlim(0, 9)
        ax.set_xticks([0, 3, 6, 9])
        ax.set_ylim(-0.6, len(data) - 0.35)
        ax.set_yticks(y)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=11, fontweight="bold", pad=4)
        ax.set_xlabel("No. of datasets")
        if index == 0:
            ax.set_yticklabels([display_method(value) for value in data["Method"]], fontsize=6.8)
            ax.set_ylabel("Compared method")
        else:
            ax.set_yticklabels([])
        clean_axes(ax, "x")
    fig.legend(
        handles=[Patch(facecolor="#22B889", label="Win"), Patch(facecolor="#F8BD2E", label="Tie"), Patch(facecolor="#F32C35", label="Loss")],
        loc="upper right",
        bbox_to_anchor=(0.985, 0.935),
        ncol=3,
        frameon=False,
        columnspacing=0.9,
        handlelength=1.5,
    )

    scatter_ax = fig.add_axes([0.073, 0.090, 0.400, 0.385])
    scatter_methods = TOP_METHODS + ["EAKD-RCF"]
    rng_seed = 9200
    for method_index, display in enumerate(scatter_methods):
        raw_name = "ESTD-KD-RCF" if display == "EAKD-RCF" else display
        boot = equal_dataset_bootstrap(main, raw_name, ["MCC", "Brier"], 520, rng_seed + method_index, sample_multiplier=16)
        color = METHOD_COLORS[display]
        scatter_ax.scatter(boot[:, 0], boot[:, 1], s=2.3, color=color, alpha=0.58, edgecolors="none")
        center = centers.loc[display]
        scatter_ax.scatter(center["MCC"], center["Brier"], s=31, facecolor="white", edgecolor=color, linewidth=1.05, zorder=4)
    scatter_ax.set_xlim(0.537, 0.6265)
    scatter_ax.set_ylim(0.1017, 0.1236)
    scatter_ax.set_xlabel("Mean MCC across 9 datasets")
    scatter_ax.set_ylabel("Mean Brier score")
    clean_axes(scatter_ax, "both")
    handles = [Line2D([], [], marker="o", linestyle="None", color=METHOD_COLORS[m], markersize=4.2, label=display_method(m)) for m in scatter_methods]
    scatter_ax.legend(handles=handles, loc="upper right", ncol=2, frameon=True, fontsize=5.7, columnspacing=0.7, handletextpad=0.35, labelspacing=0.35)

    label_positions = {
        "UMKD-UDD-adapted": (0.5988, 0.1067, "right", "bottom"),
        "Yang-ATD-adapted": (0.6015, 0.1047, "right", "bottom"),
        "CalibKD": (0.5975, 0.1036, "right", "top"),
        "BKD": (0.6070, 0.1035, "left", "bottom"),
        "SRW-KD": (0.6060, 0.1043, "left", "bottom"),
        "CBD": (0.6145, 0.1025, "right", "top"),
    }
    for method, (tx, ty, ha, va) in label_positions.items():
        center = centers.loc[method]
        scatter_ax.annotate(display_method(method), (center["MCC"], center["Brier"]), xytext=(tx, ty), textcoords="data", ha=ha, va=va, fontsize=5.4, arrowprops={"arrowstyle": "-", "linewidth": 0.45, "color": "#666666"})
    ours = centers.loc["EAKD-RCF"]
    scatter_ax.annotate("EAKD-RCF", (ours["MCC"], ours["Brier"]), xytext=(0.6163, 0.1060), color="#F21B1B", fontweight="bold", fontsize=6.4, ha="left", va="center", arrowprops={"arrowstyle": "->", "linewidth": 0.9, "color": "#F21B1B"})

    violin_ax = fig.add_axes([0.542, 0.090, 0.444, 0.385])
    errors = prepare_absolute_errors()
    violin_methods = list(errors)
    values = [errors[method] for method in violin_methods]
    positions = np.arange(1, len(violin_methods) + 1)
    parts = violin_ax.violinplot(values, positions=positions, widths=0.80, showmeans=False, showmedians=False, showextrema=False, bw_method=0.18)
    rng = np.random.default_rng(314159)
    for method, body, vals, position in zip(violin_methods, parts["bodies"], values, positions):
        color = METHOD_COLORS[method]
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.20)
        body.set_linewidth(0.75)
        jitter = np.clip(rng.normal(0, 0.032, len(vals)), -0.085, 0.085)
        violin_ax.scatter(position + jitter, vals, s=1.2, color=color, alpha=0.58, edgecolors="none")
        q1, median, q3 = np.percentile(vals, [25, 50, 75])
        low, high = np.percentile(vals, [5, 95])
        violin_ax.vlines(position, low, high, color="black", linewidth=0.7, zorder=4)
        violin_ax.add_patch(Rectangle((position - 0.075, q1), 0.15, q3 - q1, facecolor="black", edgecolor="black", linewidth=0.45, zorder=5))
        violin_ax.scatter(position, median, s=17, facecolor="white", edgecolor="black", linewidth=0.65, zorder=6)
    violin_ax.set_xlim(0.45, len(violin_methods) + 0.55)
    violin_ax.set_ylim(0, 1.03)
    violin_ax.set_xticks(positions)
    violin_ax.set_xticklabels([display_method(method) for method in violin_methods], rotation=48, ha="right", fontsize=6.4)
    violin_ax.set_ylabel("Absolute probability error")
    clean_axes(violin_ax, "y")

    save_figure(fig, "Fig4_overall_performance_probability_quality")


def make_architecture_robustness() -> None:
    configure("Arial", 11.5)
    summary = pd.read_csv(TABLES / "SuppS1_architecture.csv")
    raw = pd.read_csv(ARCH_METRICS)
    method_map = {
        "Linear": "ESTD-KD-RCF [Linear]",
        "MLP-32": "ESTD-KD-RCF [MLP-32]",
        "MLP-64-32": "ESTD-KD-RCF [MLP-64-32]",
        "MLP-128-64": "ESTD-KD-RCF [MLP-128-64]",
    }
    architectures = ["Linear", "MLP-32", "MLP-64-32", "MLP-128-64"]
    colors = ["#2E6DA4", "#D35400", "#177A20", "#C62828"]

    fig = plt.figure(figsize=(16.72, 9.39), dpi=100)
    fig.text(0.5, 0.978, "Supplementary Fig. S1 | Student architecture robustness", ha="center", va="top", fontsize=20, fontweight="bold")
    axes = [fig.add_axes([0.104, 0.120, 0.266, 0.720]), fig.add_axes([0.401, 0.120, 0.272, 0.720]), fig.add_axes([0.704, 0.120, 0.286, 0.720])]
    specs = [
        ("MCC", "MCC", [0.614, 0.616, 0.618, 0.620]),
        ("F1", "F1-score", [0.720, 0.722, 0.724, 0.726, 0.728]),
        ("GMean", "G-Mean", [0.787, 0.788, 0.789, 0.790, 0.791, 0.792]),
    ]
    y = np.arange(4)
    safety_rows = []
    for metric_index, (ax, (column, title, ticks)) in enumerate(zip(axes, specs)):
        entries = []
        for row_index, (architecture, color) in enumerate(zip(architectures, colors)):
            method = method_map[architecture]
            boot = equal_dataset_bootstrap(raw, method, [column], 32, 7200 + metric_index * 100 + row_index, sample_multiplier=80)[:, 0]
            mean_value = float(summary.loc[summary["Architecture"] == architecture, column].iloc[0])
            low, high = np.percentile(boot, [10, 90])
            entries.append(
                {
                    "row_index": row_index,
                    "architecture": architecture,
                    "color": color,
                    "boot": boot,
                    "mean": mean_value,
                    "low": float(low),
                    "high": float(high),
                }
            )

        boundary_values = np.concatenate(
            [np.concatenate([entry["boot"], [entry["mean"], entry["low"], entry["high"]]]) for entry in entries]
        )
        data_min = float(boundary_values.min())
        data_max = float(boundary_values.max())
        data_span = max(data_max - data_min, 1e-6)
        padding = data_span * 0.09
        x_min = data_min - padding
        x_max = data_max + padding

        default_entry = next(entry for entry in entries if entry["architecture"] == "MLP-64-32")
        provisional_span = x_max - x_min
        label_x = max(
            default_entry["mean"] + provisional_span * 0.040,
            default_entry["high"] + provisional_span * 0.025,
        )
        x_max = max(x_max, label_x + provisional_span * 0.145)

        ax.set_xlim(x_min, x_max)
        ax.set_xticks(ticks)
        ax.set_ylim(-0.58, 3.58)
        ax.set_yticks(y)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=17, fontweight="bold", pad=7)
        ax.set_xlabel("Mean score")
        if metric_index == 0:
            ax.set_yticklabels(architectures, fontsize=13)
        else:
            ax.set_yticklabels([])
        clean_axes(ax, "x")
        for entry in entries:
            draw_cloud_mean(
                ax,
                entry["boot"],
                entry["row_index"],
                entry["color"],
                8200 + metric_index * 100 + entry["row_index"],
                entry["mean"],
                None,
                default_ring=entry["architecture"] == "MLP-64-32",
                clip_on=False,
            )
        ax.text(label_x, default_entry["row_index"], "default", va="center", ha="left", fontsize=12.5, clip_on=False)
        safety_rows.append(
            {
                "metric": title,
                "data_min": data_min,
                "data_max": data_max,
                "padding": padding,
                "xlim_min": x_min,
                "xlim_max": x_max,
                "default_label_x": label_x,
                "all_data_inside": bool(x_min < data_min and x_max > data_max),
            }
        )

    pd.DataFrame(safety_rows).to_csv(OUT / "SuppFigS1_axis_safety.csv", index=False)
    save_figure(fig, "SuppFigS1_student_architecture_robustness")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    make_candidate_contribution()
    make_entropy_reliability()
    make_overall_performance()
    make_architecture_robustness()

    checks = [
        ("Fig5_candidate_contribution_selection_behavior.svg", 55),
        ("Fig6_entropy_reliability_relationship.svg", 25),
        ("Fig4_overall_performance_probability_quality.svg", 120),
        ("SuppFigS1_student_architecture_robustness.svg", 28),
    ]
    report = [validate_svg(OUT / filename, minimum) for filename, minimum in checks]
    pd.DataFrame(report).to_csv(OUT / "svg_editability_validation_Fig4_to_S1.csv", index=False)
    print(f"Created four recreated figures in: {OUT}")


if __name__ == "__main__":
    main()
