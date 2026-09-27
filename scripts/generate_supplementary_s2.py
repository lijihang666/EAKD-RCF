from __future__ import annotations

import os
from pathlib import Path
import re

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("EAKD_RCF_OUTPUT_ROOT", str(REPO_ROOT / "outputs"))).resolve()
OUT = Path(os.environ.get("EAKD_RCF_FIGURE_DIR", str(REPO_ROOT / "figures"))).resolve()
DEFAULT_METRICS = ROOT / "02_main_comparison" / "metrics" / "all_metrics.csv"
ALPHA_METRICS = ROOT / "06_alpha_sensitivity" / "metrics" / "all_metrics.csv"
CORE_ROOT = ROOT / "07_core_parameter_sensitivity"
JOINT_METRICS = ROOT / "08_gamma_tau_check" / "metrics" / "all_metrics.csv"

KEYS = ["dataset", "repeat_id", "fold_id"]
DEFAULT_METHOD = "ESTD-KD-RCF"
Y_TICKS = np.arange(-0.012, 0.0021, 0.002)
OUTPUT_STEM = "Supplementary_Fig_S2_parameter_sensitivity_PPTsafe"


def configure() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 10.5,
            "axes.labelsize": 11.5,
            "axes.titlesize": 14.0,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9.5,
            "axes.linewidth": 0.9,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.width": 0.85,
            "ytick.major.width": 0.85,
            "xtick.major.size": 4.2,
            "ytick.major.size": 4.2,
            "svg.fonttype": "none",
            "svg.hashsalt": "supplementary-s2-pptsafe",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def load_metrics(path: Path, method: str | None = None) -> pd.DataFrame:
    data = pd.read_csv(path, usecols=KEYS + ["method", "MCC"])
    if method is not None:
        data = data[data["method"] == method].copy()
    if data.empty:
        raise ValueError(f"No rows found in {path} for method={method}")
    return data


def equal_dataset_mean(data: pd.DataFrame) -> float:
    return float(data.groupby("dataset", sort=False)["MCC"].mean().mean())


def compact_bootstrap_delta(
    data: pd.DataFrame,
    default_mean: float,
    seed: int,
    count: int = 24,
    sample_multiplier: int = 72,
) -> np.ndarray:
    grouped = [group["MCC"].to_numpy(float) for _, group in data.groupby("dataset", sort=False)]
    if len(grouped) != 9:
        raise ValueError(f"Expected nine datasets, found {len(grouped)}")
    rng = np.random.default_rng(seed)
    draws = np.empty(count, dtype=float)
    for draw_index in range(count):
        dataset_means = []
        for values in grouped:
            size = len(values) * sample_multiplier
            picked = values[rng.integers(0, len(values), size)]
            dataset_means.append(float(picked.mean()))
        draws[draw_index] = float(np.mean(dataset_means) - default_mean)
    return draws


def style_axis(ax: plt.Axes) -> None:
    ax.set_yticks(Y_TICKS)
    ax.axhline(0.0, color="#858585", linestyle="--", linewidth=0.85, zorder=1)
    ax.grid(True, color="#D9DDE1", linewidth=0.55, alpha=0.85)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylabel("ΔMCC from default")


def draw_setting(
    ax: plt.Axes,
    x: float,
    data: pd.DataFrame,
    default_mean: float,
    color: str,
    marker: str,
    seed: int,
    x_span: float,
    default: bool = False,
) -> dict[str, float]:
    center = equal_dataset_mean(data) - default_mean
    draws = compact_bootstrap_delta(data, default_mean, seed)
    rng = np.random.default_rng(seed + 10000)
    x_jitter = np.clip(rng.normal(0, x_span * 0.009, len(draws)), -x_span * 0.020, x_span * 0.020)
    ax.scatter(
        x + x_jitter,
        draws,
        s=17,
        color=color,
        alpha=0.27,
        marker=marker,
        edgecolors="none",
        zorder=2,
        clip_on=False,
    )
    spread = float(np.std(draws, ddof=1))
    low, high = np.percentile(draws, [10, 90])
    ax.errorbar(
        x,
        center,
        yerr=spread,
        fmt=marker,
        markersize=8.0,
        markerfacecolor=color,
        markeredgecolor=color,
        markeredgewidth=0.6,
        ecolor=color,
        elinewidth=1.25,
        capsize=4,
        capthick=1.1,
        zorder=4,
        clip_on=False,
    )
    if default:
        ax.scatter(
            x,
            center,
            s=245,
            facecolors="none",
            edgecolors="#242424",
            linewidths=1.25,
            zorder=5,
            clip_on=False,
        )
    return {
        "x": x,
        "delta_mcc": center,
        "bootstrap_sd": spread,
        "bootstrap_p10": float(low),
        "bootstrap_p90": float(high),
        "scatter_x_min": float(np.min(x + x_jitter)),
        "scatter_x_max": float(np.max(x + x_jitter)),
        "scatter_y_min": float(np.min(draws)),
        "scatter_y_max": float(np.max(draws)),
        "error_y_min": float(center - spread),
        "error_y_max": float(center + spread),
        "is_default": bool(default),
    }


def core_data(folder: str) -> pd.DataFrame:
    return load_metrics(CORE_ROOT / folder / "metrics" / "all_metrics.csv", DEFAULT_METHOD)


def save_figure(fig: plt.Figure) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{OUTPUT_STEM}.svg", format="svg")
    fig.savefig(OUT / f"{OUTPUT_STEM}.png", format="png", dpi=600)
    plt.close(fig)


def validate_svg(path: Path) -> None:
    content = path.read_text(encoding="utf-8")
    text_count = len(re.findall(r"<text(?:\s|>)", content))
    image_count = len(re.findall(r"<image(?:\s|>)", content))
    if text_count < 65:
        raise RuntimeError(f"Too few editable text elements: {text_count}")
    if image_count:
        raise RuntimeError(f"Embedded raster elements found: {image_count}")
    pd.DataFrame(
        [
            {
                "file": path.name,
                "editable_text_elements": text_count,
                "embedded_images": image_count,
                "bytes": path.stat().st_size,
            }
        ]
    ).to_csv(OUT / "svg_editability_validation_S2_PPTsafe.csv", index=False)


def finalize_safe_axes(
    axes: np.ndarray,
    panel_rows: dict[str, list[dict[str, float]]],
    panel_defaults: dict[str, dict[str, float]],
) -> list[dict[str, object]]:
    """Add data-driven margins so PPT does not need clipping to hide overflow."""
    all_y_bounds: list[float] = []
    for rows in panel_rows.values():
        for row in rows:
            all_y_bounds.extend(
                [
                    row["scatter_y_min"],
                    row["scatter_y_max"],
                    row["error_y_min"],
                    row["error_y_max"],
                ]
            )

    y_data_min = min(all_y_bounds)
    y_data_max = max(all_y_bounds)
    y_span = max(y_data_max - y_data_min, 1e-9)
    y_padding = y_span * 0.10
    y_limits = (y_data_min - y_padding, y_data_max + y_padding)

    safety_rows: list[dict[str, object]] = []
    for panel, ax in zip("abcdef", axes.flat):
        rows = panel_rows[panel]
        x_data_min = min(min(row["scatter_x_min"], row["x"]) for row in rows)
        x_data_max = max(max(row["scatter_x_max"], row["x"]) for row in rows)
        x_span = max(x_data_max - x_data_min, 1e-9)
        x_padding = x_span * 0.10
        x_min = x_data_min - x_padding
        x_max = x_data_max + x_padding

        default_info = panel_defaults.get(panel)
        label_x = np.nan
        label_y = np.nan
        if default_info is not None:
            label_x = default_info["x"] + x_span * 0.030
            label_y = default_info["delta_mcc"] + y_span * 0.030
            # Reserve enough horizontal room for the editable seven-letter label.
            x_max = max(x_max, default_info["x"] + x_span * 0.165)
            ax.text(
                label_x,
                label_y,
                "default",
                ha="left",
                va="bottom",
                fontsize=10.0,
                clip_on=False,
                zorder=6,
            )

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(*y_limits)
        safety_rows.append(
            {
                "panel": panel,
                "x_data_min": x_data_min,
                "x_data_max": x_data_max,
                "x_padding": x_padding,
                "xlim_min": x_min,
                "xlim_max": x_max,
                "y_data_min": y_data_min,
                "y_data_max": y_data_max,
                "y_padding": y_padding,
                "ylim_min": y_limits[0],
                "ylim_max": y_limits[1],
                "default_label_x": label_x,
                "default_label_y": label_y,
                "all_data_inside": bool(
                    x_data_min > x_min
                    and x_data_max < x_max
                    and y_data_min > y_limits[0]
                    and y_data_max < y_limits[1]
                ),
            }
        )
    return safety_rows


def main() -> None:
    configure()
    OUT.mkdir(parents=True, exist_ok=True)

    default = load_metrics(DEFAULT_METRICS, DEFAULT_METHOD)
    default_mean = equal_dataset_mean(default)
    alpha = load_metrics(ALPHA_METRICS)
    joint = load_metrics(JOINT_METRICS)

    fig, axes = plt.subplots(2, 3, figsize=(14.80, 10.86), dpi=100)
    fig.subplots_adjust(left=0.083, right=0.985, top=0.875, bottom=0.095, wspace=0.36, hspace=0.46)
    fig.suptitle("Supplementary Fig. S2 — Parameter sensitivity analysis", fontsize=18.0, fontweight="bold", y=0.975)

    panel_letters = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]
    panel_titles = [
        "RCF mixing bounds",
        "Entropy exponent γ",
        "KD strength λ_max",
        "Brier weight μ_B",
        "Reliability threshold τ",
        "Joint γ–τ",
    ]
    for ax, letter, title in zip(axes.flat, panel_letters, panel_titles):
        style_axis(ax)
        ax.set_title(title, fontweight="bold", pad=12)
        ax.text(-0.30, 1.095, letter, transform=ax.transAxes, fontsize=16.0, fontweight="bold", ha="left", va="center")

    source_rows: list[dict[str, object]] = []
    panel_rows: dict[str, list[dict[str, float]]] = {panel: [] for panel in "abcdef"}
    panel_defaults: dict[str, dict[str, float]] = {}

    def record(panel: str, row: dict[str, float], metadata: dict[str, object]) -> None:
        panel_rows[panel].append(row)
        if row["is_default"]:
            panel_defaults[panel] = row
        source_rows.append({"panel": panel, **metadata, **row})

    # (a) RCF mixing bounds
    ax = axes[0, 0]
    alpha_mins = [0.10, 0.30, 0.50]
    alpha_maxes = [0.70, 0.90, 0.95]
    alpha_markers = ["o", "s", "^"]
    alpha_colors = ["#3296F0", "#237BD1", "#0B559F"]
    alpha_offsets = [-0.012, 0.0, 0.012]
    for series_index, (alpha_min, marker, color, offset) in enumerate(zip(alpha_mins, alpha_markers, alpha_colors, alpha_offsets)):
        for value_index, alpha_max in enumerate(alpha_maxes):
            method = f"ESTD-KD-RCF [a{alpha_min:.2f}-{alpha_max:.2f}]"
            subset = alpha[alpha["method"] == method]
            row = draw_setting(
                ax,
                alpha_max + offset,
                subset,
                default_mean,
                color,
                marker,
                1100 + series_index * 100 + value_index,
                0.29,
                default=alpha_min == 0.50 and alpha_max == 0.95,
            )
            record(
                "a",
                row,
                {"series": f"α_min={alpha_min:.2f}", "parameter": "α_max", "actual_x": alpha_max},
            )
    ax.set_xticks(alpha_maxes)
    ax.set_xlabel("α_max")
    ax.legend(
        handles=[
            Line2D([], [], marker=marker, linestyle="None", color=color, markersize=7, label=f"α_min = {value:.2f}")
            for value, marker, color in zip(alpha_mins, alpha_markers, alpha_colors)
        ],
        loc="lower left",
        bbox_to_anchor=(0.105, 0.015),
        frameon=True,
        framealpha=0.72,
        facecolor="white",
        edgecolor="#E1E1E1",
    )

    # (b) Entropy exponent gamma
    ax = axes[0, 1]
    gamma_values = [0.5, 1.0, 2.0, 3.0]
    gamma_folders = ["gamma_0p5", "gamma_1p0", "gamma_2p0", "gamma_3p0"]
    for index, (value, folder) in enumerate(zip(gamma_values, gamma_folders)):
        row = draw_setting(ax, value, core_data(folder), default_mean, "#FF6F22", "o", 2100 + index, 2.8, default=value == 2.0)
        record("b", row, {"series": "gamma", "parameter": "γ"})
    ax.set_xticks(gamma_values)
    ax.set_xlabel("γ")

    # (c) KD strength lambda_max
    ax = axes[0, 2]
    lambda_values = [0.3, 0.5, 0.7]
    lambda_folders = ["lambda_max_0p3", "lambda_max_0p5", "lambda_max_0p7"]
    for index, (value, folder) in enumerate(zip(lambda_values, lambda_folders)):
        row = draw_setting(ax, value, core_data(folder), default_mean, "#12A62C", "o", 3100 + index, 0.46, default=value == 0.5)
        record("c", row, {"series": "lambda_max", "parameter": "λ_max"})
    ax.set_xticks(lambda_values)
    ax.set_xlabel("λ_max")

    # (d) Brier weight mu_B
    ax = axes[1, 0]
    mu_values = [0.00, 0.05, 0.10]
    mu_folders = ["mu_brier_0p0", "mu_brier_0p05", "mu_brier_0p1"]
    for index, (value, folder) in enumerate(zip(mu_values, mu_folders)):
        row = draw_setting(ax, value, core_data(folder), default_mean, "#A22DE0", "o", 4100 + index, 0.12, default=value == 0.05)
        record("d", row, {"series": "mu_B", "parameter": "μ_B"})
    ax.set_xticks(mu_values)
    ax.set_xticklabels(["0.00", "0.05", "0.10"])
    ax.set_xlabel("μ_B")

    # (e) Reliability threshold tau; tau=0.5 is the frozen default.
    ax = axes[1, 1]
    tau_values = [0.4, 0.5, 0.6, 0.7]
    tau_data = {
        0.4: core_data("reliability_tau_0p4"),
        0.5: default,
        0.6: core_data("reliability_tau_0p6"),
        0.7: core_data("reliability_tau_0p7"),
    }
    for index, value in enumerate(tau_values):
        row = draw_setting(ax, value, tau_data[value], default_mean, "#FF2424", "o", 5100 + index, 0.36, default=value == 0.5)
        record("e", row, {"series": "tau", "parameter": "τ"})
    ax.set_xticks(tau_values)
    ax.set_xlabel("τ")

    # (f) Joint gamma-tau scan.
    ax = axes[1, 2]
    joint_specs = [
        (0.5, "#2F93EA", "o", -0.007),
        (2.0, "#07539A", "s", 0.007),
    ]
    for series_index, (gamma, color, marker, offset) in enumerate(joint_specs):
        for value_index, tau in enumerate([0.4, 0.6, 0.7]):
            method = f"ESTD-KD-RCF-g{str(gamma).replace('.', 'p')}-t{str(tau).replace('.', 'p')}"
            subset = joint[joint["method"] == method]
            row = draw_setting(ax, tau + offset, subset, default_mean, color, marker, 6100 + series_index * 100 + value_index, 0.36)
            record(
                "f",
                row,
                {"series": f"γ={gamma:.1f}", "parameter": "τ", "actual_x": tau},
            )
    ax.set_xticks([0.4, 0.5, 0.6, 0.7])
    ax.set_xlabel("τ")
    ax.legend(
        handles=[
            Line2D([], [], marker=marker, linestyle="None", color=color, markersize=7, label=f"γ = {gamma:.1f}")
            for gamma, color, marker, _ in joint_specs
        ],
        loc="lower left",
        bbox_to_anchor=(0.025, 0.015),
        frameon=True,
        framealpha=0.72,
        facecolor="white",
        edgecolor="#E1E1E1",
    )

    safety_rows = finalize_safe_axes(axes, panel_rows, panel_defaults)
    pd.DataFrame(source_rows).to_csv(OUT / f"{OUTPUT_STEM}_source_data.csv", index=False)
    pd.DataFrame(safety_rows).to_csv(OUT / f"{OUTPUT_STEM}_axis_safety.csv", index=False)
    save_figure(fig)
    validate_svg(OUT / f"{OUTPUT_STEM}.svg")
    print(f"Created Supplementary Fig. S2 in: {OUT}")


if __name__ == "__main__":
    main()
