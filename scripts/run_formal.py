from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


WORKSPACE = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)
CODE = WORKSPACE / "src" / "eakd_rcf_final.py"
DATA = Path(os.environ.get("EAKD_RCF_DATA_DIR", str(WORKSPACE / "data"))).resolve()
ROOT = Path(os.environ.get("EAKD_RCF_OUTPUT_ROOT", str(WORKSPACE / "outputs"))).resolve()
LOGS = ROOT / "09_logs"
EXPECTED_SHA256 = "E04209F7EF6C0AF093E17C07D21615C57C055619A60F1A5932296710F92F0EF6"
DATASETS = [
    "BCWO", "Haberman", "WDBC", "WPBC", "Pima", "Sick",
    "Mammographic", "Heart", "Hepatitis",
]
METRICS = ["MCC", "F1", "GMean", "PRAUC", "AUC", "ECE", "Brier"]
MAIN = [
    "LR", "RF", "XGBoost", "BalancedRF", "EasyEnsemble", "KD",
    "Yang-ATD-adapted", "BKD", "CBD", "CalibKD", "CCKD", "SRW-KD",
    "UMKD-UDD-adapted", "CB-DistillGrad", "ESTD-KD-RCF",
]
STAGE = [
    "ESTD-Student", "ESTD-KD-S", "ESTD-KD-RCF (fixed fusion)", "ESTD-KD-RCF",
]
ABLATION = [
    "ESTD-KD-RCF", "ESTD-KD-RCF (w/o adaptive temperature)",
    "ESTD-KD-RCF (w/o global reliability KD strength)",
    "ESTD-KD-RCF (w/o Brier)", "ESTD-KD-RCF (fixed fusion)",
    "ESTD-KD-RCF (w/o candidate pool)",
]
ARCH = [
    "ESTD-KD-RCF [Linear]", "ESTD-KD-RCF [MLP-32]",
    "ESTD-KD-RCF [MLP-64-32]", "ESTD-KD-RCF [MLP-128-64]",
]
ALPHA = [
    f"ESTD-KD-RCF [a{low:.2f}-{high:.2f}]"
    for low in (0.10, 0.30, 0.50)
    for high in (0.70, 0.90, 0.95)
]
GAMMA_TAU = [
    f"ESTD-KD-RCF-g{gamma}-t{tau}"
    for gamma in ("0p5", "2p0")
    for tau in ("0p4", "0p6", "0p7")
]
CORE = [
    ("gamma_0p5", "--gamma", "0.5"),
    ("gamma_1p0", "--gamma", "1.0"),
    ("gamma_2p0", "--gamma", "2.0"),
    ("gamma_3p0", "--gamma", "3.0"),
    ("lambda_max_0p3", "--lambda_max", "0.3"),
    ("lambda_max_0p5", "--lambda_max", "0.5"),
    ("lambda_max_0p7", "--lambda_max", "0.7"),
    ("mu_brier_0p0", "--mu_brier", "0.0"),
    ("mu_brier_0p05", "--mu_brier", "0.05"),
    ("mu_brier_0p1", "--mu_brier", "0.1"),
    ("reliability_tau_0p4", "--reliability_tau", "0.4"),
    ("reliability_tau_0p6", "--reliability_tau", "0.6"),
    ("reliability_tau_0p7", "--reliability_tau", "0.7"),
]


@dataclass(frozen=True)
class RunSpec:
    name: str
    directory: str
    methods: list[str]
    flags: list[str]

    @property
    def expected_rows(self) -> int:
        return 9 * 25 * len(self.methods)

    @property
    def output_dir(self) -> Path:
        return ROOT / self.directory

    @property
    def log_path(self) -> Path:
        return LOGS / f"{self.name}.log"


RUNS = [
    RunSpec(
        "02_main_comparison", "02_main_comparison", MAIN,
        ["--with_balanced_rf", "--with_easy_ensemble", "--with_public_kd_methods", "--with_rcf_methods"],
    ),
    RunSpec(
        "03_stage_contribution", "03_stage_contribution", STAGE,
        ["--with_stage_contribution"],
    ),
    RunSpec(
        "04_final_ablation", "04_final_ablation", ABLATION,
        ["--with_rcf_ablation_suite"],
    ),
    RunSpec(
        "05_student_architecture", "05_student_architecture", ARCH,
        ["--with_student_arch_sensitivity"],
    ),
    RunSpec(
        "06_alpha_sensitivity", "06_alpha_sensitivity", ALPHA,
        ["--with_alpha_bounds_sensitivity", "--alpha_min_grid", "0.10,0.30,0.50", "--alpha_max_grid", "0.70,0.90,0.95"],
    ),
    *[
        RunSpec(f"07_{name}", f"07_core_parameter_sensitivity/{name}", ["ESTD-KD-RCF"], [flag, value])
        for name, flag, value in CORE
    ],
    RunSpec(
        "08_gamma_tau_check", "08_gamma_tau_check", GAMMA_TAU,
        ["--with_gamma_tau_grid"],
    ),
]


def check_frozen_code() -> None:
    digest = hashlib.sha256(CODE.read_bytes()).hexdigest().upper()
    if digest != EXPECTED_SHA256:
        raise RuntimeError(f"Frozen-code SHA256 mismatch: {digest}")
    if not DATA.is_dir():
        raise RuntimeError(f"Missing data directory: {DATA}")


def audit(spec: RunSpec) -> dict[str, object]:
    metrics_path = spec.output_dir / "metrics" / "all_metrics.csv"
    if not metrics_path.is_file():
        raise RuntimeError(f"Missing completed metrics: {metrics_path}")
    df = pd.read_csv(metrics_path)
    keys = ["dataset", "method", "repeat_id", "fold_id", "seed"]
    counts = df.groupby(["dataset", "method"]).size()
    expected_pairs = {(dataset, method) for dataset in DATASETS for method in spec.methods}
    actual_pairs = set(counts.index.to_list())
    metric_values = df[METRICS].apply(pd.to_numeric, errors="coerce").to_numpy()
    log = spec.log_path.read_text(encoding="utf-8", errors="ignore")
    checks = {
        "rows": len(df) == spec.expected_rows,
        "methods": set(df["method"]) == set(spec.methods),
        "datasets": set(df["dataset"]) == set(DATASETS),
        "dataset_method_pairs": actual_pairs == expected_pairs,
        "each_pair_25": bool((counts == 25).all()),
        "unique_fold_keys": not df.duplicated(keys).any(),
        "finite_metrics": bool(np.isfinite(metric_values).all()),
        "repeats": set(df["repeat_id"].astype(int)) == {1, 2, 3, 4, 5},
        "folds": set(df["fold_id"].astype(int)) == {1, 2, 3, 4, 5},
        "seeds": set(df["seed"].astype(int)) == {42, 43, 44, 45, 46},
        "seed_repeat_alignment": bool(
            (df["seed"].astype(int) == df["repeat_id"].astype(int) + 41).all()
        ),
        "log_clean": re.search(r"Traceback|\bERROR\b|\bFAILED\b|\bSKIP\b", log, re.I) is None,
    }
    config_path = spec.output_dir / "metrics" / "experiment_config.json"
    if not config_path.is_file():
        checks["config_present"] = False
    else:
        cfg = json.loads(config_path.read_text(encoding="utf-8-sig"))
        checks["frozen_config"] = (
            cfg.get("n_splits") == 5
            and cfg.get("n_repeats") == 5
            and cfg.get("random_state") == 42
            and cfg.get("use_reliable_soft_label") is False
            and cfg.get("estd_decouple_kd_class_weight") is False
            and cfg.get("student_type") == "mlp"
            and cfg.get("mlp_hidden1") == 64
            and cfg.get("mlp_hidden2") == 32
        )
    if spec.name == "02_main_comparison":
        required = [
            "metrics/final_results_mean_std.csv",
            "metrics/paper_main_comparison_mean_std.csv",
            "metrics/statistical_average_ranks.csv",
            "metrics/statistical_friedman.csv",
            "metrics/statistical_wilcoxon_holm_vs_ours.csv",
            "metrics/teacher_quality.csv",
            "metrics/teacher_state_samples.csv",
            "metrics/teacher_state_summary_by_dataset.csv",
            "metrics/teacher_state_summary_overall.csv",
            "metrics/runtime_summary.csv",
            "preds/all_predictions.csv",
        ]
        checks["required_main_outputs"] = all(
            (spec.output_dir / path).is_file() for path in required
        )
    result = {
        "suite": spec.name,
        "expected_rows": spec.expected_rows,
        "actual_rows": len(df),
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
    }
    if result["status"] != "PASS":
        raise RuntimeError(f"Audit failed for {spec.name}: {result}")
    return result


def write_state(state: dict[str, object]) -> None:
    path = ROOT / "runner_state.json"
    state["updated_local"] = datetime.now().astimezone().isoformat()
    path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def run_spec(spec: RunSpec) -> dict[str, object]:
    check_frozen_code()
    spec.output_dir.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    if (spec.output_dir / "metrics" / "all_metrics.csv").is_file():
        result = audit(spec)
        print(f"PASS existing {spec.name}: {result['actual_rows']} rows", flush=True)
        return result

    command = [
        str(PYTHON), "-u", str(CODE),
        "--data_dir", str(DATA),
        "--output_dir", str(spec.output_dir),
        "--datasets", "all",
        "--formal", "--n_splits", "5", "--resume",
        *spec.flags,
        "--methods", ",".join(spec.methods),
    ]
    with spec.log_path.open("ab") as log:
        log.write(f"\n[{datetime.now().astimezone().isoformat()}] BEGIN_OR_RESUME {spec.name}\n".encode())
        log.flush()
        returncode = subprocess.run(command, cwd=WORKSPACE, stdout=log, stderr=subprocess.STDOUT).returncode
        log.write(f"\n[{datetime.now().astimezone().isoformat()}] EXIT_CODE {returncode}\n".encode())
    if returncode != 0:
        raise RuntimeError(f"{spec.name} exited with code {returncode}; see {spec.log_path}")
    result = audit(spec)
    with spec.log_path.open("ab") as log:
        log.write(f"[{datetime.now().astimezone().isoformat()}] AUDIT_PASS rows={result['actual_rows']}\n".encode())
    print(f"PASS {spec.name}: {result['actual_rows']} rows", flush=True)
    return result


def main() -> None:
    global DATA, ROOT, LOGS
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["main", "remaining", "all"], default="all")
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--output-root", type=Path, default=ROOT)
    args = parser.parse_args()
    DATA = args.data_dir.resolve()
    ROOT = args.output_root.resolve()
    LOGS = ROOT / "09_logs"
    specs = RUNS[:1] if args.phase == "main" else RUNS[1:] if args.phase == "remaining" else RUNS
    state: dict[str, object] = {"phase": args.phase, "status": "RUNNING", "completed": []}
    write_state(state)
    try:
        for spec in specs:
            state["current_suite"] = spec.name
            write_state(state)
            result = run_spec(spec)
            state["completed"].append(result)
            write_state(state)
        state["status"] = "PASS"
        state["current_suite"] = None
        write_state(state)
        print(f"SCIREP {args.phase.upper()} FORMAL RUNS COMPLETED", flush=True)
    except Exception as exc:
        state["status"] = "FAILED"
        state["error"] = str(exc)
        write_state(state)
        raise


if __name__ == "__main__":
    main()
