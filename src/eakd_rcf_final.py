# -*- coding: utf-8 -*-
"""
ESTD-KD-RCF Scientific Reports FINAL：最终冻结实验版本。

该版本依据前期完整组件消融和联合消融结果进行方法简化，最终不再使用：
1) reliability-corrected soft target；
2) 逐样本 reliability -> lambda_i 的 KD 强度调节。

最终方法保留两条核心主线：
A. Entropy-driven adaptive distillation (ESTD)
   - 质量受控异构教师：OOF 预测 + Platt 校准 + 教师概率质量加权；
   - 样本预测熵 H_i -> 样本级温度 T_i；
   - 训练折上的教师可靠性仅用于估计一个折级/数据级全局 KD 系数 lambda_bar，
     不再逐样本改变 KD 强度；
   - 类别平衡监督/KD训练 + 轻量 Brier 概率正则。

B. Reliable candidate fusion (RCF)
   - 在内部验证集上比较学生、教师、固定融合与自适应融合候选；
   - 同时考虑不平衡分类性能和概率质量；
   - 外层测试集只用于最终评价，不参与训练、阈值、校准或候选择优。

最终蒸馏关系：
    uncertainty H_i -> temperature T_i              (sample-wise)
    training reliability {r_i} -> lambda_bar         (fold-level/global)
    class imbalance -> class-balanced sample weight  (training weight)

不再存在 reliability -> soft target 或 reliability -> sample-wise lambda_i 的重复控制。

本文件用于从头到尾重新生成 Scientific Reports 投稿所需的统一结果。
所有最终论文结果应来自同一版本、同一 9 个数据集、同一 5-fold x 5-repeat 协议。
"""

import os
import json
import time
import math
import warnings
import argparse
import re
from dataclasses import dataclass, asdict, replace as dataclass_replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ------------------------------
# 可选绘图依赖
# ------------------------------
import matplotlib.pyplot as plt
plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "lines.linewidth": 2,
    "lines.markersize": 8,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "-",
    "figure.figsize": (8, 6),
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "text.antialiased": True,
    "lines.antialiased": True
})
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

COLORS = {
    "ours": "#d62728",
    "fixed_t": "#1f77b4",
    "atd": "#ff7f0e",
    "calibkd": "#2ca02c",
    "student": "#9467bd",
    "ideal": "#000000",
}

# ------------------------------
# ML 依赖
# ------------------------------
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.isotonic import IsotonicRegression

try:
    from scipy.stats import friedmanchisquare, wilcoxon
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

try:
    from imblearn.ensemble import BalancedRandomForestClassifier, EasyEnsembleClassifier
    IMBLEARN_OK = True
except Exception:
    IMBLEARN_OK = False

try:
    import xgboost as xgb
    XGB_OK = True
except Exception:
    XGB_OK = False

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_OK = True
except Exception as e:
    TORCH_OK = False
    raise RuntimeError("该脚本需要 PyTorch。请先安装 torch 后再运行。") from e


# ============================================================
# 一、全局配置
# ============================================================

@dataclass
class ExperimentConfig:
    # 路径
    data_dir: str = "data"
    output_dir: str = "outputs"

    # 数据集
    datasets: str = "all"
    methods_filter: str = ""

    # 外层交叉验证
    n_splits: int = 5
    n_repeats: int = 1
    random_state: int = 42

    # ESTD-KD 核心参数
    gamma: float = 2.0
    fixed_temperature: float = 2.0
    t_min: float = 1.0
    t_max: float = 2.0
    # lambda_kd 仅保留给传统/适配 KD 基线；最终 ESTD 使用训练折可靠性估计的全局 lambda_bar
    lambda_kd: float = 0.7

    # 最终 ESTD 参数：soft target 不做可靠性修正；可靠性只用于估计训练折全局 KD 强度
    use_class_balanced: bool = True
    use_focal: bool = False
    focal_gamma: float = 0.5
    cb_beta: float = 0.999
    use_adaptive_lambda: bool = True
    use_reliability_lambda: bool = True
    # 可靠性映射范围；逐样本 lambda_i 只用于计算训练折平均 lambda_bar
    lambda_min: float = 0.05
    lambda_max: float = 0.50
    reliability_tau: float = 0.50
    reliability_power: float = 1.0
    # 最终版本固定关闭可靠性修正 soft target
    use_reliable_soft_label: bool = False
    # 最终版本保持 class-balanced 权重作用于监督/KD样本损失；不再逐样本调节 lambda_i。
    estd_decouple_kd_class_weight: bool = False
    label_smoothing_eps: float = 0.05
    use_brier_loss: bool = True
    mu_brier: float = 0.05
    use_weighted_brier: bool = False
    use_global_reliability_scale: bool = False
    global_reliability_power: float = 1.0
    use_margin_loss: bool = False
    margin_weight: float = 0.0
    margin_value: float = 1.0
    use_reliability_hard_boost: bool = False
    hard_boost_weight: float = 0.0
    hard_boost_power: float = 1.0
    use_pairwise_rank_loss: bool = False
    rank_loss_weight: float = 0.0
    rank_loss_margin: float = 1.0

    # 输出概率后校准：用训练折内部验证集拟合 Platt 标定器，再应用到测试概率
    use_post_calibration: bool = True

    # 教师概率
    use_teacher_oof: bool = True
    use_teacher_calibration: bool = True
    teacher_calibration_method: str = "sigmoid"   # sigmoid=Platt，isotonic 小样本不推荐
    calibration_cv: int = 3
    inner_oof_splits: int = 5
    use_teacher_weighting: bool = True
    teacher_weight_power: float = 1.0
    teacher_weight_metric: str = "brier"
    use_samplewise_teacher_weighting: bool = False
    samplewise_teacher_certainty_power: float = 1.0
    enable_teacher_svm: bool = False  # 默认关闭 SVM 教师以提升速度；需要时可用 --with_teacher_svm 开启

    # 阈值选择
    use_threshold_search: bool = True
    threshold_metric: str = "MCC"
    threshold_val_size: float = 0.2

    # 学生模型训练
    student_lr: float = 1e-2
    student_weight_decay: float = 1e-4
    student_epochs: int = 200
    student_batch_size: int = 128
    student_patience: int = 30
    grad_clip_norm: float = 5.0
    student_type: str = "mlp"  # linear 或 mlp；默认 mlp 提升非线性数据集表现
    mlp_hidden1: int = 64
    mlp_hidden2: int = 32
    mlp_dropout: float = 0.10

    # 方法开关
    enable_lr: bool = True
    enable_svm: bool = True
    enable_rf: bool = True
    enable_xgb: bool = True
    enable_balanced_rf: bool = False
    enable_easy_ensemble: bool = False

    # 公开论文方法对比开关：默认不强制加入，使用 --with_public_kd_methods 或 --methods 指定时启用
    enable_public_kd_methods: bool = False
    enable_dtkd_style_kd: bool = True
    enable_bkd_style_kd: bool = True
    enable_cbd_style_kd: bool = True
    enable_krdistill_style_kd: bool = True
    enable_bikd_style_kd: bool = True
    # 正式发表的前沿方法适配版：CCKD/ACCKD(ICML 2024)、SRW-KD(ACCV 2024)、UMKD(MICCAI 2025)、CB-DistillGrad(IEEE Access 2026)
    enable_cckd_style_kd: bool = True
    enable_srw_style_kd: bool = True
    enable_umkd_style_kd: bool = True
    enable_cbdistillgrad_style_kd: bool = True

    # 公开方法二分类/表格适配参数
    bkd_weight_beta: float = 0.999
    cbd_resample_ratio: float = 1.0
    bikd_lambda_min: float = 0.0
    bikd_lambda_max: float = 0.6
    bikd_student_hardness_power: float = 1.0
    bikd_teacher_reliability_power: float = 1.0
    cckd_prior_power: float = 1.0
    srw_k_neighbors: int = 10
    srw_weight_strength: float = 1.0
    umkd_uncertainty_power: float = 3.0
    cbdg_conflict_floor: float = 0.25

    # 定性/可解释性输出：可靠性曲线、不确定性统计、预测置信度分布等
    enable_diagnostic_outputs: bool = True
    make_xai_plots: bool = False

    # 退稿意见专项实验开关
    enable_stage_contribution_suite: bool = False
    enable_student_arch_sensitivity: bool = False
    enable_alpha_bounds_sensitivity: bool = False
    enable_teacher_state_analysis: bool = True
    enable_statistical_tests: bool = True
    teacher_state_quantile: float = 0.25
    alpha_min_grid: str = "0.10,0.30,0.50"
    alpha_max_grid: str = "0.70,0.90,0.95"

    enable_student: bool = True
    enable_fixed_t_kd: bool = True
    enable_atd_style_kd: bool = True
    enable_calibkd_style: bool = True
    enable_ours: bool = True
    enable_ours_blend: bool = False
    blend_teacher_weight: float = 0.25
    enable_ours_auto_blend: bool = False
    enable_ours_alpha_sweep: bool = False
    blend_alpha_grid: str = "0.0,0.25,0.5,0.75,1.0"
    blend_selection_metric: str = "balanced"
    blend_space: str = "prob"
    enable_rcf_methods: bool = False
    enable_gamma_tau_grid: bool = False
    enable_rcf_ablation_suite: bool = False
    rcf_min_alpha: float = 0.50
    rcf_max_alpha: float = 0.95
    rcf_C: float = 0.25
    rcf_min_val_samples: int = 60
    rcf_enable_teacher_candidate: bool = True
    rcf_enable_teacher_pool_candidate: bool = True
    rcf_teacher_guard_min_class: int = 40
    rcf_include_raw_teacher_candidate: bool = False
    rcf_adaptive_post_calibration: bool = True
    rcf_enable_student_guard: bool = False
    rcf_candidate_min_gain: float = 0.0
    rcf_teacher_min_gain: float = 0.0
    rcf_teacher_tie_tol: float = 0.005
    rcf_teacher_pool_ece_tol: float = 0.04
    rcf_teacher_pool_brier_tol: float = 0.02
    rcf_dynamic_ece_tol: float = 0.03
    rcf_dynamic_brier_tol: float = 0.015
    rcf_dynamic_tie_tol: float = 0.005
    rcf_student_min_gain: float = 0.05
    rcf_student_mcc_gain: float = 0.05
    rcf_student_max_fixed_mcc: float = 0.55
    rcf_pareto_guard: bool = False
    rcf_pareto_min_gain: float = 0.005
    rcf_pareto_ece_tol: float = 0.025
    rcf_pareto_brier_tol: float = 0.012
    rcf_enable_calibrator_pool: bool = False
    rcf_isotonic_min_class: int = 20
    rcf_threshold_metric_pool: str = ""
    rcf_dual_pareto_guard: bool = False
    rcf_dual_mcc_tol: float = 0.005
    rcf_dual_gmean_tol: float = 0.010
    rcf_safe_calibrator_pool: bool = False
    rcf_safe_cal_ece_tol: float = 0.002
    rcf_safe_cal_brier_tol: float = 0.001
    teacher_disagreement_gate: bool = False
    teacher_entropy_gate_power: float = 1.0
    teacher_disagreement_gate_power: float = 1.0

    # 消融
    enable_ablation: bool = False
    ablation_without_adaptive_temp: bool = True
    ablation_without_adaptive_lambda: bool = True
    # final controlled ablation：恢复旧机制/旧 lambda 下界作为反向消融，不作为新机制
    ablation_with_legacy_reliable_soft_target: bool = True
    ablation_with_v6_lambda_floor: bool = True
    # SCIENTIFIC-REPORTS-FINAL 关键反向消融：恢复“监督与KD共用类别平衡权重”的旧耦合行为
    ablation_with_coupled_cb_kd_weight: bool = True
    ablation_without_reliability_kd_weight: bool = True
    ablation_without_brier: bool = True
    ablation_without_cbfocal: bool = True
    ablation_without_teacher_calibration: bool = False

    # 绘图
    plot_reliability: bool = False
    plot_temp_distribution: bool = False

    # 运行控制
    torch_num_threads: int = 1
    resume_from_partial: bool = False
    verbose: int = 1

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=4, ensure_ascii=False)


# ============================================================
# 二、数据读取
# ============================================================

class DatasetLoader:
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)

    @staticmethod
    def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """统一处理 '?'、空字符串、对象列。"""
        df = df.replace("?", np.nan)
        df = df.replace("", np.nan)
        return df

    @staticmethod
    def _binary_encode(y: pd.Series, positive_label: Optional[Any] = None) -> pd.Series:
        y = pd.Series(y)
        if positive_label is not None:
            return (y == positive_label).astype(int)

        vals = list(y.dropna().unique())
        if len(vals) != 2:
            raise ValueError(f"标签不是二分类，发现唯一值为：{vals}")
        # 默认将出现较少的一类映射为正类 1，更适合不平衡二分类指标解释
        counts = y.value_counts()
        pos_val = counts.idxmin()
        return (y == pos_val).astype(int)

    def _finalize(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        X = self._clean_dataframe(X.copy())
        out = X.copy()
        out["target"] = pd.Series(y).astype(int).values
        out["original_row_id"] = np.arange(len(out))
        return out

    def _exists(self, filename: str) -> bool:
        return (self.data_dir / filename).exists()

    def load_all(self) -> Dict[str, pd.DataFrame]:
        loaders = {
            "BCWO": self.load_bcwo,
            "Haberman": self.load_haberman,
            "WDBC": self.load_wdbc,
            "WPBC": self.load_wpbc,
            "Pima": self.load_pima,
            "Sick": self.load_sick,
            "Mammographic": self.load_mammographic,
            "Heart": self.load_heart,
            "Hepatitis": self.load_hepatitis,
        }
        datasets = {}
        for name, fn in loaders.items():
            try:
                df = fn()
                # 至少两类才保留
                if df["target"].nunique() == 2:
                    datasets[name] = df
                    print(f"[OK] 已加载数据集 {name}: {df.shape}, 类别分布={df['target'].value_counts().to_dict()}")
                else:
                    print(f"[WARN] 跳过 {name}: 非二分类。")
            except FileNotFoundError:
                print(f"[WARN] 跳过 {name}: 文件不存在。")
            except Exception as e:
                print(f"[WARN] 跳过 {name}: 读取失败：{e}")
        if not datasets:
            raise RuntimeError(f"没有成功加载任何数据集，请检查 data_dir={self.data_dir}")
        return datasets

    def load_selected(self, selected: str) -> Dict[str, pd.DataFrame]:
        all_data = self.load_all()
        if selected is None or selected.strip().lower() == "all":
            return all_data
        names = [s.strip() for s in selected.split(",") if s.strip()]
        out = {}
        for name in names:
            # 兼容大小写
            hit = None
            for k in all_data.keys():
                if k.lower() == name.lower():
                    hit = k
                    break
            if hit is None:
                print(f"[WARN] 指定数据集 {name} 未找到，已跳过。可用数据集：{list(all_data.keys())}")
            else:
                out[hit] = all_data[hit]
        if not out:
            raise RuntimeError("指定的数据集均未成功加载。")
        return out

    # -------- 原 4 个数据集 --------

    def load_bcwo(self) -> pd.DataFrame:
        path = self.data_dir / "bcwo.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path, na_values="?")
        label_col = None
        for cand in ["Class", "class", "label", "Label", "Outcome", "outcome", df.columns[-1]]:
            if cand in df.columns:
                label_col = cand
                break
        y_raw = df[label_col]
        if set(pd.Series(y_raw).dropna().unique()).issubset({2, 4}):
            y = y_raw.map({2: 0, 4: 1})
        else:
            y = self._binary_encode(y_raw)
        X = df.drop(columns=[label_col])
        return self._finalize(X, y)

    def load_haberman(self) -> pd.DataFrame:
        path = self.data_dir / "Haberman.data"
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path, header=None, na_values="?")
        df.columns = ["age", "year", "nodes", "target_raw"]
        # 1=survived, 2=died; 将较少的 2 映射为正类
        y = df["target_raw"].map({1: 0, 2: 1})
        X = df.drop(columns=["target_raw"])
        return self._finalize(X, y)

    def load_wdbc(self) -> pd.DataFrame:
        path = self.data_dir / "WDBC.data"
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path, header=None, na_values="?")
        feature_cols = [f"f{i}" for i in range(1, df.shape[1] - 1)]
        df.columns = ["id", "label"] + feature_cols
        y = df["label"].map({"B": 0, "M": 1})
        X = df.drop(columns=["id", "label"])
        return self._finalize(X, y)

    def load_wpbc(self) -> pd.DataFrame:
        path = self.data_dir / "wpbc.data"
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path, header=None, na_values="?")
        feature_cols = [f"f{i}" for i in range(1, df.shape[1] - 1)]
        df.columns = ["id", "label"] + feature_cols
        y = df["label"].map({"N": 0, "R": 1})
        X = df.drop(columns=["id", "label"])
        return self._finalize(X, y)

    # -------- 新增数据集 --------

    def load_pima(self) -> pd.DataFrame:
        path = self.data_dir / "pima.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path, header=None, na_values="?")
        df.columns = [
            "pregnancies", "glucose", "blood_pressure", "skin_thickness",
            "insulin", "bmi", "diabetes_pedigree", "age", "target_raw"
        ]
        # Pima 中部分医学检测项 0 可视为缺失
        zero_as_missing = ["glucose", "blood_pressure", "skin_thickness", "insulin", "bmi"]
        for c in zero_as_missing:
            df[c] = df[c].replace(0, np.nan)
        y = df["target_raw"].astype(int)
        X = df.drop(columns=["target_raw"])
        return self._finalize(X, y)

    def load_sick(self) -> pd.DataFrame:
        path = self.data_dir / "sick.data"
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path, header=None, na_values="?")
        # 最后一列形如 negative.|3733，去掉 | 后面的 id
        label = df.iloc[:, -1].astype(str).str.split("|").str[0].str.replace(".", "", regex=False).str.strip()
        y = (label == "sick").astype(int)
        X = df.iloc[:, :-1].copy()
        X.columns = [f"f{i}" for i in range(X.shape[1])]
        return self._finalize(X, y)

    def load_mammographic(self) -> pd.DataFrame:
        path = self.data_dir / "mammographic_masses.data"
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path, header=None, na_values="?")
        df.columns = ["bi_rads", "age", "shape", "margin", "density", "target_raw"]
        y = df["target_raw"].astype(int)
        X = df.drop(columns=["target_raw"])
        return self._finalize(X, y)

    def load_heart(self) -> pd.DataFrame:
        path = self.data_dir / "heart_disease_cleveland.data"
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path, header=None, na_values="?")
        cols = [
            "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target_raw"
        ]
        df.columns = cols
        # Cleveland 原标签 0=无病，1-4=有病
        y = (df["target_raw"].astype(float) > 0).astype(int)
        X = df.drop(columns=["target_raw"])
        return self._finalize(X, y)

    def load_hepatitis(self) -> pd.DataFrame:
        path = self.data_dir / "hepatitis.data"
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path, header=None, na_values="?")
        # UCI Hepatitis 第一列为 class: 1=die, 2=live；将 die 映射为正类
        y = (df.iloc[:, 0].astype(float) == 1).astype(int)
        X = df.iloc[:, 1:].copy()
        X.columns = [f"f{i}" for i in range(X.shape[1])]
        return self._finalize(X, y)


# ============================================================
# 三、评价指标与阈值
# ============================================================

def sanitize_probabilities(y_prob: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    prob = np.asarray(y_prob, dtype=float)
    prob = np.nan_to_num(prob, nan=0.5, posinf=1.0 - eps, neginf=eps)
    return np.clip(prob, eps, 1.0 - eps)


def blend_probabilities(
    student_prob: np.ndarray,
    teacher_prob: Optional[np.ndarray],
    alpha: float,
    space: str = "prob",
) -> np.ndarray:
    student_prob = sanitize_probabilities(student_prob)
    if teacher_prob is None:
        return student_prob
    teacher_prob = sanitize_probabilities(teacher_prob)
    alpha = float(np.clip(alpha, 0.0, 1.0))
    if str(space).strip().lower() == "logit":
        s_logit = np.log(student_prob / (1.0 - student_prob))
        t_logit = np.log(teacher_prob / (1.0 - teacher_prob))
        z = np.clip((1.0 - alpha) * s_logit + alpha * t_logit, -50.0, 50.0)
        return sanitize_probabilities(1.0 / (1.0 + np.exp(-z)))
    return sanitize_probabilities((1.0 - alpha) * student_prob + alpha * teacher_prob)


class MetricUtils:
    @staticmethod
    def gmean_from_confusion(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        return float(np.sqrt(max(tpr * tnr, 0.0)))

    @staticmethod
    def ece_score(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
        y_true = np.asarray(y_true).astype(int)
        y_prob = sanitize_probabilities(y_prob)
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        bin_ids = np.digitize(y_prob, bins) - 1
        bin_ids = np.clip(bin_ids, 0, n_bins - 1)

        ece = 0.0
        n = len(y_true)
        if n == 0:
            return 0.0
        for b in range(n_bins):
            mask = bin_ids == b
            cnt = np.sum(mask)
            if cnt == 0:
                continue
            acc = np.mean(y_true[mask])
            conf = np.mean(y_prob[mask])
            ece += (cnt / n) * abs(acc - conf)
        return float(ece)

    @staticmethod
    def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray, metric: str = "MCC") -> float:
        y_true = np.asarray(y_true).astype(int)
        y_prob = sanitize_probabilities(y_prob)
        if len(np.unique(y_true)) < 2:
            return 0.5
        thresholds = np.linspace(0.01, 0.99, 199)
        best_t = 0.5
        best_score = -1e18
        metric = metric.upper()

        for t in thresholds:
            y_pred = (y_prob >= t).astype(int)
            if metric == "F1":
                score = f1_score(y_true, y_pred, zero_division=0)
            elif metric == "GMEAN":
                score = MetricUtils.gmean_from_confusion(y_true, y_pred)
            elif metric == "BALANCED":
                score = (
                    matthews_corrcoef(y_true, y_pred)
                    + 0.25 * f1_score(y_true, y_pred, zero_division=0)
                    + 0.25 * MetricUtils.gmean_from_confusion(y_true, y_pred)
                )
            else:
                score = matthews_corrcoef(y_true, y_pred)
            if score > best_score:
                best_score = score
                best_t = t
        return float(best_t)

    @staticmethod
    def summarize(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
        y_true = np.asarray(y_true).astype(int)
        y_prob = sanitize_probabilities(y_prob)
        y_pred = (y_prob >= threshold).astype(int)
        out = {
            "threshold": float(threshold),
            "MCC": matthews_corrcoef(y_true, y_pred),
            "F1": f1_score(y_true, y_pred, zero_division=0),
            "GMean": MetricUtils.gmean_from_confusion(y_true, y_pred),
            "PRAUC": average_precision_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else np.nan,
            "AUC": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else np.nan,
            "ECE": MetricUtils.ece_score(y_true, y_prob, n_bins=10),
            "Brier": brier_score_loss(y_true, y_prob),
        }
        return out

    @staticmethod
    def composite_score(metrics: Dict[str, float], mode: str = "balanced") -> float:
        mode = str(mode or "balanced").strip().lower()
        if mode == "mcc":
            return float(metrics.get("MCC", 0.0))
        if mode == "f1":
            return float(metrics.get("F1", 0.0))
        if mode == "gmean":
            return float(metrics.get("GMean", 0.0))
        if mode == "calibrated":
            return float(
                metrics.get("MCC", 0.0)
                + 0.25 * metrics.get("F1", 0.0)
                + 0.25 * metrics.get("GMean", 0.0)
                - 0.50 * metrics.get("ECE", 0.0)
                - 0.50 * metrics.get("Brier", 0.0)
            )
        return float(
            metrics.get("MCC", 0.0)
            + 0.25 * metrics.get("F1", 0.0)
            + 0.25 * metrics.get("GMean", 0.0)
            - 0.20 * metrics.get("ECE", 0.0)
            - 0.20 * metrics.get("Brier", 0.0)
        )


class ProbabilityCalibrator:
    """Platt-style calibrator for already predicted probabilities."""

    def __init__(self, eps: float = 1e-6):
        self.eps = eps
        self.model = None
        self.constant_prob = None
        self.failed = False

    def _to_logit_feature(self, y_prob: np.ndarray) -> np.ndarray:
        p = sanitize_probabilities(y_prob, eps=self.eps)
        return np.log(p / (1.0 - p)).reshape(-1, 1)

    def fit(self, y_true: np.ndarray, y_prob: np.ndarray):
        y_true = np.asarray(y_true).astype(int)
        if len(y_true) == 0:
            self.failed = True
            return self

        unique = np.unique(y_true)
        if len(unique) < 2:
            self.constant_prob = float(np.mean(y_true))
            return self

        try:
            self.model = LogisticRegression(max_iter=1000, solver="lbfgs")
            self.model.fit(self._to_logit_feature(y_prob), y_true)
        except Exception:
            self.failed = True
            self.model = None
        return self

    def transform(self, y_prob: np.ndarray) -> np.ndarray:
        original = sanitize_probabilities(y_prob, eps=self.eps)
        if self.constant_prob is not None:
            return np.full_like(original, self.constant_prob, dtype=float)
        if self.failed or self.model is None:
            return original
        try:
            return self.model.predict_proba(self._to_logit_feature(original))[:, 1]
        except Exception:
            return original


class TemperatureProbabilityCalibrator:
    """One-parameter temperature scaling for binary probabilities."""

    def __init__(self, eps: float = 1e-6):
        self.eps = eps
        self.temperature = 1.0
        self.failed = False

    def fit(self, y_true: np.ndarray, y_prob: np.ndarray):
        y_true = np.asarray(y_true).astype(int)
        prob = sanitize_probabilities(y_prob, eps=self.eps)
        if len(np.unique(y_true)) < 2:
            self.failed = True
            return self
        logits = np.log(prob / (1.0 - prob))
        best_t = 1.0
        best_loss = float("inf")
        # Small validation folds are common here; a stable grid search is enough.
        for temp in np.linspace(0.5, 3.0, 51):
            scaled = 1.0 / (1.0 + np.exp(-logits / temp))
            loss = brier_score_loss(y_true, sanitize_probabilities(scaled, eps=self.eps))
            if loss < best_loss:
                best_loss = float(loss)
                best_t = float(temp)
        self.temperature = best_t
        return self

    def transform(self, y_prob: np.ndarray) -> np.ndarray:
        original = sanitize_probabilities(y_prob, eps=self.eps)
        if self.failed:
            return original
        logits = np.log(original / (1.0 - original))
        return sanitize_probabilities(1.0 / (1.0 + np.exp(-logits / self.temperature)), eps=self.eps)


class IsotonicProbabilityCalibrator:
    """Isotonic calibrator for validation-selected post calibration."""

    def __init__(self, eps: float = 1e-6):
        self.eps = eps
        self.model = None
        self.failed = False

    def fit(self, y_true: np.ndarray, y_prob: np.ndarray):
        y_true = np.asarray(y_true).astype(int)
        prob = sanitize_probabilities(y_prob, eps=self.eps)
        if len(np.unique(y_true)) < 2:
            self.failed = True
            return self
        try:
            self.model = IsotonicRegression(out_of_bounds="clip", y_min=self.eps, y_max=1.0 - self.eps)
            self.model.fit(prob, y_true)
        except Exception:
            self.failed = True
            self.model = None
        return self

    def transform(self, y_prob: np.ndarray) -> np.ndarray:
        original = sanitize_probabilities(y_prob, eps=self.eps)
        if self.failed or self.model is None:
            return original
        try:
            return sanitize_probabilities(self.model.transform(original), eps=self.eps)
        except Exception:
            return original


# ============================================================
# 四、预处理与模型工厂
# ============================================================

class PreprocessorFactory:
    @staticmethod
    def _make_ohe():
        # 兼容新旧 sklearn
        try:
            return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
        except TypeError:
            return OneHotEncoder(handle_unknown="ignore", sparse=True)

    @staticmethod
    def build(X: pd.DataFrame) -> ColumnTransformer:
        X = X.copy()
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = [c for c in X.columns if c not in numeric_cols]

        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", PreprocessorFactory._make_ohe()),
            ]
        )
        return ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_cols),
                ("cat", categorical_transformer, categorical_cols),
            ]
        )


class TeacherFactory:
    @staticmethod
    def build_teachers(cfg: ExperimentConfig) -> Dict[str, object]:
        # 默认使用 LR + RF + XGB 作为异构教师：速度更快，且覆盖线性/集成/梯度提升三类模型。
        # 若需要完全保留旧版 RF+SVM+XGB，可使用 --with_teacher_svm 开启 SVM 教师。
        teachers: Dict[str, object] = {
            "LR": LogisticRegression(
                max_iter=3000,
                random_state=cfg.random_state,
                class_weight="balanced",
                solver="lbfgs",
            ),
            "RF": RandomForestClassifier(
                n_estimators=200,
                random_state=cfg.random_state,
                class_weight="balanced_subsample",
                n_jobs=-1,
            ),
        }
        if cfg.enable_xgb and XGB_OK:
            teachers["XGB"] = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                objective="binary:logistic",
                random_state=cfg.random_state,
                n_jobs=1,
                verbosity=0,
            )
        if cfg.enable_teacher_svm:
            teachers["SVM"] = SVC(
                C=1.0,
                kernel="rbf",
                probability=True,
                random_state=cfg.random_state,
                class_weight="balanced",
            )
        return teachers


class ClassicalBaselineFactory:
    @staticmethod
    def build(cfg: ExperimentConfig, y_train: Optional[np.ndarray] = None) -> Dict[str, object]:
        baselines: Dict[str, object] = {}
        if cfg.enable_lr:
            baselines["LR"] = LogisticRegression(
                max_iter=3000,
                random_state=cfg.random_state,
                class_weight="balanced",
                solver="lbfgs",
            )
        if cfg.enable_svm:
            baselines["SVM"] = SVC(
                C=1.0,
                kernel="rbf",
                probability=True,
                random_state=cfg.random_state,
                class_weight="balanced",
            )
        if cfg.enable_rf:
            baselines["RF"] = RandomForestClassifier(
                n_estimators=300,
                random_state=cfg.random_state,
                class_weight="balanced_subsample",
                n_jobs=-1,
            )
        if cfg.enable_xgb and XGB_OK:
            scale_pos_weight = 1.0
            if y_train is not None:
                n_pos = np.sum(y_train == 1)
                n_neg = np.sum(y_train == 0)
                scale_pos_weight = float(n_neg / max(n_pos, 1))
            baselines["XGBoost"] = xgb.XGBClassifier(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                objective="binary:logistic",
                random_state=cfg.random_state,
                scale_pos_weight=scale_pos_weight,
                n_jobs=1,
                verbosity=0,
            )
        if cfg.enable_balanced_rf and IMBLEARN_OK:
            baselines["BalancedRF"] = BalancedRandomForestClassifier(
                n_estimators=300,
                random_state=cfg.random_state,
                n_jobs=-1,
            )
        if getattr(cfg, "enable_easy_ensemble", False) and IMBLEARN_OK:
            baselines["EasyEnsemble"] = EasyEnsembleClassifier(
                n_estimators=10,
                random_state=cfg.random_state,
                n_jobs=-1,
            )
        return baselines


def make_calibrated_classifier(base_model, method: str = "sigmoid", cv: int = 3):
    """兼容 sklearn 新旧版本的 CalibratedClassifierCV。"""
    try:
        return CalibratedClassifierCV(estimator=base_model, method=method, cv=cv)
    except TypeError:
        return CalibratedClassifierCV(base_estimator=base_model, method=method, cv=cv)


def maybe_calibrate_model(model, y_train: np.ndarray, cfg: ExperimentConfig):
    if not cfg.use_teacher_calibration:
        return model

    # SVC(probability=True) 自身会进行 Platt 概率拟合；再套 CalibratedClassifierCV 会非常慢，
    # 因此这里对 SVM 教师直接使用其 predict_proba。
    if isinstance(model, SVC):
        return model

    # 重要修复：部分 xgboost + sklearn 版本组合下，CalibratedClassifierCV 会把
    # XGBClassifier 误判成 regressor，从而报：
    # “XGBClassifier should either be a classifier ... Got a regressor ...”。
    # 所以 XGB 教师不再套 sklearn 的 CalibratedClassifierCV，直接使用其 predict_proba。
    # 其它教师（LR/RF）仍保留 Platt 校准，用来稳定教师概率。
    if XGB_OK and isinstance(model, xgb.XGBClassifier):
        return model

    min_class = int(np.min(np.bincount(np.asarray(y_train).astype(int), minlength=2)))
    if min_class < 2:
        return model
    cv = max(2, min(cfg.calibration_cv, min_class))
    try:
        return make_calibrated_classifier(model, method=cfg.teacher_calibration_method, cv=cv)
    except Exception:
        return model


# ============================================================
# 五、蒸馏工具
# ============================================================

class KDDistillationUtils:
    @staticmethod
    def logit_np(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
        return np.log(p / (1.0 - p))

    @staticmethod
    def sigmoid_np(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def entropy_binary_np(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
        return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))

    @staticmethod
    def norm_entropy_np(p: np.ndarray) -> np.ndarray:
        return KDDistillationUtils.entropy_binary_np(p) / np.log(2.0)

    @staticmethod
    def temperature_from_prob(
        p_teacher: np.ndarray,
        t_min: float,
        t_max: float,
        gamma: float,
    ) -> np.ndarray:
        u = KDDistillationUtils.norm_entropy_np(p_teacher)
        temp = t_min + (t_max - t_min) * (u ** gamma)
        return np.asarray(temp, dtype=np.float32)

    @staticmethod
    def soft_label_with_temperature(p_teacher: np.ndarray, temp: np.ndarray) -> np.ndarray:
        z = KDDistillationUtils.logit_np(p_teacher)
        return KDDistillationUtils.sigmoid_np(z / temp)

    @staticmethod
    def adaptive_lambda_from_prob(p_teacher: np.ndarray, lambda_base: float) -> np.ndarray:
        u = KDDistillationUtils.norm_entropy_np(p_teacher)
        lam = lambda_base * (0.5 + 0.5 * u)
        return np.asarray(lam, dtype=np.float32)

    @staticmethod
    def yang_atd_temperature_from_prob(
        p_teacher: np.ndarray,
        t_min: float,
        t_max: float,
        gamma: float = 1.0,
    ) -> np.ndarray:
        """
        Yang et al. (Neurocomputing 2025) ATD 的表格二分类输出层适配。

        保留论文明确给出的方向：教师熵越高 -> 样本越难 -> 使用较低温度；
        教师熵越低 -> 样本越易 -> 使用较高温度。
        由于原方法的 logit-mixup 依赖其图像训练流程，这里不伪装成完整复现，
        因此论文表格中应标为 Yang-ATD-adapted。
        """
        u = np.clip(KDDistillationUtils.norm_entropy_np(p_teacher), 0.0, 1.0)
        gamma = max(float(gamma), 1e-8)
        # u=0 -> Tmax；u=1 -> Tmin。
        temp = float(t_min) + (float(t_max) - float(t_min)) * (1.0 - u ** gamma)
        return np.asarray(np.maximum(temp, 1.0), dtype=np.float32)


# ============================================================
# 六、线性学生与校准感知熵驱动蒸馏损失
# ============================================================

class LinearStudentNet(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(1)


class MLPStudentNet(nn.Module):
    """轻量 MLP 学生；hidden2<=0 时退化为单隐藏层，便于结构敏感性实验。"""

    def __init__(self, in_dim: int, hidden1: int = 64, hidden2: int = 32, dropout: float = 0.10):
        super().__init__()
        hidden1 = max(int(hidden1), 1)
        hidden2 = int(hidden2)
        layers: List[nn.Module] = [
            nn.Linear(in_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
        ]
        if hidden2 > 0:
            layers.extend([
                nn.Linear(hidden1, hidden2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden2, 1),
            ])
        else:
            layers.append(nn.Linear(hidden1, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


def build_student_net(cfg: ExperimentConfig, in_dim: int) -> nn.Module:
    if str(cfg.student_type).lower() == "mlp":
        return MLPStudentNet(
            in_dim,
            hidden1=cfg.mlp_hidden1,
            hidden2=cfg.mlp_hidden2,
            dropout=cfg.mlp_dropout,
        )
    return LinearStudentNet(in_dim)


class TabularStudentKD:
    def __init__(
        self,
        cfg: ExperimentConfig,
        lambda_kd: Optional[float] = None,
        use_focal: Optional[bool] = None,
        use_class_balanced: Optional[bool] = None,
        use_brier_loss: Optional[bool] = None,
        use_adaptive_lambda: Optional[bool] = None,
        use_temperature_kd: bool = True,
        use_kd: bool = True,
        fixed_lambda: Optional[float] = None,
        mu_brier: Optional[float] = None,
        use_weighted_brier: Optional[bool] = None,
        use_global_reliability_scale: Optional[bool] = None,
        global_reliability_power: Optional[float] = None,
        use_margin_loss: Optional[bool] = None,
        margin_weight: Optional[float] = None,
        margin_value: Optional[float] = None,
        use_reliability_hard_boost: Optional[bool] = None,
        hard_boost_weight: Optional[float] = None,
        hard_boost_power: Optional[float] = None,
        use_pairwise_rank_loss: Optional[bool] = None,
        rank_loss_weight: Optional[float] = None,
        rank_loss_margin: Optional[float] = None,
        random_state: Optional[int] = None,
    ):
        self.cfg = cfg
        self.lambda_kd = cfg.lambda_kd if lambda_kd is None else lambda_kd
        self.use_focal = cfg.use_focal if use_focal is None else use_focal
        self.use_class_balanced = cfg.use_class_balanced if use_class_balanced is None else use_class_balanced
        self.use_brier_loss = cfg.use_brier_loss if use_brier_loss is None else use_brier_loss
        self.use_adaptive_lambda = cfg.use_adaptive_lambda if use_adaptive_lambda is None else use_adaptive_lambda
        self.use_temperature_kd = use_temperature_kd
        self.use_kd = use_kd
        self.fixed_lambda = fixed_lambda
        self.mu_brier = cfg.mu_brier if mu_brier is None else mu_brier
        self.use_weighted_brier = cfg.use_weighted_brier if use_weighted_brier is None else use_weighted_brier
        self.use_global_reliability_scale = (
            cfg.use_global_reliability_scale if use_global_reliability_scale is None else use_global_reliability_scale
        )
        self.global_reliability_power = (
            cfg.global_reliability_power if global_reliability_power is None else global_reliability_power
        )
        self.use_margin_loss = cfg.use_margin_loss if use_margin_loss is None else use_margin_loss
        self.margin_weight = cfg.margin_weight if margin_weight is None else margin_weight
        self.margin_value = cfg.margin_value if margin_value is None else margin_value
        self.use_reliability_hard_boost = (
            cfg.use_reliability_hard_boost if use_reliability_hard_boost is None else use_reliability_hard_boost
        )
        self.hard_boost_weight = cfg.hard_boost_weight if hard_boost_weight is None else hard_boost_weight
        self.hard_boost_power = cfg.hard_boost_power if hard_boost_power is None else hard_boost_power
        self.use_pairwise_rank_loss = (
            cfg.use_pairwise_rank_loss if use_pairwise_rank_loss is None else use_pairwise_rank_loss
        )
        self.rank_loss_weight = cfg.rank_loss_weight if rank_loss_weight is None else rank_loss_weight
        self.rank_loss_margin = cfg.rank_loss_margin if rank_loss_margin is None else rank_loss_margin
        self.random_state = cfg.random_state if random_state is None else random_state

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.preprocessor = None
        self.student = None
        self.loss_log: List[float] = []

    def _set_random_seed(self):
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)

    @staticmethod
    def _effective_num_weight(y: np.ndarray, beta: float = 0.999) -> np.ndarray:
        y = np.asarray(y).astype(int)
        n_pos = int(np.sum(y == 1))
        n_neg = int(np.sum(y == 0))
        beta = float(beta)

        def eff_weight(nc: int) -> float:
            if nc <= 0:
                return 0.0
            if abs(beta - 1.0) < 1e-12:
                return 1.0 / nc
            return (1.0 - beta) / (1.0 - beta ** nc)

        w_pos = eff_weight(n_pos)
        w_neg = eff_weight(n_neg)
        w = np.where(y == 1, w_pos, w_neg).astype(np.float32)
        mean_w = float(np.mean(w)) if len(w) > 0 else 1.0
        if mean_w <= 0:
            return np.ones_like(y, dtype=np.float32)
        return (w / mean_w).astype(np.float32)

    @staticmethod
    def _inverse_freq_weight(y: np.ndarray) -> np.ndarray:
        y = np.asarray(y).astype(int)
        n = len(y)
        n_pos = int(np.sum(y == 1))
        n_neg = int(np.sum(y == 0))
        w = np.where(y == 1, n / (2 * max(n_pos, 1)), n / (2 * max(n_neg, 1)))
        return w.astype(np.float32)

    @staticmethod
    def _bernoulli_kl(q: torch.Tensor, p: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        q = torch.clamp(q, eps, 1.0 - eps)
        p = torch.clamp(p, eps, 1.0 - eps)
        return q * torch.log(q / p) + (1.0 - q) * torch.log((1.0 - q) / (1.0 - p))

    @staticmethod
    def _focal_bce_with_logits(logits: torch.Tensor, target: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
        ce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * target + (1.0 - p) * (1.0 - target)
        focal_weight = (1.0 - p_t) ** gamma
        return focal_weight * ce

    @staticmethod
    def _norm_entropy_torch(p: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        p = torch.clamp(p, eps, 1.0 - eps)
        ent = -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p))
        return ent / math.log(2.0)

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        teacher_prob_raw: Optional[np.ndarray] = None,
        sample_temperature: Optional[np.ndarray] = None,
        sample_lambda: Optional[np.ndarray] = None,
        sample_weight_supervised: Optional[np.ndarray] = None,
        sample_weight_kd: Optional[np.ndarray] = None,
    ):
        self._set_random_seed()

        self.preprocessor = PreprocessorFactory.build(X_train)
        Xp = self.preprocessor.fit_transform(X_train)
        if hasattr(Xp, "toarray"):
            Xp = Xp.toarray()
        Xp = np.asarray(Xp, dtype=np.float32)

        y_train = np.asarray(y_train).astype(np.float32)

        if teacher_prob_raw is None or not self.use_kd:
            teacher_prob_raw = np.full(len(y_train), 0.5, dtype=np.float32)
            sample_temperature = np.ones(len(y_train), dtype=np.float32)
            has_teacher = False
        else:
            teacher_prob_raw = np.clip(np.asarray(teacher_prob_raw, dtype=np.float32), 1e-6, 1.0 - 1e-6)
            if sample_temperature is None:
                sample_temperature = np.full(len(y_train), self.cfg.fixed_temperature, dtype=np.float32)
            sample_temperature = np.maximum(np.asarray(sample_temperature, dtype=np.float32), 1.0)
            has_teacher = True

        global_reliability_scale = 1.0
        if has_teacher and self.use_global_reliability_scale:
            teacher_reliability = y_train * teacher_prob_raw + (1.0 - y_train) * (1.0 - teacher_prob_raw)
            tau_np = float(getattr(self.cfg, "reliability_tau", 0.5))
            denom_np = max(1.0 - tau_np, 1e-6)
            global_reliability_scale = float(np.clip((float(np.mean(teacher_reliability)) - tau_np) / denom_np, 0.0, 1.0))
            global_power = float(self.global_reliability_power)
            if abs(global_power - 1.0) > 1e-8:
                global_reliability_scale = float(np.clip(global_reliability_scale, 0.0, 1.0) ** global_power)
        self.last_global_reliability_scale = global_reliability_scale

        if self.use_class_balanced:
            sample_weight = self._effective_num_weight(y_train, beta=self.cfg.cb_beta)
        else:
            sample_weight = self._inverse_freq_weight(y_train)

        # 默认保持旧行为：监督项与蒸馏项使用同一组样本权重。
        # BKD/CBD/BiKD 等公开方法适配版可显式传入不同权重或样本级 lambda。
        if sample_weight_supervised is None:
            sample_weight_supervised = sample_weight
        else:
            sample_weight_supervised = np.asarray(sample_weight_supervised, dtype=np.float32)
        if sample_weight_kd is None:
            sample_weight_kd = sample_weight
        else:
            sample_weight_kd = np.asarray(sample_weight_kd, dtype=np.float32)
        if sample_lambda is None:
            sample_lambda = np.full(len(y_train), np.nan, dtype=np.float32)
        else:
            sample_lambda = np.asarray(sample_lambda, dtype=np.float32)

        X_tensor = torch.tensor(Xp, dtype=torch.float32)
        y_tensor = torch.tensor(y_train, dtype=torch.float32)
        tprob_tensor = torch.tensor(teacher_prob_raw, dtype=torch.float32)
        temp_tensor = torch.tensor(sample_temperature, dtype=torch.float32)
        sup_w_tensor = torch.tensor(sample_weight_supervised, dtype=torch.float32)
        kd_w_tensor = torch.tensor(sample_weight_kd, dtype=torch.float32)
        lambda_tensor = torch.tensor(sample_lambda, dtype=torch.float32)

        dataset = TensorDataset(X_tensor, y_tensor, tprob_tensor, temp_tensor, sup_w_tensor, kd_w_tensor, lambda_tensor)
        loader = DataLoader(
            dataset,
            batch_size=min(self.cfg.student_batch_size, len(dataset)),
            shuffle=True,
            drop_last=False,
        )

        self.student = build_student_net(self.cfg, Xp.shape[1]).to(self.device)
        optimizer = torch.optim.Adam(
            self.student.parameters(),
            lr=self.cfg.student_lr,
            weight_decay=self.cfg.student_weight_decay,
        )

        best_loss = float("inf")
        best_state = None
        bad_epochs = 0
        self.loss_log = []

        self.student.train()
        for epoch in range(self.cfg.student_epochs):
            epoch_loss = 0.0
            n_seen = 0

            for xb, yb, tpb, tb, sup_wb, kd_wb, lambda_ext in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                tpb = tpb.to(self.device)
                tb = tb.to(self.device)
                sup_wb = sup_wb.to(self.device)
                kd_wb = kd_wb.to(self.device)
                lambda_ext = lambda_ext.to(self.device)

                optimizer.zero_grad()
                logits = self.student(xb)
                p_student = torch.sigmoid(logits)

                if self.use_focal:
                    supervised_each = self._focal_bce_with_logits(
                        logits, yb, gamma=self.cfg.focal_gamma
                    )
                else:
                    supervised_each = F.binary_cross_entropy_with_logits(logits, yb, reduction="none")

                hard_boost_factor = torch.ones_like(supervised_each)
                if has_teacher and self.use_kd:
                    # 教师与学生在同一温度下比较，保证蒸馏项理论一致
                    tb_safe = torch.clamp(tb, min=1.0, max=20.0)
                    teacher_logit = torch.logit(torch.clamp(tpb, 1e-6, 1.0 - 1e-6))
                    teacher_soft = torch.sigmoid(teacher_logit / tb_safe)

                    if self.use_temperature_kd:
                        p_student_t = torch.sigmoid(logits / tb_safe)
                        kd_each = self._bernoulli_kl(teacher_soft, p_student_t) * (tb_safe ** 2)
                    else:
                        kd_each = self._bernoulli_kl(teacher_soft, p_student)

                    # v7: 熵只控制温度；蒸馏强度由教师对真实标签的可靠性控制。
                    # r_i = P_teacher(true class)，g_i 越高说明教师越可信。
                    reliability = yb * tpb + (1.0 - yb) * (1.0 - tpb)
                    tau = float(getattr(self.cfg, "reliability_tau", 0.5))
                    denom = max(1.0 - tau, 1e-6)
                    g_i = torch.clamp((reliability - tau) / denom, 0.0, 1.0)
                    power = float(getattr(self.cfg, "reliability_power", 1.0))
                    if abs(power - 1.0) > 1e-8:
                        g_i = torch.clamp(g_i, 0.0, 1.0) ** power

                    if self.use_reliability_hard_boost and float(self.hard_boost_weight) > 0:
                        hard_power = float(self.hard_boost_power)
                        hard_gate = torch.clamp(1.0 - g_i, 0.0, 1.0)
                        if abs(hard_power - 1.0) > 1e-8:
                            hard_gate = hard_gate ** hard_power
                        hard_boost_factor = 1.0 + float(self.hard_boost_weight) * hard_gate

                    # v7 默认不进入该分支；仅供 V6 旧行为回退实验使用。
                    if getattr(self.cfg, "use_reliable_soft_label", False):
                        eps = float(getattr(self.cfg, "label_smoothing_eps", 0.05))
                        y_smooth = yb * (1.0 - eps) + (1.0 - yb) * eps
                        teacher_soft = g_i * teacher_soft + (1.0 - g_i) * y_smooth
                        kd_each = self._bernoulli_kl(teacher_soft, p_student_t) * (tb_safe ** 2)

                    if self.fixed_lambda is not None:
                        lambda_i = torch.full_like(yb, float(self.fixed_lambda))
                    elif self.use_adaptive_lambda and getattr(self.cfg, "use_reliability_lambda", True):
                        lam_min = float(getattr(self.cfg, "lambda_min", 0.0))
                        lam_max = float(getattr(self.cfg, "lambda_max", 0.50))
                        lambda_i = lam_min + (lam_max - lam_min) * g_i
                    elif self.use_adaptive_lambda:
                        # 兼容旧版：熵驱动 lambda。默认不再使用，避免高熵不可靠样本被强蒸馏。
                        u = self._norm_entropy_torch(tpb)
                        lambda_i = self.lambda_kd * (0.5 + 0.5 * u)
                    else:
                        lambda_i = torch.full_like(yb, float(self.lambda_kd))

                    # 公开方法适配版可传入外部样本级 lambda（如 BiKD-style 的 hard/soft 样本权重）。
                    if torch.any(torch.isfinite(lambda_ext)):
                        lambda_i = torch.where(torch.isfinite(lambda_ext), lambda_ext, lambda_i)

                    if self.use_global_reliability_scale:
                        lambda_i = lambda_i * float(global_reliability_scale)
                    lambda_i = torch.clamp(lambda_i, 0.0, 0.99)
                else:
                    kd_each = torch.zeros_like(supervised_each)
                    lambda_i = torch.zeros_like(supervised_each)

                supervised_part = (1.0 - lambda_i) * supervised_each * hard_boost_factor
                kd_part = lambda_i * kd_each
                loss_each = sup_wb * supervised_part + kd_wb * kd_part

                if self.use_margin_loss and float(self.margin_weight) > 0:
                    y_sign = yb * 2.0 - 1.0
                    margin = float(self.margin_value)
                    margin_each = F.softplus(margin - y_sign * logits)
                    loss_each = loss_each + sup_wb * float(self.margin_weight) * margin_each

                if self.use_brier_loss and self.mu_brier > 0:
                    brier_each = (p_student - yb) ** 2
                    if self.use_weighted_brier:
                        brier_weight = sup_wb
                    else:
                        brier_weight = torch.ones_like(sup_wb)
                    loss_each = loss_each + self.mu_brier * brier_weight * brier_each

                loss = loss_each.mean()
                if self.use_pairwise_rank_loss and float(self.rank_loss_weight) > 0:
                    pos_logits = logits[yb > 0.5]
                    neg_logits = logits[yb <= 0.5]
                    if pos_logits.numel() > 0 and neg_logits.numel() > 0:
                        diffs = pos_logits[:, None] - neg_logits[None, :]
                        rank_margin = float(self.rank_loss_margin)
                        rank_loss = F.softplus(rank_margin - diffs).mean()
                        loss = loss + float(self.rank_loss_weight) * rank_loss
                if not torch.isfinite(loss):
                    continue
                loss.backward()
                grad_clip_norm = float(getattr(self.cfg, "grad_clip_norm", 0.0))
                if grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=grad_clip_norm)
                optimizer.step()

                epoch_loss += loss.item() * xb.size(0)
                n_seen += xb.size(0)

            epoch_loss /= max(n_seen, 1)
            self.loss_log.append(epoch_loss)

            if epoch_loss < best_loss - 1e-7:
                best_loss = epoch_loss
                best_state = {k: v.detach().cpu().clone() for k, v in self.student.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1

            if bad_epochs >= self.cfg.student_patience:
                break

        if best_state is not None:
            self.student.load_state_dict(best_state)
        self.student.eval()
        return self

    def predict_proba(self, X_test: pd.DataFrame) -> np.ndarray:
        Xp = self.preprocessor.transform(X_test)
        if hasattr(Xp, "toarray"):
            Xp = Xp.toarray()
        Xp = np.asarray(Xp, dtype=np.float32)
        X_tensor = torch.tensor(Xp, dtype=torch.float32).to(self.device)

        self.student.eval()
        with torch.no_grad():
            logits = self.student(X_tensor)
            prob = torch.sigmoid(logits).detach().cpu().numpy()
        return sanitize_probabilities(prob)


# ============================================================
# 七、教师概率构建与方法运行
# ============================================================

class MethodRunner:
    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg
        self.teacher_records: List[Dict[str, Any]] = []
        self.last_train_teacher_stack: Optional[np.ndarray] = None
        self.last_test_teacher_stack: Optional[np.ndarray] = None
        self.last_teacher_weights: Optional[np.ndarray] = None
        self.last_teacher_names: List[str] = []

    def _fit_predict_pipe(self, model, X_train, y_train, X_pred) -> np.ndarray:
        pre = PreprocessorFactory.build(X_train)
        pipe = Pipeline([("pre", pre), ("clf", clone(model))])
        pipe.fit(X_train, y_train)
        prob = pipe.predict_proba(X_pred)[:, 1]
        return sanitize_probabilities(prob)

    def _fit_predict_teacher(self, model, X_train, y_train, X_pred, calibrate: bool = True) -> np.ndarray:
        pre = PreprocessorFactory.build(X_train)
        base_raw = clone(model)
        base = maybe_calibrate_model(base_raw, y_train, self.cfg) if calibrate else base_raw
        pipe = Pipeline([("pre", pre), ("clf", base)])
        try:
            pipe.fit(X_train, y_train)
        except Exception as e:
            # 兼容修复：少数 sklearn/xgboost 版本下，CalibratedClassifierCV 可能把某些
            # 第三方分类器误判为 regressor；若校准失败，则自动退回未校准教师，
            # 保证实验能继续运行。
            if calibrate:
                if getattr(self.cfg, "verbose", 1) >= 1:
                    print(f"[WARN] 教师校准失败，已自动退回未校准教师：{type(model).__name__} | {str(e)[:120]}")
                pre = PreprocessorFactory.build(X_train)
                pipe = Pipeline([("pre", pre), ("clf", clone(model))])
                pipe.fit(X_train, y_train)
            else:
                raise
        prob = pipe.predict_proba(X_pred)[:, 1]
        return sanitize_probabilities(prob)

    def train_teacher_ensemble(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_test: pd.DataFrame,
        dataset_name: str = "",
        fold_id: int = 0,
        calibrate: Optional[bool] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        """
        返回：
        train_teacher_prob: 用于训练学生的教师概率，默认 OOF；
        test_teacher_prob: 测试集教师概率；
        weights_dict: 各教师融合权重。
        """
        if calibrate is None:
            calibrate = self.cfg.use_teacher_calibration

        teachers = TeacherFactory.build_teachers(self.cfg)
        y_train = np.asarray(y_train).astype(int)
        n = len(y_train)
        min_class = int(np.min(np.bincount(y_train, minlength=2)))

        train_probs = []
        test_probs = []
        quality_scores = []
        weights_dict = {}

        use_oof = self.cfg.use_teacher_oof and min_class >= 2
        inner_splits = min(self.cfg.inner_oof_splits, min_class) if use_oof else 0
        if inner_splits < 2:
            use_oof = False

        for tname, model in teachers.items():
            t0 = time.time()

            if use_oof:
                oof = np.zeros(n, dtype=float)
                inner_cv = StratifiedKFold(
                    n_splits=inner_splits,
                    shuffle=True,
                    random_state=self.cfg.random_state,
                )
                for inner_tr, inner_val in inner_cv.split(X_train, y_train):
                    X_tr = X_train.iloc[inner_tr].reset_index(drop=True)
                    X_val = X_train.iloc[inner_val].reset_index(drop=True)
                    y_tr = y_train[inner_tr]
                    oof[inner_val] = self._fit_predict_teacher(
                        model, X_tr, y_tr, X_val, calibrate=calibrate
                    )
                train_prob = sanitize_probabilities(oof)
            else:
                train_prob = self._fit_predict_teacher(
                    model, X_train, y_train, X_train, calibrate=calibrate
                )

            test_prob = self._fit_predict_teacher(
                model, X_train, y_train, X_test, calibrate=calibrate
            )

            brier = brier_score_loss(y_train, train_prob)
            ece = MetricUtils.ece_score(y_train, train_prob)
            teacher_threshold = MetricUtils.find_best_threshold(y_train, train_prob, metric="MCC")
            teacher_metrics = MetricUtils.summarize(y_train, train_prob, threshold=teacher_threshold)
            weight_metric = str(getattr(self.cfg, "teacher_weight_metric", "brier")).strip().lower()
            if weight_metric == "balanced":
                score = max(MetricUtils.composite_score(teacher_metrics, mode="balanced"), 1e-6)
            elif weight_metric == "mcc":
                score = max(float(teacher_metrics.get("MCC", 0.0)), 0.0) + 1e-6
            else:
                # Brier 越低，质量分越高
                score = 1.0 / (brier + 1e-6)

            train_probs.append(train_prob)
            test_probs.append(test_prob)
            quality_scores.append(score)

            self.teacher_records.append({
                "dataset": dataset_name,
                "fold": fold_id,
                "teacher": tname,
                "calibrated": bool(calibrate),
                "oof": bool(use_oof),
                "train_brier": float(brier),
                "train_ece": float(ece),
                "train_mcc": float(teacher_metrics.get("MCC", np.nan)),
                "train_f1": float(teacher_metrics.get("F1", np.nan)),
                "train_gmean": float(teacher_metrics.get("GMean", np.nan)),
                "train_threshold": float(teacher_threshold),
                "quality_score": float(score),
                "time_sec": float(time.time() - t0),
            })

        train_stack = np.vstack(train_probs)
        test_stack = np.vstack(test_probs)

        if self.cfg.use_teacher_weighting and len(quality_scores) > 0:
            weights = np.asarray(quality_scores, dtype=float)
            power = float(getattr(self.cfg, "teacher_weight_power", 1.0))
            if abs(power - 1.0) > 1e-8:
                weights = np.clip(weights, 1e-12, None) ** power
            weights = weights / np.sum(weights)
        else:
            weights = np.ones(len(train_probs), dtype=float) / len(train_probs)

        teacher_names = list(teachers.keys())
        for name, w in zip(teacher_names, weights):
            weights_dict[name] = float(w)
            # 补充权重到最后对应 teacher_records
            for rec in reversed(self.teacher_records):
                if rec["dataset"] == dataset_name and rec["fold"] == fold_id and rec["teacher"] == name and "ensemble_weight" not in rec:
                    rec["ensemble_weight"] = float(w)
                    break

        if getattr(self.cfg, "use_samplewise_teacher_weighting", False) and len(weights) > 1:
            certainty_power = float(getattr(self.cfg, "samplewise_teacher_certainty_power", 1.0))
            certainty_power = max(certainty_power, 0.0)
            train_cert = np.clip(1.0 - KDDistillationUtils.norm_entropy_np(train_stack), 1e-6, 1.0)
            test_cert = np.clip(1.0 - KDDistillationUtils.norm_entropy_np(test_stack), 1e-6, 1.0)
            if certainty_power > 0:
                train_cert = train_cert ** certainty_power
                test_cert = test_cert ** certainty_power
            train_w = weights[:, None] * train_cert
            test_w = weights[:, None] * test_cert
            train_w = train_w / np.clip(np.sum(train_w, axis=0, keepdims=True), 1e-12, None)
            test_w = test_w / np.clip(np.sum(test_w, axis=0, keepdims=True), 1e-12, None)
            train_ens = np.sum(train_stack * train_w, axis=0)
            test_ens = np.sum(test_stack * test_w, axis=0)
        else:
            train_ens = np.average(train_stack, axis=0, weights=weights)
            test_ens = np.average(test_stack, axis=0, weights=weights)
        self.last_train_teacher_stack = train_stack
        self.last_test_teacher_stack = test_stack
        self.last_teacher_weights = weights
        self.last_teacher_names = teacher_names
        return (
            sanitize_probabilities(train_ens),
            sanitize_probabilities(test_ens),
            weights_dict,
        )

    def train_single_teacher(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_test: pd.DataFrame,
        calibrate: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        teacher = RandomForestClassifier(
            n_estimators=200,
            random_state=self.cfg.random_state,
            class_weight="balanced_subsample",
            n_jobs=-1,
        )
        # 单教师消融使用 RF，但训练概率仍建议 OOF，避免过拟合概率
        old_weighting = self.cfg.use_teacher_weighting
        self.cfg.use_teacher_weighting = False
        try:
            # 为了单教师逻辑更直接，这里单独实现
            y_train = np.asarray(y_train).astype(int)
            min_class = int(np.min(np.bincount(y_train, minlength=2)))
            use_oof = self.cfg.use_teacher_oof and min_class >= 2
            inner_splits = min(self.cfg.inner_oof_splits, min_class) if use_oof else 0
            if use_oof and inner_splits >= 2:
                train_prob = np.zeros(len(y_train), dtype=float)
                cv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=self.cfg.random_state)
                for tr, va in cv.split(X_train, y_train):
                    train_prob[va] = self._fit_predict_teacher(
                        teacher,
                        X_train.iloc[tr].reset_index(drop=True),
                        y_train[tr],
                        X_train.iloc[va].reset_index(drop=True),
                        calibrate=calibrate,
                    )
            else:
                train_prob = self._fit_predict_teacher(teacher, X_train, y_train, X_train, calibrate=calibrate)
            test_prob = self._fit_predict_teacher(teacher, X_train, y_train, X_test, calibrate=calibrate)
            return sanitize_probabilities(train_prob), sanitize_probabilities(test_prob)
        finally:
            self.cfg.use_teacher_weighting = old_weighting

    def run_student_only(self, X_train, y_train, X_test) -> np.ndarray:
        kd = TabularStudentKD(
            self.cfg,
            use_kd=False,
            use_focal=True,
            use_class_balanced=True,
            use_brier_loss=True,
            use_adaptive_lambda=False,
            fixed_lambda=0.0,
            use_weighted_brier=False,
            use_global_reliability_scale=False,
            use_margin_loss=False,
            use_reliability_hard_boost=False,
            use_pairwise_rank_loss=False,
        )
        kd.fit(X_train, y_train, teacher_prob_raw=None, sample_temperature=None)
        return kd.predict_proba(X_test)

    def run_estd_supervised_student(self, X_train, y_train, X_test) -> np.ndarray:
        """
        阶段贡献实验的纯学生起点。
        与完整 ESTD-KD-S 使用相同学生结构、class-balanced 监督和 Brier 约束，
        唯一去掉教师蒸馏，避免把“学生训练协议不同”误当成蒸馏贡献。
        """
        kd = TabularStudentKD(
            self.cfg,
            use_kd=False,
            use_focal=self.cfg.use_focal,
            use_class_balanced=self.cfg.use_class_balanced,
            use_brier_loss=self.cfg.use_brier_loss,
            use_adaptive_lambda=False,
            fixed_lambda=0.0,
            use_weighted_brier=self.cfg.use_weighted_brier,
            use_global_reliability_scale=False,
            use_margin_loss=False,
            use_reliability_hard_boost=False,
            use_pairwise_rank_loss=False,
        )
        kd.fit(X_train, y_train, teacher_prob_raw=None, sample_temperature=None)
        return kd.predict_proba(X_test)

    def run_fixed_t_kd(self, X_train, y_train, X_test, teacher_prob: np.ndarray) -> np.ndarray:
        temp = np.full(len(y_train), self.cfg.fixed_temperature, dtype=np.float32)
        # 固定温度 KD 作为传统蒸馏基线，不加入 CB-Focal 和 Brier，避免抢 ESTD-KD v2 的机制
        kd = TabularStudentKD(
            self.cfg,
            use_kd=True,
            use_focal=False,
            use_class_balanced=False,
            use_brier_loss=False,
            use_adaptive_lambda=False,
            fixed_lambda=self.cfg.lambda_kd,
            use_weighted_brier=False,
            use_global_reliability_scale=False,
            use_margin_loss=False,
            use_reliability_hard_boost=False,
            use_pairwise_rank_loss=False,
        )
        kd.fit(X_train, y_train, teacher_prob_raw=teacher_prob, sample_temperature=temp)
        return kd.predict_proba(X_test)

    def run_yang_atd_adapted(self, X_train, y_train, X_test, teacher_prob: np.ndarray) -> np.ndarray:
        # Yang-ATD-adapted：高熵(难)低温，低熵(易)高温。
        # 原文的 logit-mixup 图像训练机制不在表格数据上强行伪复现。
        temp = KDDistillationUtils.yang_atd_temperature_from_prob(
            teacher_prob, self.cfg.t_min, self.cfg.t_max, gamma=self.cfg.gamma
        )
        kd = TabularStudentKD(
            self.cfg,
            use_kd=True,
            use_focal=False,
            use_class_balanced=False,
            use_brier_loss=False,
            use_adaptive_lambda=False,
            fixed_lambda=self.cfg.lambda_kd,
            use_weighted_brier=False,
            use_global_reliability_scale=False,
            use_margin_loss=False,
            use_reliability_hard_boost=False,
            use_pairwise_rank_loss=False,
        )
        kd.fit(X_train, y_train, teacher_prob_raw=teacher_prob, sample_temperature=temp)
        return kd.predict_proba(X_test)

    # 兼容旧函数名；新论文请使用 Yang-ATD-adapted 标签。
    def run_atd_style_kd(self, X_train, y_train, X_test, teacher_prob: np.ndarray) -> np.ndarray:
        return self.run_yang_atd_adapted(X_train, y_train, X_test, teacher_prob)

    def run_calibkd_style(self, X_train, y_train, X_test, teacher_prob: np.ndarray) -> np.ndarray:
        temp = np.full(len(y_train), self.cfg.fixed_temperature, dtype=np.float32)
        # CalibKD-style：固定温度 + Brier，模拟校准迁移倾向
        kd = TabularStudentKD(
            self.cfg,
            use_kd=True,
            use_focal=False,
            use_class_balanced=False,
            use_brier_loss=True,
            use_adaptive_lambda=False,
            fixed_lambda=self.cfg.lambda_kd,
            mu_brier=self.cfg.mu_brier,
            use_weighted_brier=False,
            use_global_reliability_scale=False,
            use_margin_loss=False,
            use_reliability_hard_boost=False,
            use_pairwise_rank_loss=False,
        )
        kd.fit(X_train, y_train, teacher_prob_raw=teacher_prob, sample_temperature=temp)
        return kd.predict_proba(X_test)

    def run_estd_kd_v2(
        self,
        X_train,
        y_train,
        X_test,
        teacher_prob: np.ndarray,
        variant: str = "full",
        teacher_stack_train: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Final ESTD-KD-S and controlled ablations.

        Final method (variant="full"):
          - reliability-corrected soft target: OFF
          - sample-wise reliability KD weight: OFF
          - entropy-driven sample-wise temperature: ON
          - training-fold reliability is summarized into one global KD coefficient lambda_bar
          - class-balanced sample weighting: ON for supervised/KD losses
          - student Brier regularization: ON

        Main final-paper ablations:
          full                    final simplified ESTD
          no_adaptive_temp        replace T_i by fixed temperature, keep global lambda_bar
          no_global_reliability   replace reliability-derived lambda_bar by the neutral midpoint
                                  (lambda_min + lambda_max) / 2
          no_brier                disable only student Brier regularization

        Legacy diagnostic variants are retained only for reproducibility and are not part of the
        final paper ablation suite.
        """
        variant = variant.lower().strip()
        y_arr = np.asarray(y_train).astype(int)
        teacher_prob = sanitize_probabilities(teacher_prob)

        # Entropy-driven temperature. Only the dedicated ablation removes it.
        if variant == "no_adaptive_temp":
            temp = np.full(len(y_arr), self.cfg.fixed_temperature, dtype=np.float32)
        else:
            temp = KDDistillationUtils.temperature_from_prob(
                teacher_prob, self.cfg.t_min, self.cfg.t_max, self.cfg.gamma
            )

        student_cfg = dataclass_replace(self.cfg, use_reliable_soft_label=False)
        use_brier = self.cfg.use_brier_loss
        use_focal = self.cfg.use_focal
        use_cb = self.cfg.use_class_balanced
        decouple_kd_class_weight = bool(getattr(self.cfg, "estd_decouple_kd_class_weight", False))
        sample_weight_kd = (
            np.ones(len(y_arr), dtype=np.float32) if decouple_kd_class_weight else None
        )

        # ------------------------------------------------------------
        # Final simplification: reliability is not used as sample-wise KD weight.
        # It is summarized within the training fold into one scalar lambda_bar.
        # ------------------------------------------------------------
        reliability = y_arr * teacher_prob + (1 - y_arr) * (1.0 - teacher_prob)
        tau = float(getattr(self.cfg, "reliability_tau", 0.50))
        gate = np.clip((reliability - tau) / max(1.0 - tau, 1e-6), 0.0, 1.0)
        rel_power = float(getattr(self.cfg, "reliability_power", 1.0))
        if abs(rel_power - 1.0) > 1e-8:
            gate = gate ** max(rel_power, 0.0)

        lam_min = float(getattr(self.cfg, "lambda_min", 0.05))
        lam_max = float(getattr(self.cfg, "lambda_max", 0.50))
        mapped_lambda = lam_min + (lam_max - lam_min) * gate
        lambda_bar = float(np.mean(mapped_lambda))

        if variant == "no_global_reliability":
            # Neutral, pre-specified fixed coefficient for the ablation.
            # It does not use labels/teacher reliability and is not selected from test performance.
            lambda_bar = 0.5 * (lam_min + lam_max)

        sample_lambda = np.full(len(y_arr), lambda_bar, dtype=np.float32)
        use_adapt_lam = False

        # Final paper keeps Brier as a lightweight auxiliary regularizer.
        if variant in {"no_brier", "no_global_reliability_no_brier"}:
            use_brier = False
        if variant == "no_cbfocal":
            use_focal = False
            use_cb = False

        # Legacy diagnostics only (not in final paper suite).
        if variant == "legacy_samplewise_reliability":
            sample_lambda = None
            use_adapt_lam = True
        elif variant == "legacy_reliable_soft_target":
            student_cfg = dataclass_replace(self.cfg, use_reliable_soft_label=True)
            sample_lambda = None
            use_adapt_lam = True
        elif variant == "lambda_zero_legacy":
            legacy_cfg = dataclass_replace(self.cfg, lambda_min=0.0, use_reliable_soft_label=False)
            student_cfg = legacy_cfg
            lam_min_legacy = 0.0
            mapped = lam_min_legacy + (lam_max - lam_min_legacy) * gate
            sample_lambda = np.full(len(y_arr), float(np.mean(mapped)), dtype=np.float32)
            use_adapt_lam = False
        elif variant == "decoupled_cb_kd_weight_legacy":
            sample_weight_kd = np.ones(len(y_arr), dtype=np.float32)

        kd = TabularStudentKD(
            student_cfg,
            use_kd=True,
            use_focal=use_focal,
            use_class_balanced=use_cb,
            use_brier_loss=use_brier,
            use_adaptive_lambda=use_adapt_lam,
            fixed_lambda=None,
            use_global_reliability_scale=False,
        )

        kd.fit(
            X_train,
            y_arr,
            teacher_prob_raw=teacher_prob,
            sample_temperature=temp,
            sample_weight_kd=sample_weight_kd,
            sample_lambda=sample_lambda,
        )
        return kd.predict_proba(X_test)


    def _dynamic_temperature_from_sharpness(self, teacher_prob: np.ndarray) -> np.ndarray:
        """
        DTKD-style 二分类适配：用教师输出 sharpness=|2p-1| 估计样本难度，
        sharpness 低（教师更平滑/更不确定）时提高温度，sharpness 高时降低温度。
        该实现用于表格二分类公平对比，不复现图像多分类中的全部训练细节。
        """
        p = sanitize_probabilities(teacher_prob)
        sharpness = np.abs(2.0 * p - 1.0)
        difficulty = np.clip(1.0 - sharpness, 0.0, 1.0)
        gamma = float(getattr(self.cfg, "gamma", 2.0))
        temp = float(self.cfg.t_min) + (float(self.cfg.t_max) - float(self.cfg.t_min)) * (difficulty ** gamma)
        return np.asarray(np.maximum(temp, 1.0), dtype=np.float32)

    def _class_balanced_weight(self, y: np.ndarray, kind: str = "effective") -> np.ndarray:
        y = np.asarray(y).astype(int)
        if kind == "inverse":
            return TabularStudentKD._inverse_freq_weight(y)
        beta = float(getattr(self.cfg, "bkd_weight_beta", getattr(self.cfg, "cb_beta", 0.999)))
        return TabularStudentKD._effective_num_weight(y, beta=beta)

    def _balanced_bootstrap_indices(self, y: np.ndarray, ratio: float = 1.0) -> np.ndarray:
        """CBD-style 用：按类别均衡重采样训练索引。"""
        rng = np.random.default_rng(int(getattr(self.cfg, "random_state", 42)))
        y = np.asarray(y).astype(int)
        idx0 = np.where(y == 0)[0]
        idx1 = np.where(y == 1)[0]
        if len(idx0) == 0 or len(idx1) == 0:
            return np.arange(len(y))
        max_n = int(max(len(idx0), len(idx1)) * float(np.clip(ratio, 0.1, 2.0)))
        max_n = max(max_n, min(len(idx0), len(idx1)))
        res0 = rng.choice(idx0, size=max_n, replace=(len(idx0) < max_n))
        res1 = rng.choice(idx1, size=max_n, replace=(len(idx1) < max_n))
        out = np.concatenate([res0, res1])
        rng.shuffle(out)
        return out.astype(int)

    def run_dtkd_style_kd(self, X_train, y_train, X_test, teacher_prob: np.ndarray) -> np.ndarray:
        # DTKD-style：使用 sharpness 诱导的样本级动态温度；不加入本文可靠性 lambda、Brier 或 RCF。
        temp = self._dynamic_temperature_from_sharpness(teacher_prob)
        kd = TabularStudentKD(
            self.cfg,
            use_kd=True,
            use_focal=False,
            use_class_balanced=False,
            use_brier_loss=False,
            use_adaptive_lambda=False,
            fixed_lambda=self.cfg.lambda_kd,
            use_weighted_brier=False,
            use_global_reliability_scale=False,
            use_margin_loss=False,
            use_reliability_hard_boost=False,
            use_pairwise_rank_loss=False,
        )
        kd.fit(X_train, y_train, teacher_prob_raw=teacher_prob, sample_temperature=temp)
        return kd.predict_proba(X_test)

    def run_bkd_style_kd(self, X_train, y_train, X_test, teacher_prob: np.ndarray) -> np.ndarray:
        """
        BKD-style 二分类适配：instance-balanced classification loss + class-balanced distillation loss。
        即监督 CE 不加类权重，KD 项使用有效样本数权重加强少数类知识迁移。
        """
        y_arr = np.asarray(y_train).astype(int)
        temp = np.full(len(y_arr), self.cfg.fixed_temperature, dtype=np.float32)
        sup_w = np.ones(len(y_arr), dtype=np.float32)
        kd_w = self._class_balanced_weight(y_arr, kind="effective")
        kd = TabularStudentKD(
            self.cfg,
            use_kd=True,
            use_focal=False,
            use_class_balanced=False,
            use_brier_loss=False,
            use_adaptive_lambda=False,
            fixed_lambda=self.cfg.lambda_kd,
            use_weighted_brier=False,
            use_global_reliability_scale=False,
            use_margin_loss=False,
            use_reliability_hard_boost=False,
            use_pairwise_rank_loss=False,
        )
        kd.fit(
            X_train,
            y_arr,
            teacher_prob_raw=teacher_prob,
            sample_temperature=temp,
            sample_weight_supervised=sup_w,
            sample_weight_kd=kd_w,
        )
        return kd.predict_proba(X_test)

    def run_cbd_style_kd(self, X_train, y_train, X_test, teacher_prob: np.ndarray) -> np.ndarray:
        """
        CBD-style 二分类适配：用类别均衡重采样构造第二阶段蒸馏训练集，
        重点验证 class-balanced sampling 对少数类表示/判别的影响。
        """
        y_arr = np.asarray(y_train).astype(int)
        idx = self._balanced_bootstrap_indices(y_arr, ratio=float(getattr(self.cfg, "cbd_resample_ratio", 1.0)))
        X_bal = X_train.iloc[idx].reset_index(drop=True)
        y_bal = y_arr[idx]
        teacher_bal = sanitize_probabilities(np.asarray(teacher_prob)[idx])
        temp = np.full(len(y_bal), self.cfg.fixed_temperature, dtype=np.float32)
        kd = TabularStudentKD(
            self.cfg,
            use_kd=True,
            use_focal=False,
            use_class_balanced=False,
            use_brier_loss=False,
            use_adaptive_lambda=False,
            fixed_lambda=self.cfg.lambda_kd,
            use_weighted_brier=False,
            use_global_reliability_scale=False,
            use_margin_loss=False,
            use_reliability_hard_boost=False,
            use_pairwise_rank_loss=False,
        )
        kd.fit(X_bal, y_bal, teacher_prob_raw=teacher_bal, sample_temperature=temp)
        return kd.predict_proba(X_test)


    def run_krdistill_style_kd(self, X_train, y_train, X_test, teacher_prob: np.ndarray) -> np.ndarray:
        """
        KRDistill-style 二分类适配：根据训练集类别先验把教师 logit 纠正到均衡先验，
        用以缓解教师从不平衡数据继承的头类偏置。原方法还包含表示纠正；表格二分类
        代码中主要实现可公平复现的 logit prior rectification 部分。
        """
        y_arr = np.asarray(y_train).astype(int)
        p_pos = float(np.mean(y_arr == 1))
        p_neg = float(np.mean(y_arr == 0))
        p_pos = float(np.clip(p_pos, 1e-6, 1.0 - 1e-6))
        p_neg = float(np.clip(p_neg, 1e-6, 1.0 - 1e-6))
        # balanced prior correction: logit p(y=1|x) + log(pi_bal_pos/pi_train_pos) - log(pi_bal_neg/pi_train_neg)
        delta = math.log(p_neg / p_pos)
        z = KDDistillationUtils.logit_np(teacher_prob) + delta
        rectified_teacher = sanitize_probabilities(KDDistillationUtils.sigmoid_np(z))
        temp = np.full(len(y_arr), self.cfg.fixed_temperature, dtype=np.float32)
        kd_w = self._class_balanced_weight(y_arr, kind="effective")
        kd = TabularStudentKD(
            self.cfg,
            use_kd=True,
            use_focal=False,
            use_class_balanced=False,
            use_brier_loss=False,
            use_adaptive_lambda=False,
            fixed_lambda=self.cfg.lambda_kd,
            use_weighted_brier=False,
            use_global_reliability_scale=False,
            use_margin_loss=False,
            use_reliability_hard_boost=False,
            use_pairwise_rank_loss=False,
        )
        kd.fit(
            X_train,
            y_arr,
            teacher_prob_raw=rectified_teacher,
            sample_temperature=temp,
            sample_weight_supervised=np.ones(len(y_arr), dtype=np.float32),
            sample_weight_kd=kd_w,
        )
        return kd.predict_proba(X_test)

    def run_bikd_style_kd(self, X_train, y_train, X_test, teacher_prob: np.ndarray) -> np.ndarray:
        """
        BiKD-style 二分类适配：根据教师可靠性与学生训练难度生成样本级 hard/soft 权重。
        原 BiKD 使用双层优化与权重生成网络；这里用表格二分类可运行的轻量近似，
        作为公开前沿不平衡 KD 思想的公平适配对比，而不是本文方法本身。
        """
        y_arr = np.asarray(y_train).astype(int)
        # pilot student 用于估计学生行为/样本困难度，呼应 BiKD 中“考虑学生行为”的思想。
        pilot = TabularStudentKD(
            self.cfg,
            use_kd=False,
            use_focal=False,
            use_class_balanced=True,
            use_brier_loss=False,
            use_adaptive_lambda=False,
            fixed_lambda=0.0,
            use_weighted_brier=False,
            use_global_reliability_scale=False,
            use_margin_loss=False,
            use_reliability_hard_boost=False,
            use_pairwise_rank_loss=False,
        )
        pilot.fit(X_train, y_arr, teacher_prob_raw=None, sample_temperature=None)
        student_train_prob = pilot.predict_proba(X_train)

        t_prob = sanitize_probabilities(teacher_prob)
        teacher_true_prob = y_arr * t_prob + (1 - y_arr) * (1.0 - t_prob)
        student_true_prob = y_arr * student_train_prob + (1 - y_arr) * (1.0 - student_train_prob)
        tau = float(getattr(self.cfg, "reliability_tau", 0.5))
        teacher_gate = np.clip((teacher_true_prob - tau) / max(1.0 - tau, 1e-6), 0.0, 1.0)
        teacher_gate = teacher_gate ** float(getattr(self.cfg, "bikd_teacher_reliability_power", 1.0))
        student_hardness = np.clip(1.0 - student_true_prob, 0.0, 1.0)
        student_gate = np.clip(student_hardness / 0.5, 0.0, 1.0)
        student_gate = student_gate ** float(getattr(self.cfg, "bikd_student_hardness_power", 1.0))
        class_w = self._class_balanced_weight(y_arr, kind="effective")
        class_gate = np.clip(class_w / max(float(np.max(class_w)), 1e-6), 0.0, 1.0)
        raw_gate = teacher_gate * (0.50 + 0.35 * student_gate + 0.15 * class_gate)
        lam_min = float(getattr(self.cfg, "bikd_lambda_min", 0.0))
        lam_max = float(getattr(self.cfg, "bikd_lambda_max", 0.6))
        sample_lambda = np.clip(lam_min + (lam_max - lam_min) * raw_gate, 0.0, 0.95).astype(np.float32)

        temp = np.full(len(y_arr), self.cfg.fixed_temperature, dtype=np.float32)
        kd = TabularStudentKD(
            self.cfg,
            use_kd=True,
            use_focal=False,
            use_class_balanced=False,
            use_brier_loss=False,
            use_adaptive_lambda=False,
            fixed_lambda=self.cfg.lambda_kd,
            use_weighted_brier=False,
            use_global_reliability_scale=False,
            use_margin_loss=False,
            use_reliability_hard_boost=False,
            use_pairwise_rank_loss=False,
        )
        kd.fit(
            X_train,
            y_arr,
            teacher_prob_raw=t_prob,
            sample_temperature=temp,
            sample_lambda=sample_lambda,
            sample_weight_supervised=np.ones(len(y_arr), dtype=np.float32),
            sample_weight_kd=class_w,
        )
        return kd.predict_proba(X_test)


    def _prior_correct_prob(self, teacher_prob: np.ndarray, y_train: np.ndarray, power: float = 1.0) -> np.ndarray:
        """
        Class-conditional / prior-corrected probability for binary class-imbalanced KD.
        p_bal = (p / pi_pos^power) / (p / pi_pos^power + (1-p) / pi_neg^power)
        """
        y_arr = np.asarray(y_train).astype(int)
        p = sanitize_probabilities(teacher_prob)
        pi_pos = float(np.mean(y_arr == 1))
        pi_neg = float(np.mean(y_arr == 0))
        pi_pos = float(np.clip(pi_pos, 1e-6, 1.0 - 1e-6))
        pi_neg = float(np.clip(pi_neg, 1e-6, 1.0 - 1e-6))
        power = float(np.clip(power, 0.0, 3.0))
        pos_term = p / (pi_pos ** power)
        neg_term = (1.0 - p) / (pi_neg ** power)
        return sanitize_probabilities(pos_term / np.clip(pos_term + neg_term, 1e-8, None))

    def _samplewise_similarity_weights(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        k_neighbors: int = 10,
        strength: float = 1.0,
    ) -> np.ndarray:
        """
        SRW-KD-style 表格适配：在预处理后的表格特征空间内估计样本局部密度。
        同类邻域平均相似度越低，说明样本越像 sample-wise tail/hard sample，
        给予更高 KD 权重。返回均值归一化后的样本权重。
        """
        y_arr = np.asarray(y_train).astype(int)
        n = len(y_arr)
        if n == 0:
            return np.array([], dtype=np.float32)
        try:
            pre = PreprocessorFactory.build(X_train)
            Xp = pre.fit_transform(X_train)
            if hasattr(Xp, "toarray"):
                Xp = Xp.toarray()
            Xp = np.asarray(Xp, dtype=np.float32)
        except Exception:
            return np.ones(n, dtype=np.float32)

        raw_tail_score = np.zeros(n, dtype=float)
        k_neighbors = int(max(2, k_neighbors))
        for cls in np.unique(y_arr):
            idx_cls = np.where(y_arr == cls)[0]
            if len(idx_cls) <= 2:
                raw_tail_score[idx_cls] = 1.0
                continue
            kk = min(k_neighbors + 1, len(idx_cls))
            try:
                nn = NearestNeighbors(n_neighbors=kk, metric="cosine")
                nn.fit(Xp[idx_cls])
                dist, _ = nn.kneighbors(Xp[idx_cls], return_distance=True)
                # 去掉自己（第一列通常为 0），cosine similarity = 1 - distance
                if dist.shape[1] > 1:
                    neigh_dist = dist[:, 1:]
                else:
                    neigh_dist = dist
                avg_sim = 1.0 - np.mean(neigh_dist, axis=1)
                avg_sim = np.nan_to_num(avg_sim, nan=0.0, posinf=1.0, neginf=-1.0)
                # 相似度低 => tail/hard 分数高
                s_min, s_max = float(np.min(avg_sim)), float(np.max(avg_sim))
                if abs(s_max - s_min) < 1e-8:
                    tail_score = np.zeros_like(avg_sim)
                else:
                    tail_score = 1.0 - (avg_sim - s_min) / (s_max - s_min)
                raw_tail_score[idx_cls] = tail_score
            except Exception:
                raw_tail_score[idx_cls] = 0.0

        class_w = self._class_balanced_weight(y_arr, kind="effective")
        class_w = class_w / max(float(np.mean(class_w)), 1e-8)
        strength = float(max(strength, 0.0))
        w = (1.0 + strength * raw_tail_score) * np.sqrt(np.clip(class_w, 1e-6, None))
        w = w / max(float(np.mean(w)), 1e-8)
        return w.astype(np.float32)

    def _uncertainty_weighted_teacher(
        self,
        teacher_stack: Optional[np.ndarray],
        fallback_prob: np.ndarray,
        power: float = 3.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        UMKD-style 表格适配：根据每个专家的输出熵进行样本级专家加权。
        返回：加权教师概率、专家平均不确定性、专家分歧方差。
        """
        fallback = sanitize_probabilities(fallback_prob)
        if teacher_stack is None:
            u = np.clip(KDDistillationUtils.norm_entropy_np(fallback), 0.0, 1.0)
            return fallback, u, np.zeros(len(fallback), dtype=float)

        stack = np.asarray(teacher_stack, dtype=float)
        if stack.ndim != 2 or stack.shape[1] != len(fallback):
            u = np.clip(KDDistillationUtils.norm_entropy_np(fallback), 0.0, 1.0)
            return fallback, u, np.zeros(len(fallback), dtype=float)

        stack = sanitize_probabilities(stack)
        ent = np.clip(KDDistillationUtils.norm_entropy_np(stack), 0.0, 1.0)
        # entropy 越低，专家权重越高；逐样本归一化
        logits = -float(power) * ent
        logits = logits - np.max(logits, axis=0, keepdims=True)
        weights = np.exp(logits)
        weights = weights / np.clip(np.sum(weights, axis=0, keepdims=True), 1e-8, None)
        prob = np.sum(weights * stack, axis=0)
        mean_uncertainty = np.sum(weights * ent, axis=0)
        disagreement = np.var(stack, axis=0)
        return sanitize_probabilities(prob), mean_uncertainty, disagreement

    def run_cckd_style_kd(self, X_train, y_train, X_test, teacher_prob: np.ndarray) -> np.ndarray:
        """
        CCKD/ACCKD-style 二分类表格适配：
        保留“类别条件/类别先验校正知识”的核心思想，将教师概率按训练折类别先验
        做 class-conditional prior correction，并用 class-balanced KD 权重加强尾类知识。
        """
        y_arr = np.asarray(y_train).astype(int)
        p_cc = self._prior_correct_prob(
            teacher_prob,
            y_arr,
            power=float(getattr(self.cfg, "cckd_prior_power", 1.0)),
        )
        temp = np.full(len(y_arr), self.cfg.fixed_temperature, dtype=np.float32)
        kd_w = self._class_balanced_weight(y_arr, kind="effective")
        kd = TabularStudentKD(
            self.cfg,
            use_kd=True,
            use_focal=False,
            use_class_balanced=False,
            use_brier_loss=False,
            use_adaptive_lambda=False,
            fixed_lambda=self.cfg.lambda_kd,
            use_weighted_brier=False,
            use_global_reliability_scale=False,
            use_margin_loss=False,
            use_reliability_hard_boost=False,
            use_pairwise_rank_loss=False,
        )
        kd.fit(
            X_train,
            y_arr,
            teacher_prob_raw=p_cc,
            sample_temperature=temp,
            sample_weight_supervised=np.ones(len(y_arr), dtype=np.float32),
            sample_weight_kd=kd_w,
        )
        return kd.predict_proba(X_test)

    def run_srw_kd_style(self, X_train, y_train, X_test, teacher_prob: np.ndarray) -> np.ndarray:
        """
        SRW-KD-style 二分类表格适配：
        在表格特征空间估计 sample-wise tail/hard 权重，对局部稀疏或少数类困难样本
        提高蒸馏损失权重，模拟 ACCV 2024 的 sample-wise re-weighting 思想。
        """
        y_arr = np.asarray(y_train).astype(int)
        temp = np.full(len(y_arr), self.cfg.fixed_temperature, dtype=np.float32)
        srw_w = self._samplewise_similarity_weights(
            X_train,
            y_arr,
            k_neighbors=int(getattr(self.cfg, "srw_k_neighbors", 10)),
            strength=float(getattr(self.cfg, "srw_weight_strength", 1.0)),
        )
        kd = TabularStudentKD(
            self.cfg,
            use_kd=True,
            use_focal=False,
            use_class_balanced=False,
            use_brier_loss=False,
            use_adaptive_lambda=False,
            fixed_lambda=self.cfg.lambda_kd,
            use_weighted_brier=False,
            use_global_reliability_scale=False,
            use_margin_loss=False,
            use_reliability_hard_boost=False,
            use_pairwise_rank_loss=False,
        )
        kd.fit(
            X_train,
            y_arr,
            teacher_prob_raw=teacher_prob,
            sample_temperature=temp,
            sample_weight_supervised=np.ones(len(y_arr), dtype=np.float32),
            sample_weight_kd=srw_w,
        )
        return kd.predict_proba(X_test)

    def run_umkd_style_kd(
        self,
        X_train,
        y_train,
        X_test,
        teacher_prob: np.ndarray,
        teacher_pred_prob: Optional[np.ndarray] = None,
        teacher_stack_train: Optional[np.ndarray] = None,
        teacher_stack_pred: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        UMKD-style 二分类表格适配：
        用多教师输出熵进行样本级专家加权，形成 uncertainty-aware teacher；
        训练时再根据加权教师的不确定性降低不可靠知识迁移强度。
        """
        y_arr = np.asarray(y_train).astype(int)
        train_teacher, train_u, train_dis = self._uncertainty_weighted_teacher(
            teacher_stack_train,
            teacher_prob,
            power=float(getattr(self.cfg, "umkd_uncertainty_power", 3.0)),
        )
        temp = np.full(len(y_arr), self.cfg.fixed_temperature, dtype=np.float32)
        # UDD-style: uncertainty 越低，KD 权重越高；专家分歧越大，KD 权重越低。
        certainty_gate = np.clip(1.0 - train_u, 0.0, 1.0)
        disagreement_gate = np.clip(1.0 - 4.0 * train_dis, 0.0, 1.0)
        sample_lambda = np.clip(
            float(self.cfg.lambda_kd) * (0.30 + 0.70 * certainty_gate * disagreement_gate),
            0.0,
            0.95,
        ).astype(np.float32)
        kd_w = self._class_balanced_weight(y_arr, kind="effective")
        kd = TabularStudentKD(
            self.cfg,
            use_kd=True,
            use_focal=False,
            use_class_balanced=False,
            use_brier_loss=False,
            use_adaptive_lambda=False,
            fixed_lambda=self.cfg.lambda_kd,
            use_weighted_brier=False,
            use_global_reliability_scale=False,
            use_margin_loss=False,
            use_reliability_hard_boost=False,
            use_pairwise_rank_loss=False,
        )
        kd.fit(
            X_train,
            y_arr,
            teacher_prob_raw=train_teacher,
            sample_temperature=temp,
            sample_lambda=sample_lambda,
            sample_weight_supervised=np.ones(len(y_arr), dtype=np.float32),
            sample_weight_kd=kd_w,
        )
        return kd.predict_proba(X_test)

    def run_cbdistillgrad_style_kd(self, X_train, y_train, X_test, teacher_prob: np.ndarray) -> np.ndarray:
        """
        CB-DistillGrad-style 二分类表格适配：
        保留 class-balanced distillation + conflict-aware weighting 思想。
        当教师软标签与真实标签方向冲突时降低 KD 权重，减少 hard loss 与 soft loss 的梯度冲突。
        """
        y_arr = np.asarray(y_train).astype(int)
        t_prob = sanitize_probabilities(teacher_prob)
        teacher_true_prob = y_arr * t_prob + (1 - y_arr) * (1.0 - t_prob)
        teacher_pred = (t_prob >= 0.5).astype(int)
        conflict = (teacher_pred != y_arr).astype(float)
        # 真实标签方向上教师越可靠，KD 比重越高；冲突样本显著降低 KD，避免软标签把学生压成先验常数。
        tau = float(getattr(self.cfg, "reliability_tau", 0.5))
        agreement_gate = np.clip((teacher_true_prob - tau) / max(1.0 - tau, 1e-6), 0.0, 1.0)
        floor = float(np.clip(getattr(self.cfg, "cbdg_conflict_floor", 0.25), 0.0, 1.0))
        non_conflict_lambda = float(self.cfg.lambda_kd) * agreement_gate
        conflict_lambda = float(self.cfg.lambda_kd) * floor * agreement_gate
        sample_lambda = np.where(conflict > 0.5, conflict_lambda, non_conflict_lambda)
        sample_lambda = np.clip(sample_lambda, 0.0, 0.45).astype(np.float32)
        class_w = self._class_balanced_weight(y_arr, kind="effective")
        sup_w = class_w * (1.0 + 0.5 * conflict + 0.5 * (1.0 - agreement_gate))
        kd_w = class_w * (0.25 + 0.75 * agreement_gate) * (1.0 - 0.75 * conflict)
        temp = np.full(len(y_arr), self.cfg.fixed_temperature, dtype=np.float32)
        kd = TabularStudentKD(
            self.cfg,
            use_kd=True,
            use_focal=False,
            use_class_balanced=True,
            use_brier_loss=True,
            use_adaptive_lambda=False,
            fixed_lambda=None,
            use_weighted_brier=False,
            use_global_reliability_scale=False,
            use_margin_loss=False,
            use_reliability_hard_boost=False,
            use_pairwise_rank_loss=False,
        )
        kd.fit(
            X_train,
            y_arr,
            teacher_prob_raw=t_prob,
            sample_temperature=temp,
            sample_lambda=sample_lambda,
            sample_weight_supervised=sup_w,
            sample_weight_kd=kd_w,
        )
        return kd.predict_proba(X_test)

    def predict_method_prepared(
        self,
        method: str,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_pred: pd.DataFrame,
        ensemble_teacher_prob: Optional[np.ndarray] = None,
        ensemble_pred_prob: Optional[np.ndarray] = None,
        single_teacher_prob: Optional[np.ndarray] = None,
        uncalibrated_teacher_prob: Optional[np.ndarray] = None,
        uncalibrated_pred_prob: Optional[np.ndarray] = None,
        ensemble_teacher_stack_train: Optional[np.ndarray] = None,
        ensemble_teacher_stack_pred: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        使用已经预先构建好的教师概率进行预测，避免每个方法重复训练 OOF 教师。
        ensemble_teacher_prob / single_teacher_prob 的长度必须与 y_train 一致。
        """
        method = method.strip()

        if method in ["LR", "SVM", "RF", "XGBoost", "BalancedRF", "EasyEnsemble"]:
            baselines = ClassicalBaselineFactory.build(self.cfg, y_train=y_train)
            if method not in baselines:
                raise ValueError(f"方法 {method} 不可用，可能依赖未安装或开关关闭。")
            return self._fit_predict_pipe(baselines[method], X_train, y_train, X_pred)

        if method == "Student":
            return self.run_student_only(X_train, y_train, X_pred)

        if method == "ESTD-Student":
            return self.run_estd_supervised_student(X_train, y_train, X_pred)

        if method == "TeacherEns":
            if uncalibrated_pred_prob is not None:
                return sanitize_probabilities(uncalibrated_pred_prob)
            if ensemble_pred_prob is not None:
                return sanitize_probabilities(ensemble_pred_prob)
            raise ValueError("TeacherEns needs uncalibrated_pred_prob or ensemble_pred_prob.")

        if method == "CalibratedTeacherEns":
            if ensemble_pred_prob is not None:
                return sanitize_probabilities(ensemble_pred_prob)
            raise ValueError("CalibratedTeacherEns needs ensemble_pred_prob.")

        if ensemble_teacher_prob is None:
            raise ValueError(f"{method} 需要 ensemble_teacher_prob，但当前未提供。")

        if method == "KD":
            return self.run_fixed_t_kd(X_train, y_train, X_pred, ensemble_teacher_prob)

        if method in {"ATD", "Yang-ATD-adapted"}:
            prob = single_teacher_prob if single_teacher_prob is not None else ensemble_teacher_prob
            return self.run_yang_atd_adapted(X_train, y_train, X_pred, prob)

        if method == "DTKD":
            prob = single_teacher_prob if single_teacher_prob is not None else ensemble_teacher_prob
            return self.run_dtkd_style_kd(X_train, y_train, X_pred, prob)

        if method == "BKD":
            return self.run_bkd_style_kd(X_train, y_train, X_pred, ensemble_teacher_prob)

        if method == "CBD":
            return self.run_cbd_style_kd(X_train, y_train, X_pred, ensemble_teacher_prob)

        if method == "KRDistill":
            return self.run_krdistill_style_kd(X_train, y_train, X_pred, ensemble_teacher_prob)

        if method == "BiKD":
            return self.run_bikd_style_kd(X_train, y_train, X_pred, ensemble_teacher_prob)

        if method in ["CCKD", "ACCKD"]:
            return self.run_cckd_style_kd(X_train, y_train, X_pred, ensemble_teacher_prob)

        if method == "SRW-KD":
            return self.run_srw_kd_style(X_train, y_train, X_pred, ensemble_teacher_prob)

        if method in {"UMKD", "UMKD-UDD-adapted"}:
            return self.run_umkd_style_kd(
                X_train,
                y_train,
                X_pred,
                ensemble_teacher_prob,
                teacher_pred_prob=ensemble_pred_prob,
                teacher_stack_train=ensemble_teacher_stack_train,
                teacher_stack_pred=ensemble_teacher_stack_pred,
            )

        if method == "CB-DistillGrad":
            return self.run_cbdistillgrad_style_kd(X_train, y_train, X_pred, ensemble_teacher_prob)

        if method == "CalibKD":
            prob = single_teacher_prob if single_teacher_prob is not None else ensemble_teacher_prob
            return self.run_calibkd_style(X_train, y_train, X_pred, prob)

        if method == "ESTD-KD":
            return self.run_estd_kd_v2(X_train, y_train, X_pred, ensemble_teacher_prob, variant="full")

        if method == "ESTD-KD-S":
            return self.run_estd_kd_v2(X_train, y_train, X_pred, ensemble_teacher_prob, variant="full")

        if method == "ESTD-KD+":
            student_prob = self.run_estd_kd_v2(X_train, y_train, X_pred, ensemble_teacher_prob, variant="full")
            if ensemble_pred_prob is None:
                return student_prob
            alpha = float(getattr(self.cfg, "blend_teacher_weight", 0.25))
            return blend_probabilities(
                student_prob,
                ensemble_pred_prob,
                alpha,
                space=getattr(self.cfg, "blend_space", "prob"),
            )

        if method == "ESTD-KD-Auto":
            student_prob = self.run_estd_kd_v2(X_train, y_train, X_pred, ensemble_teacher_prob, variant="full")
            if ensemble_pred_prob is None:
                return student_prob
            alpha = float(getattr(self.cfg, "blend_teacher_weight", 0.25))
            return blend_probabilities(
                student_prob,
                ensemble_pred_prob,
                alpha,
                space=getattr(self.cfg, "blend_space", "prob"),
            )

        if method == "ESTD-KD (w/o adaptive temperature)":
            return self.run_estd_kd_v2(X_train, y_train, X_pred, ensemble_teacher_prob, variant="no_adaptive_temp")

        if method == "ESTD-KD (V6 reliable soft target)":
            return self.run_estd_kd_v2(X_train, y_train, X_pred, ensemble_teacher_prob, variant="legacy_reliable_soft_target")

        if method == "ESTD-KD (V6 lambda_min=0.05)":
            return self.run_estd_kd_v2(X_train, y_train, X_pred, ensemble_teacher_prob, variant="full")

        if method == "ESTD-KD (coupled CB-KD weight)":
            return self.run_estd_kd_v2(X_train, y_train, X_pred, ensemble_teacher_prob, variant="decoupled_cb_kd_weight_legacy")

        if method == "ESTD-KD (lambda_min=0)":
            return self.run_estd_kd_v2(X_train, y_train, X_pred, ensemble_teacher_prob, variant="lambda_zero_legacy")

        if method == "ESTD-KD (w/o reliable soft target)":
            # Legacy compatibility; final full already disables reliable soft targets.
            return self.run_estd_kd_v2(X_train, y_train, X_pred, ensemble_teacher_prob, variant="full")

        if method == "ESTD-KD (w/o global reliability KD strength)":
            return self.run_estd_kd_v2(X_train, y_train, X_pred, ensemble_teacher_prob, variant="no_global_reliability")

        if method == "ESTD-KD (legacy sample-wise reliability KD weight)":
            return self.run_estd_kd_v2(X_train, y_train, X_pred, ensemble_teacher_prob, variant="legacy_samplewise_reliability")

        if method == "ESTD-KD (w/o global reliability KD strength & w/o Brier)":
            return self.run_estd_kd_v2(
                X_train, y_train, X_pred, ensemble_teacher_prob,
                variant="no_global_reliability_no_brier"
            )

        if method == "ESTD-KD (w/o adaptive lambda)":
            return self.run_estd_kd_v2(X_train, y_train, X_pred, ensemble_teacher_prob, variant="no_global_reliability")

        if method == "ESTD-KD (w/o Brier)":
            return self.run_estd_kd_v2(X_train, y_train, X_pred, ensemble_teacher_prob, variant="no_brier")

        if method == "ESTD-KD (w/o CB-Focal)":
            return self.run_estd_kd_v2(X_train, y_train, X_pred, ensemble_teacher_prob, variant="no_cbfocal")

        if method == "ESTD-KD (w/o teacher calibration)":
            prob = uncalibrated_teacher_prob if uncalibrated_teacher_prob is not None else ensemble_teacher_prob
            return self.run_estd_kd_v2(X_train, y_train, X_pred, prob, variant="full")

        raise ValueError(f"未知方法：{method}")

    def predict_method(
        self,
        method: str,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_pred: pd.DataFrame,
        dataset_name: str = "",
        fold_id: int = 0,
    ) -> np.ndarray:
        """
        兼容旧调用：内部会重新训练教师。正式实验主流程已经改为 predict_method_prepared，
        这里仅作为备用。
        """
        method = method.strip()

        if method in ["LR", "SVM", "RF", "XGBoost", "BalancedRF", "EasyEnsemble"]:
            baselines = ClassicalBaselineFactory.build(self.cfg, y_train=y_train)
            if method not in baselines:
                raise ValueError(f"方法 {method} 不可用，可能依赖未安装或开关关闭。")
            return self._fit_predict_pipe(baselines[method], X_train, y_train, X_pred)

        ensemble_prob, ensemble_pred_prob, _ = self.train_teacher_ensemble(
            X_train, y_train, X_pred, dataset_name=dataset_name, fold_id=fold_id
        )
        single_prob, _ = self.train_single_teacher(
            X_train, y_train, X_pred, calibrate=self.cfg.use_teacher_calibration
        )
        return self.predict_method_prepared(
            method,
            X_train,
            y_train,
            X_pred,
            ensemble_teacher_prob=ensemble_prob,
            ensemble_pred_prob=ensemble_pred_prob,
            single_teacher_prob=single_prob,
        )


# ============================================================
# 八、实验主流程
# ============================================================

class ExperimentRunner:
    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg
        self.output_dir = Path(cfg.output_dir)
        self.metrics_dir = self.output_dir / "metrics"
        self.figures_dir = self.output_dir / "figures"
        self.preds_dir = self.output_dir / "preds"
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.preds_dir.mkdir(parents=True, exist_ok=True)
        cfg.save(str(self.metrics_dir / "experiment_config.json"))

        self.all_teacher_records: List[Dict[str, Any]] = []
        self.all_time_records: List[Dict[str, Any]] = []
        self.all_teacher_state_records: List[Dict[str, Any]] = []

        method_notes = {
            "Yang-ATD-adapted": "Tabular binary-classification adaptation preserving the entropy/difficulty-driven temperature principle; image-specific logit mixup is not claimed as reproduced.",
            "UMKD-UDD-adapted": "Tabular binary-classification adaptation preserving output-space uncertainty-aware multi-expert distillation; convolutional SFA/CFA feature modules are not claimed as reproduced.",
            "ESTD-KD-RCF": "Scientific Reports FINAL: reliability-corrected soft target and sample-wise reliability KD weighting are removed. Entropy controls sample-wise temperature; training-fold teacher reliability is summarized into one global KD coefficient; Brier remains an auxiliary regularizer; RCF selects among validated candidates.",
            "ESTD-KD-RCF (w/o adaptive temperature)": "Final ablation: replaces entropy-driven T_i with the fixed temperature while preserving the global reliability-derived KD coefficient and RCF.",
            "ESTD-KD-RCF (w/o global reliability KD strength)": "Final ablation: replaces the training-fold reliability-derived global KD coefficient with the pre-specified midpoint (lambda_min+lambda_max)/2.",
            "ESTD-KD-RCF (w/o Brier)": "Final ablation: disables only the student Brier regularization term; teacher-quality Brier and evaluation Brier remain unchanged.",
            "ESTD-KD-RCF (fixed fusion)": "Final ablation: replaces validation-driven candidate selection with the fixed teacher-student fusion baseline.",
            "ESTD-KD-RCF (w/o candidate pool)": "Final ablation: removes teacher-related candidate-pool options while preserving the remaining RCF pipeline.",
        }
        with open(self.metrics_dir / "method_implementation_notes.json", "w", encoding="utf-8") as f:
            json.dump(method_notes, f, indent=2, ensure_ascii=False)

    def method_list(self) -> List[str]:
        methods: List[str] = []
        if self.cfg.enable_lr:
            methods.append("LR")
        if self.cfg.enable_svm:
            methods.append("SVM")
        if self.cfg.enable_rf:
            methods.append("RF")
        if self.cfg.enable_xgb and XGB_OK:
            methods.append("XGBoost")
        if self.cfg.enable_balanced_rf and IMBLEARN_OK:
            methods.append("BalancedRF")
        if getattr(self.cfg, "enable_easy_ensemble", False) and IMBLEARN_OK:
            methods.append("EasyEnsemble")

        if self.cfg.enable_student:
            methods.append("Student")
        if self.cfg.enable_fixed_t_kd:
            methods.append("KD")
        if self.cfg.enable_atd_style_kd:
            methods.append("Yang-ATD-adapted")
        if getattr(self.cfg, "enable_public_kd_methods", False):
            if getattr(self.cfg, "enable_dtkd_style_kd", True): methods.append("DTKD")
            if getattr(self.cfg, "enable_bkd_style_kd", True): methods.append("BKD")
            if getattr(self.cfg, "enable_cbd_style_kd", True): methods.append("CBD")
            if getattr(self.cfg, "enable_krdistill_style_kd", True): methods.append("KRDistill")
            if getattr(self.cfg, "enable_bikd_style_kd", True): methods.append("BiKD")
            if getattr(self.cfg, "enable_cckd_style_kd", True): methods.append("CCKD")
            if getattr(self.cfg, "enable_srw_style_kd", True): methods.append("SRW-KD")
            if getattr(self.cfg, "enable_umkd_style_kd", True): methods.append("UMKD-UDD-adapted")
            if getattr(self.cfg, "enable_cbdistillgrad_style_kd", True): methods.append("CB-DistillGrad")
        if self.cfg.enable_calibkd_style:
            methods.append("CalibKD")
        if self.cfg.enable_ours:
            methods.append("ESTD-KD")

        if getattr(self.cfg, "enable_ours_blend", False): methods.append("ESTD-KD+")
        if getattr(self.cfg, "enable_ours_auto_blend", False): methods.append("ESTD-KD-Auto")
        if getattr(self.cfg, "enable_ours_alpha_sweep", False): methods.append("ESTD-KD-Sweep")
        if getattr(self.cfg, "enable_rcf_methods", False):
            methods.extend(["TeacherEns", "CalibratedTeacherEns", "ESTD-KD-S", "ESTD-KD-RCF"])

        if getattr(self.cfg, "enable_stage_contribution_suite", False):
            methods.extend([
                "ESTD-Student",
                "ESTD-KD-S",
                "ESTD-KD-RCF (fixed fusion)",
                "ESTD-KD-RCF",
            ])

        if getattr(self.cfg, "enable_gamma_tau_grid", False):
            methods.extend([
                "ESTD-KD-RCF-g0p5-t0p4", "ESTD-KD-RCF-g0p5-t0p6", "ESTD-KD-RCF-g0p5-t0p7",
                "ESTD-KD-RCF-g2p0-t0p4", "ESTD-KD-RCF-g2p0-t0p6", "ESTD-KD-RCF-g2p0-t0p7",
            ])

        if getattr(self.cfg, "enable_rcf_ablation_suite", False):
            methods.extend([
                "ESTD-KD-RCF (w/o adaptive temperature)",
                "ESTD-KD-RCF (w/o global reliability KD strength)",
                "ESTD-KD-RCF (w/o Brier)",
                "ESTD-KD-RCF (fixed fusion)",
                "ESTD-KD-RCF (w/o candidate pool)",
                "ESTD-KD-RCF",
            ])

        if getattr(self.cfg, "enable_student_arch_sensitivity", False):
            methods.extend([
                "ESTD-KD-RCF [Linear]",
                "ESTD-KD-RCF [MLP-32]",
                "ESTD-KD-RCF [MLP-64-32]",
                "ESTD-KD-RCF [MLP-128-64]",
            ])

        if getattr(self.cfg, "enable_alpha_bounds_sensitivity", False):
            try:
                amin_list = [float(x.strip()) for x in str(self.cfg.alpha_min_grid).split(",") if x.strip()]
                amax_list = [float(x.strip()) for x in str(self.cfg.alpha_max_grid).split(",") if x.strip()]
            except Exception:
                amin_list, amax_list = [0.1, 0.3, 0.5], [0.7, 0.9, 0.95]
            for amin in amin_list:
                for amax in amax_list:
                    if amin < amax:
                        methods.append(f"ESTD-KD-RCF [a{amin:.2f}-{amax:.2f}]")

        if self.cfg.enable_ablation:
            # Student-stage final ablations only. RCF-level ablations use --with_rcf_ablation_suite.
            methods.extend([
                "ESTD-KD (w/o adaptive temperature)",
                "ESTD-KD (w/o global reliability KD strength)",
                "ESTD-KD (w/o Brier)",
            ])

        # CLI 兼容旧论文/旧命令中的简称。
        alias = {
            "atd": "Yang-ATD-adapted",
            "yang-atd": "Yang-ATD-adapted",
            "yang-atd-adapted": "Yang-ATD-adapted",
            "umkd": "UMKD-UDD-adapted",
            "umkd-udd-adapted": "UMKD-UDD-adapted",
            "dtkd": "DTKD", "bkd": "BKD", "cbd": "CBD",
            "krdistill": "KRDistill", "kr-distill": "KRDistill", "bikd": "BiKD",
            "cckd": "CCKD", "acckd": "CCKD", "srw-kd": "SRW-KD", "srwkd": "SRW-KD",
            "cb-distillgrad": "CB-DistillGrad", "cbdistillgrad": "CB-DistillGrad",
            "easyensemble": "EasyEnsemble", "easy-ensemble": "EasyEnsemble",
            "estd-student": "ESTD-Student", "estd-kd-s": "ESTD-KD-S",
            "estd-kd-rcf": "ESTD-KD-RCF", "teacherens": "TeacherEns",
            "calibratedteacherens": "CalibratedTeacherEns",
        }

        # 去重。
        deduped: List[str] = []
        seen = set()
        for method in methods:
            key = method.lower()
            if key not in seen:
                deduped.append(method)
                seen.add(key)
        methods = deduped

        if getattr(self.cfg, "methods_filter", ""):
            requested_raw = [m.strip() for m in str(self.cfg.methods_filter).split(",") if m.strip()]
            requested = [alias.get(m.lower(), m) for m in requested_raw]
            existing_lower = {m.lower() for m in methods}
            for canonical in requested:
                if canonical.lower() not in existing_lower:
                    # 对专项方法允许显式请求，即使未打开 suite 开关。
                    if (
                        canonical.startswith("ESTD-KD-RCF")
                        or canonical in {"ESTD-Student", "ESTD-KD-S", "Yang-ATD-adapted", "UMKD-UDD-adapted"}
                    ):
                        methods.append(canonical)
                        existing_lower.add(canonical.lower())
            requested_lower = {m.lower() for m in requested}
            methods = [m for m in methods if m.lower() in requested_lower]
            missing = [raw for raw, canon in zip(requested_raw, requested) if canon.lower() not in {x.lower() for x in methods}]
            if missing:
                print(f"[WARN] 请求的方法不可用或未启用，已跳过：{missing}")

        return methods

    def save_dataset_summary(self, datasets: Dict[str, pd.DataFrame]):
        rows = []
        for name, df in datasets.items():
            y = df["target"].values
            n_pos = int(np.sum(y == 1))
            n_neg = int(np.sum(y == 0))
            ir = float(max(n_neg, n_pos) / max(min(n_neg, n_pos), 1))
            rows.append({
                "dataset": name,
                "n_samples": len(df),
                "n_features": df.shape[1] - 2,
                "n_negative": n_neg,
                "n_positive": n_pos,
                "imbalance_ratio": round(ir, 4),
            })
        pd.DataFrame(rows).to_csv(self.metrics_dir / "dataset_summary.csv", index=False, encoding="utf-8-sig")

    @staticmethod
    def _can_split_val(y_train: np.ndarray, val_size: float) -> bool:
        y_train = np.asarray(y_train).astype(int)
        counts = np.bincount(y_train, minlength=2)
        if np.min(counts) < 2:
            return False
        n_val = int(math.ceil(len(y_train) * val_size))
        return n_val >= 2 and n_val < len(y_train)

    def _make_val_split(self, X_train: pd.DataFrame, y_train: np.ndarray):
        if self.cfg.use_threshold_search and self._can_split_val(y_train, self.cfg.threshold_val_size):
            try:
                X_sub, X_val, y_sub, y_val = train_test_split(
                    X_train,
                    y_train,
                    test_size=self.cfg.threshold_val_size,
                    stratify=y_train,
                    random_state=self.cfg.random_state,
                )
                return (
                    X_sub.reset_index(drop=True),
                    X_val.reset_index(drop=True),
                    np.asarray(y_sub).astype(int),
                    np.asarray(y_val).astype(int),
                    True,
                )
            except Exception:
                pass
        # 退化：用训练集自身选阈值，仅用于极小数据集兜底
        return X_train.reset_index(drop=True), X_train.reset_index(drop=True), np.asarray(y_train).astype(int), np.asarray(y_train).astype(int), False

    def _parse_alpha_grid(self) -> List[float]:
        raw = str(getattr(self.cfg, "blend_alpha_grid", "0.0,0.25,0.5,0.75,1.0"))
        alphas: List[float] = []
        for part in raw.split(","):
            part = part.strip()
            if not part:
                continue
            try:
                alpha = float(np.clip(float(part), 0.0, 1.0))
            except Exception:
                continue
            if np.isfinite(alpha):
                alphas.append(round(alpha, 6))
        if not alphas:
            alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
        return sorted(set(alphas))

    def _blend_student_teacher(self, student_prob: np.ndarray, teacher_prob: Optional[np.ndarray], alpha: float) -> np.ndarray:
        return blend_probabilities(
            student_prob,
            teacher_prob,
            alpha,
            space=getattr(self.cfg, "blend_space", "prob"),
        )

    @staticmethod
    def _teacher_variance(teacher_stack: Optional[np.ndarray], fallback_len: int) -> np.ndarray:
        if teacher_stack is None:
            return np.zeros(fallback_len, dtype=float)
        stack = np.asarray(teacher_stack, dtype=float)
        if stack.ndim != 2 or stack.shape[1] != fallback_len:
            return np.zeros(fallback_len, dtype=float)
        return np.var(sanitize_probabilities(stack), axis=0)

    def _rcf_features(
        self,
        student_prob: np.ndarray,
        teacher_prob: np.ndarray,
        teacher_stack: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        p_s = sanitize_probabilities(student_prob)
        p_t = sanitize_probabilities(teacher_prob)
        u_s = np.clip(KDDistillationUtils.norm_entropy_np(p_s), 0.0, 1.0)
        u_t = np.clip(KDDistillationUtils.norm_entropy_np(p_t), 0.0, 1.0)
        disagreement = np.abs(p_t - p_s)
        t_var = self._teacher_variance(teacher_stack, len(p_s))
        z_s = np.clip(KDDistillationUtils.logit_np(p_s), -10.0, 10.0)
        z_t = np.clip(KDDistillationUtils.logit_np(p_t), -10.0, 10.0)
        return np.column_stack([
            p_s,
            p_t,
            z_s,
            z_t,
            1.0 - u_s,
            1.0 - u_t,
            np.abs(p_s - 0.5),
            np.abs(p_t - 0.5),
            disagreement,
            t_var,
        ])

    def _fit_rcf_head(self, y_val: np.ndarray, X_feat: np.ndarray, cfg: Optional[ExperimentConfig] = None):
        cfg = self.cfg if cfg is None else cfg
        y_val = np.asarray(y_val).astype(int)
        min_val_samples = int(getattr(cfg, "rcf_min_val_samples", 60))
        if len(np.unique(y_val)) < 2 or len(y_val) < min_val_samples:
            if getattr(self.cfg, "verbose", 1) >= 2 and len(y_val) < min_val_samples:
                print(f"RCF fusion head skipped: validation samples {len(y_val)} < {min_val_samples}.")
            return None
        try:
            head = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(
                    max_iter=1000,
                    solver="lbfgs",
                    class_weight="balanced",
                    C=float(getattr(cfg, "rcf_C", 0.25)),
                    random_state=cfg.random_state,
                )),
            ])
            head.fit(X_feat, y_val)
            return head
        except Exception as e:
            if getattr(self.cfg, "verbose", 1) >= 1:
                print(f"[WARN] RCF fusion head fit failed, fallback to fixed blend. Error: {str(e)[:120]}")
            return None

    def _predict_rcf_blend(
        self,
        head,
        X_feat: np.ndarray,
        student_prob: np.ndarray,
        teacher_prob: np.ndarray,
        cfg: Optional[ExperimentConfig] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        cfg = self.cfg if cfg is None else cfg
        p_s = sanitize_probabilities(student_prob)
        p_t = sanitize_probabilities(teacher_prob)
        default_alpha = float(np.clip(getattr(cfg, "blend_teacher_weight", 0.90), 0.0, 1.0))
        alpha_min = float(np.clip(getattr(cfg, "rcf_min_alpha", 0.50), 0.0, 1.0))
        alpha_max = float(np.clip(getattr(cfg, "rcf_max_alpha", 0.95), alpha_min, 1.0))
        default_alpha = float(np.clip(default_alpha, alpha_min, alpha_max))

        if head is None:
            alpha = np.full(len(p_s), default_alpha, dtype=float)
            prob = (1.0 - alpha) * p_s + alpha * p_t
            return sanitize_probabilities(prob), alpha, sanitize_probabilities(prob)

        try:
            head_prob = sanitize_probabilities(head.predict_proba(X_feat)[:, 1])
        except Exception:
            head_prob = blend_probabilities(p_s, p_t, default_alpha, space="prob")

        denom = p_t - p_s
        raw_alpha = np.full(len(p_s), default_alpha, dtype=float)
        mask = np.abs(denom) > 1e-6
        raw_alpha[mask] = (head_prob[mask] - p_s[mask]) / denom[mask]
        alpha = np.clip(raw_alpha, alpha_min, alpha_max)
        prob = (1.0 - alpha) * p_s + alpha * p_t
        return sanitize_probabilities(prob), alpha, head_prob

    def _evaluate_rcf_candidate(
        self,
        y_val: np.ndarray,
        cand_name: str,
        cand_prob: np.ndarray,
        cand_alpha: np.ndarray,
    ) -> Dict[str, Any]:
        raw_prob = sanitize_probabilities(cand_prob)
        variants: List[Tuple[str, Optional[ProbabilityCalibrator], np.ndarray]] = [
            ("raw", None, raw_prob)
        ]

        use_post = bool(getattr(self.cfg, "use_post_calibration", True))
        adaptive_post = bool(getattr(self.cfg, "rcf_adaptive_post_calibration", True))
        if use_post:
            cand_calibrator = ProbabilityCalibrator().fit(y_val, raw_prob)
            cal_prob = cand_calibrator.transform(raw_prob)
            if adaptive_post:
                variants.append(("post", cand_calibrator, cal_prob))
            else:
                variants = [("post", cand_calibrator, cal_prob)]
            if adaptive_post and (
                getattr(self.cfg, "rcf_enable_calibrator_pool", False)
                or getattr(self.cfg, "rcf_safe_calibrator_pool", False)
            ):
                temp_calibrator = TemperatureProbabilityCalibrator().fit(y_val, raw_prob)
                variants.append(("temp", temp_calibrator, temp_calibrator.transform(raw_prob)))
                if getattr(self.cfg, "rcf_enable_calibrator_pool", False):
                    min_class = int(np.min(np.bincount(np.asarray(y_val).astype(int), minlength=2)))
                    if min_class >= int(getattr(self.cfg, "rcf_isotonic_min_class", 20)):
                        iso_calibrator = IsotonicProbabilityCalibrator().fit(y_val, raw_prob)
                        variants.append(("isotonic", iso_calibrator, iso_calibrator.transform(raw_prob)))

        threshold_metric_pool = [
            m.strip()
            for m in str(getattr(self.cfg, "rcf_threshold_metric_pool", "") or "").split(",")
            if m.strip()
        ]
        if not threshold_metric_pool:
            threshold_metric_pool = [str(self.cfg.threshold_metric)]

        best_choice: Optional[Dict[str, Any]] = None
        raw_metrics_by_threshold: Dict[str, Dict[str, float]] = {}
        safe_cal_pool = bool(getattr(self.cfg, "rcf_safe_calibrator_pool", False))
        safe_ece_tol = float(getattr(self.cfg, "rcf_safe_cal_ece_tol", 0.002))
        safe_brier_tol = float(getattr(self.cfg, "rcf_safe_cal_brier_tol", 0.001))
        for cal_mode, cand_calibrator, prob_for_threshold in variants:
            for threshold_metric in threshold_metric_pool:
                if self.cfg.use_threshold_search:
                    cand_threshold = MetricUtils.find_best_threshold(
                        y_val,
                        prob_for_threshold,
                        metric=threshold_metric,
                    )
                else:
                    cand_threshold = 0.5

                cand_metrics = MetricUtils.summarize(
                    y_val,
                    prob_for_threshold,
                    threshold=cand_threshold,
                )
                threshold_metric_key = str(threshold_metric)
                if cal_mode == "raw":
                    raw_metrics_by_threshold[threshold_metric_key] = cand_metrics
                elif safe_cal_pool:
                    raw_metrics = raw_metrics_by_threshold.get(threshold_metric_key)
                    if raw_metrics is not None:
                        if (
                            cand_metrics.get("ECE", 1.0) > raw_metrics.get("ECE", 1.0) + safe_ece_tol
                            or cand_metrics.get("Brier", 1.0) > raw_metrics.get("Brier", 1.0) + safe_brier_tol
                        ):
                            continue
                cand_score = MetricUtils.composite_score(
                    cand_metrics,
                    mode=getattr(self.cfg, "blend_selection_metric", "balanced"),
                )
                choice = {
                    "name": cand_name,
                    "calibration_mode": cal_mode,
                    "threshold_metric": str(threshold_metric),
                    "calibrator": cand_calibrator,
                    "threshold": float(cand_threshold),
                    "metrics": cand_metrics,
                    "score": float(cand_score),
                    "alpha_mean": float(np.mean(cand_alpha)),
                    "alpha_std": float(np.std(cand_alpha)),
                }
                if best_choice is None:
                    best_choice = choice
                    continue

                best_metrics = best_choice["metrics"]
                better_score = cand_score > float(best_choice["score"]) + 1e-12
                calibration_tie = (
                    cand_score >= float(best_choice["score"]) - 0.002
                    and cand_metrics.get("ECE", 1.0) <= best_metrics.get("ECE", 1.0)
                    and cand_metrics.get("Brier", 1.0) <= best_metrics.get("Brier", 1.0)
                )
                if better_score or calibration_tie:
                    best_choice = choice

        if best_choice is None:
            raise RuntimeError("No RCF candidate variant was evaluated.")
        return best_choice

    def _select_auto_blend(
        self,
        y_val: np.ndarray,
        student_val_prob: np.ndarray,
        teacher_val_prob: Optional[np.ndarray],
    ):
        candidate_alphas = self._parse_alpha_grid() if teacher_val_prob is not None else [0.0]
        best: Optional[Dict[str, Any]] = None

        for alpha in candidate_alphas:
            raw_val_prob = self._blend_student_teacher(student_val_prob, teacher_val_prob, alpha)
            calibrator = None
            if getattr(self.cfg, "use_post_calibration", True):
                calibrator = ProbabilityCalibrator().fit(y_val, raw_val_prob)
                val_prob_for_threshold = calibrator.transform(raw_val_prob)
            else:
                val_prob_for_threshold = raw_val_prob

            if self.cfg.use_threshold_search:
                threshold = MetricUtils.find_best_threshold(
                    y_val,
                    val_prob_for_threshold,
                    metric=self.cfg.threshold_metric,
                )
            else:
                threshold = 0.5

            val_metrics = MetricUtils.summarize(y_val, val_prob_for_threshold, threshold=threshold)
            score = MetricUtils.composite_score(
                val_metrics,
                mode=getattr(self.cfg, "blend_selection_metric", "balanced"),
            )
            if best is None or score > best["score"]:
                best = {
                    "alpha": float(alpha),
                    "calibrator": calibrator,
                    "threshold": float(threshold),
                    "metrics": val_metrics,
                    "score": float(score),
                }

        if best is None:
            return {
                "alpha": 0.0,
                "calibrator": None,
                "threshold": 0.5,
                "metrics": {},
                "score": float("nan"),
            }
        return best

    def _record_teacher_state_analysis(
        self,
        dataset_name: str,
        fold_id: int,
        original_ids: pd.Series,
        y_true: np.ndarray,
        teacher_prob: Optional[np.ndarray],
        teacher_stack: Optional[np.ndarray],
    ):
        """记录 OOF 教师真实状态，用于验证“熵不等于可靠性”。"""
        if not getattr(self.cfg, "enable_teacher_state_analysis", True) or teacher_prob is None:
            return
        y_arr = np.asarray(y_true).astype(int)
        p = sanitize_probabilities(teacher_prob)
        if len(y_arr) != len(p):
            return
        ent = np.clip(KDDistillationUtils.norm_entropy_np(p), 0.0, 1.0)
        rel = y_arr * p + (1 - y_arr) * (1.0 - p)
        pred = (p >= 0.5).astype(int)
        correct = (pred == y_arr)
        temp = KDDistillationUtils.temperature_from_prob(p, self.cfg.t_min, self.cfg.t_max, self.cfg.gamma)
        tau = float(getattr(self.cfg, "reliability_tau", 0.5))
        gate = np.clip((rel - tau) / max(1.0 - tau, 1e-6), 0.0, 1.0)
        power = float(getattr(self.cfg, "reliability_power", 1.0))
        if abs(power - 1.0) > 1e-8:
            gate = gate ** max(power, 0.0)
        lam = float(self.cfg.lambda_min) + (float(self.cfg.lambda_max) - float(self.cfg.lambda_min)) * gate
        q = float(np.clip(getattr(self.cfg, "teacher_state_quantile", 0.25), 0.05, 0.45))
        q_low, q_high = np.quantile(ent, [q, 1.0 - q])
        disagreement = np.zeros(len(y_arr), dtype=float)
        if teacher_stack is not None:
            st = np.asarray(teacher_stack, dtype=float)
            if st.ndim == 2 and st.shape[1] == len(y_arr):
                disagreement = np.std(st, axis=0)
        ids = pd.Series(original_ids).reset_index(drop=True)
        for i in range(len(y_arr)):
            entropy_band = "middle"
            if ent[i] <= q_low:
                entropy_band = "low"
            elif ent[i] >= q_high:
                entropy_band = "high"
            if entropy_band in {"low", "high"}:
                state = f"{entropy_band}_entropy_{'correct' if correct[i] else 'wrong'}"
            else:
                state = f"middle_entropy_{'correct' if correct[i] else 'wrong'}"
            self.all_teacher_state_records.append({
                "dataset": dataset_name,
                "fold": int(fold_id),
                "original_row_id": int(ids.iloc[i]) if i < len(ids) else int(i),
                "y_true": int(y_arr[i]),
                "teacher_prob": float(p[i]),
                "teacher_pred": int(pred[i]),
                "correct": int(correct[i]),
                "entropy": float(ent[i]),
                "reliability_true_class": float(rel[i]),
                "temperature_estd": float(temp[i]),
                "lambda_reliability": float(lam[i]),
                "teacher_disagreement_std": float(disagreement[i]),
                "entropy_q_low": float(q_low),
                "entropy_q_high": float(q_high),
                "entropy_band": entropy_band,
                "state": state,
                "is_minority": int(y_arr[i] == 1),
            })

    def _save_teacher_state_outputs(self):
        if not self.all_teacher_state_records:
            return
        df = pd.DataFrame(self.all_teacher_state_records)
        df.to_csv(self.metrics_dir / "teacher_state_samples.csv", index=False, encoding="utf-8-sig")
        four = df[df["entropy_band"].isin(["low", "high"])].copy()
        if four.empty:
            return
        summary = four.groupby(["dataset", "state"]).agg(
            count=("state", "size"),
            minority_ratio=("is_minority", "mean"),
            mean_entropy=("entropy", "mean"),
            mean_reliability=("reliability_true_class", "mean"),
            mean_temperature=("temperature_estd", "mean"),
            mean_lambda=("lambda_reliability", "mean"),
            mean_teacher_disagreement=("teacher_disagreement_std", "mean"),
        ).reset_index()
        summary.to_csv(self.metrics_dir / "teacher_state_summary_by_dataset.csv", index=False, encoding="utf-8-sig")
        overall = four.groupby(["state"]).agg(
            count=("state", "size"),
            minority_ratio=("is_minority", "mean"),
            mean_entropy=("entropy", "mean"),
            mean_reliability=("reliability_true_class", "mean"),
            mean_temperature=("temperature_estd", "mean"),
            mean_lambda=("lambda_reliability", "mean"),
            mean_teacher_disagreement=("teacher_disagreement_std", "mean"),
        ).reset_index()
        overall.to_csv(self.metrics_dir / "teacher_state_summary_overall.csv", index=False, encoding="utf-8-sig")

    def _append_pred_records(
        self,
        container: List[Dict[str, Any]],
        dataset_name: str,
        fold_id: int,
        method_name: str,
        original_ids: pd.Series,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        threshold: float,
    ):
        for idx in range(len(y_true)):
            prob = float(y_prob[idx])
            entropy = float(-(prob * math.log(max(prob, 1e-8)) + (1.0 - prob) * math.log(max(1.0 - prob, 1e-8))) / math.log(2.0))
            container.append({
                "dataset": dataset_name,
                "fold": fold_id,
                "method": method_name,
                "original_row_id": int(original_ids.iloc[idx]),
                "y_true": int(y_true[idx]),
                "y_prob": prob,
                "threshold": float(threshold),
                "y_pred": int(prob >= threshold),
                "y_pred_05": int(prob >= 0.5),
                "confidence": float(max(prob, 1.0 - prob)),
                "uncertainty_entropy": entropy,
                "prob_margin": float(abs(prob - 0.5) * 2.0),
            })

    def run(self, datasets: Dict[str, pd.DataFrame]):
        self.save_dataset_summary(datasets)
        all_records: List[Dict[str, Any]] = []
        all_pred_records: List[Dict[str, Any]] = []
        base_random_state = int(self.cfg.random_state)

        methods = self.method_list()
        print("\n将运行的方法：", methods)

        completed_folds = set()
        if bool(getattr(self.cfg, "resume_from_partial", False)):
            metrics_partial = self.metrics_dir / "all_metrics.partial.csv"
            preds_partial = self.preds_dir / "all_predictions.partial.csv"
            if metrics_partial.exists():
                prior_metrics = pd.read_csv(metrics_partial)
                trace_columns = {"dataset", "repeat_id", "fold_id", "method"}
                if trace_columns.issubset(prior_metrics.columns):
                    expected_methods = set(methods)
                    for key, group in prior_metrics.groupby(["dataset", "repeat_id", "fold_id"]):
                        if set(group["method"].astype(str)) == expected_methods and len(group) == len(expected_methods):
                            completed_folds.add((str(key[0]), int(key[1]), int(key[2])))

                    def keep_completed(frame: pd.DataFrame) -> pd.DataFrame:
                        required = {"dataset", "repeat_id", "fold_id"}
                        if frame.empty or not required.issubset(frame.columns):
                            return frame.iloc[0:0].copy()
                        keep = []
                        for _, row in frame.iterrows():
                            dataset_value = str(row["dataset"])
                            repeat_value = int(row["repeat_id"])
                            fold_value = int(row["fold_id"])
                            matched = any(
                                repeat_value == repeat_id
                                and fold_value == fold_id
                                and (dataset_value == dataset or dataset_value.startswith(dataset + "_"))
                                for dataset, repeat_id, fold_id in completed_folds
                            )
                            keep.append(matched)
                        return frame.loc[keep].copy()

                    all_records = keep_completed(prior_metrics).to_dict("records")
                    if preds_partial.exists():
                        all_pred_records = keep_completed(pd.read_csv(preds_partial)).to_dict("records")
                    teacher_partial = self.metrics_dir / "teacher_quality.partial.csv"
                    if teacher_partial.exists():
                        self.all_teacher_records = keep_completed(pd.read_csv(teacher_partial)).to_dict("records")
                    runtime_partial = self.metrics_dir / "runtime_summary.partial.csv"
                    if runtime_partial.exists():
                        self.all_time_records = keep_completed(pd.read_csv(runtime_partial)).to_dict("records")
                    state_partial = self.metrics_dir / "teacher_state_samples.partial.csv"
                    if state_partial.exists():
                        self.all_teacher_state_records = keep_completed(pd.read_csv(state_partial)).to_dict("records")
                    print(f"[RESUME] 已恢复 {len(completed_folds)} 个完整外层折。")

        for ds_name, df in datasets.items():
            print(
                f"\n========== 数据集：{ds_name} | γ={self.cfg.gamma}, "
                f"λ_min={self.cfg.lambda_min}, λ_max={self.cfg.lambda_max}, Tmax={self.cfg.t_max}, KD-CB-decoupled={getattr(self.cfg, 'estd_decouple_kd_class_weight', False)}, "
                f"μ={self.cfg.mu_brier}, student={self.cfg.student_type} =========="
            )

            X = df.drop(columns=["target", "original_row_id"]).copy()
            original_row_ids = df["original_row_id"].copy()
            y = df["target"].values.astype(int)

            # 如果某一类样本太少，自动降低 n_splits
            min_class = int(np.min(np.bincount(y, minlength=2)))
            n_splits = min(self.cfg.n_splits, min_class)
            if n_splits < 2:
                print(f"[WARN] 数据集 {ds_name} 某类样本过少，跳过。")
                continue

            split_plan = []
            for repeat_id in range(1, self.cfg.n_repeats + 1):
                repeat_seed = base_random_state + repeat_id - 1
                repeat_cv = StratifiedKFold(
                    n_splits=n_splits,
                    shuffle=True,
                    random_state=repeat_seed,
                )
                for fold_in_repeat, (train_idx, test_idx) in enumerate(repeat_cv.split(X, y), start=1):
                    split_plan.append((repeat_id, fold_in_repeat, repeat_seed, train_idx, test_idx))

            for fold_id, (repeat_id, fold_in_repeat, repeat_seed, train_idx, test_idx) in enumerate(split_plan, start=1):
                if (ds_name, int(repeat_id), int(fold_in_repeat)) in completed_folds:
                    print(
                        f"[RESUME] 跳过已完成折：{ds_name} | repeat={repeat_id} | fold={fold_in_repeat}"
                    )
                    continue
                self.cfg.random_state = repeat_seed
                fold_record_start = len(all_records)
                fold_pred_start = len(all_pred_records)
                fold_teacher_start = len(self.all_teacher_records)
                fold_time_start = len(self.all_time_records)
                fold_state_start = len(self.all_teacher_state_records)
                print(
                    f"\n--- {ds_name} | Repeat {repeat_id}/{self.cfg.n_repeats} | "
                    f"Fold {fold_in_repeat}/{n_splits} | seed={repeat_seed} ---"
                )
                X_train = X.iloc[train_idx].reset_index(drop=True)
                X_test = X.iloc[test_idx].reset_index(drop=True)
                y_train = y[train_idx]
                y_test = y[test_idx]
                test_original_ids = original_row_ids.iloc[test_idx].reset_index(drop=True)
                train_original_ids = original_row_ids.iloc[train_idx].reset_index(drop=True)

                X_sub, X_val, y_sub, y_val, valid_split = self._make_val_split(X_train, y_train)

                # 为避免每个 KD 方法重复训练教师，这里每个外层 fold 只预先构建一次教师概率：
                # 1) subtrain -> val：用于选择分类阈值；
                # 2) full train -> test：用于最终测试预测。
                baseline_method_set = {"LR", "SVM", "RF", "XGBoost", "BalancedRF", "EasyEnsemble"}
                needs_ensemble_teacher = any(
                    (m not in baseline_method_set and m != "Student") for m in methods
                )
                needs_single_teacher = any(m in {"ATD", "Yang-ATD-adapted", "CalibKD"} for m in methods)
                needs_uncal_teacher = any(m in {"ESTD-KD (w/o teacher calibration)", "TeacherEns"} for m in methods)
                if any(m == "ESTD-KD-RCF" for m in methods) and getattr(
                    self.cfg, "rcf_include_raw_teacher_candidate", True
                ):
                    needs_uncal_teacher = True

                ens_val_train_prob = None
                ens_val_pred_prob = None
                ens_val_train_stack = None
                ens_val_pred_stack = None
                ens_val_teacher_names: List[str] = []
                ens_test_train_prob = None
                ens_test_pred_prob = None
                ens_test_train_stack = None
                ens_test_pred_stack = None
                ens_test_teacher_names: List[str] = []
                single_val_train_prob = None
                single_test_train_prob = None
                uncal_val_train_prob = None
                uncal_val_pred_prob = None
                uncal_test_train_prob = None
                uncal_test_pred_prob = None

                if needs_ensemble_teacher:
                    print("预构建教师概率：validation teacher ...")
                    runner_teacher_val = MethodRunner(self.cfg)
                    ens_val_train_prob, ens_val_pred_prob, _ = runner_teacher_val.train_teacher_ensemble(
                        X_sub,
                        y_sub,
                        X_val,
                        dataset_name=ds_name + "_val",
                        fold_id=fold_id,
                        calibrate=self.cfg.use_teacher_calibration,
                    )
                    ens_val_train_stack = runner_teacher_val.last_train_teacher_stack
                    ens_val_pred_stack = runner_teacher_val.last_test_teacher_stack
                    ens_val_teacher_names = list(runner_teacher_val.last_teacher_names)
                    self.all_teacher_records.extend(runner_teacher_val.teacher_records)

                    print("预构建教师概率：test teacher ...")
                    runner_teacher_test = MethodRunner(self.cfg)
                    ens_test_train_prob, ens_test_pred_prob, _ = runner_teacher_test.train_teacher_ensemble(
                        X_train,
                        y_train,
                        X_test,
                        dataset_name=ds_name,
                        fold_id=fold_id,
                        calibrate=self.cfg.use_teacher_calibration,
                    )
                    ens_test_train_stack = runner_teacher_test.last_train_teacher_stack
                    ens_test_pred_stack = runner_teacher_test.last_test_teacher_stack
                    ens_test_teacher_names = list(runner_teacher_test.last_teacher_names)
                    self.all_teacher_records.extend(runner_teacher_test.teacher_records)

                    self._record_teacher_state_analysis(
                        ds_name,
                        fold_id,
                        train_original_ids,
                        y_train,
                        ens_test_train_prob,
                        ens_test_train_stack,
                    )

                if needs_single_teacher:
                    print("预构建单教师概率：validation/test ...")
                    runner_single_val = MethodRunner(self.cfg)
                    single_val_t0 = time.time()
                    single_val_train_prob, _ = runner_single_val.train_single_teacher(
                        X_sub, y_sub, X_val, calibrate=self.cfg.use_teacher_calibration
                    )
                    self.all_teacher_records.extend(runner_single_val.teacher_records)
                    self.all_teacher_records.append({
                        "dataset": ds_name + "_val",
                        "fold": fold_id,
                        "teacher": "RF_single",
                        "calibrated": bool(self.cfg.use_teacher_calibration),
                        "oof": bool(self.cfg.use_teacher_oof),
                        "time_sec": float(time.time() - single_val_t0),
                        "preparation_role": "single_teacher_validation",
                    })

                    runner_single_test = MethodRunner(self.cfg)
                    single_test_t0 = time.time()
                    single_test_train_prob, _ = runner_single_test.train_single_teacher(
                        X_train, y_train, X_test, calibrate=self.cfg.use_teacher_calibration
                    )
                    self.all_teacher_records.extend(runner_single_test.teacher_records)
                    self.all_teacher_records.append({
                        "dataset": ds_name,
                        "fold": fold_id,
                        "teacher": "RF_single",
                        "calibrated": bool(self.cfg.use_teacher_calibration),
                        "oof": bool(self.cfg.use_teacher_oof),
                        "time_sec": float(time.time() - single_test_t0),
                        "preparation_role": "single_teacher_test",
                    })

                if needs_uncal_teacher:
                    print("预构建未校准教师概率：validation/test ...")
                    runner_uncal_val = MethodRunner(self.cfg)
                    uncal_val_train_prob, uncal_val_pred_prob, _ = runner_uncal_val.train_teacher_ensemble(
                        X_sub, y_sub, X_val, dataset_name=ds_name + "_val_uncal", fold_id=fold_id, calibrate=False
                    )
                    self.all_teacher_records.extend(runner_uncal_val.teacher_records)

                    runner_uncal_test = MethodRunner(self.cfg)
                    uncal_test_train_prob, uncal_test_pred_prob, _ = runner_uncal_test.train_teacher_ensemble(
                        X_train, y_train, X_test, dataset_name=ds_name + "_uncal", fold_id=fold_id, calibrate=False
                    )
                    self.all_teacher_records.extend(runner_uncal_test.teacher_records)

                # Several RCF ablations share the exact same trained student and
                # differ only in the downstream fusion/selection rule. Reuse the
                # deterministic fold predictions so the formal ablation does not
                # retrain an identical network multiple times.
                rcf_student_val_cache: Dict[Tuple[Any, ...], np.ndarray] = {}
                rcf_student_test_cache: Dict[Tuple[Any, ...], np.ndarray] = {}

                for method in methods:
                    t0 = time.time()
                    print(f"运行方法：{method}")

                    if method == "ESTD-KD-Sweep":
                        try:
                            runner_val = MethodRunner(self.cfg)
                            student_val_prob = runner_val.run_estd_kd_v2(
                                X_sub,
                                y_sub,
                                X_val,
                                ens_val_train_prob,
                                variant="full",
                                teacher_stack_train=ens_val_train_stack,
                            )
                            runner_test = MethodRunner(self.cfg)
                            student_test_prob = runner_test.run_estd_kd_v2(
                                X_train,
                                y_train,
                                X_test,
                                ens_test_train_prob,
                                variant="full",
                                teacher_stack_train=ens_test_train_stack,
                            )
                        except Exception as e:
                            print(f"[ERROR] {method} 学生训练失败，已跳过。错误：{e}")
                            continue

                        for alpha in self._parse_alpha_grid():
                            method_name = f"ESTD-KD+a{alpha:.2f}"
                            try:
                                val_prob = self._blend_student_teacher(student_val_prob, ens_val_pred_prob, alpha)
                                calibrator = None
                                if getattr(self.cfg, "use_post_calibration", True):
                                    calibrator = ProbabilityCalibrator().fit(y_val, val_prob)
                                    val_prob_for_threshold = calibrator.transform(val_prob)
                                else:
                                    val_prob_for_threshold = val_prob

                                if self.cfg.use_threshold_search:
                                    threshold = MetricUtils.find_best_threshold(
                                        y_val,
                                        val_prob_for_threshold,
                                        metric=self.cfg.threshold_metric,
                                    )
                                else:
                                    threshold = 0.5
                                val_metrics = MetricUtils.summarize(y_val, val_prob_for_threshold, threshold=threshold)

                                test_prob = self._blend_student_teacher(student_test_prob, ens_test_pred_prob, alpha)
                                if getattr(self.cfg, "use_post_calibration", True) and calibrator is not None:
                                    test_prob = calibrator.transform(test_prob)

                                metrics = MetricUtils.summarize(y_test, test_prob, threshold=threshold)
                                record = {
                                    "dataset": ds_name,
                                    "fold": fold_id,
                                    "method": method_name,
                                    "blend_alpha": float(alpha),
                                    "blend_val_score": MetricUtils.composite_score(
                                        val_metrics,
                                        mode=getattr(self.cfg, "blend_selection_metric", "balanced"),
                                    ),
                                    "blend_val_MCC": val_metrics.get("MCC", np.nan),
                                    "blend_val_F1": val_metrics.get("F1", np.nan),
                                    "blend_val_GMean": val_metrics.get("GMean", np.nan),
                                    "blend_val_ECE": val_metrics.get("ECE", np.nan),
                                    "blend_val_Brier": val_metrics.get("Brier", np.nan),
                                    **metrics,
                                }
                                all_records.append(record)
                                self._append_pred_records(
                                    all_pred_records,
                                    ds_name,
                                    fold_id,
                                    method_name,
                                    test_original_ids,
                                    y_test,
                                    test_prob,
                                    threshold,
                                )

                                elapsed = time.time() - t0
                                self.all_time_records.append({
                                    "dataset": ds_name,
                                    "fold": fold_id,
                                    "method": method_name,
                                    "time_sec": float(elapsed),
                                })

                                print(
                                    f"  {method_name}: MCC={metrics['MCC']:.4f}, F1={metrics['F1']:.4f}, "
                                    f"GMean={metrics['GMean']:.4f}, ECE={metrics['ECE']:.4f}, "
                                    f"Brier={metrics['Brier']:.4f}, th={threshold:.3f}, time={elapsed:.1f}s"
                                )

                            except Exception as e:
                                print(f"[ERROR] {method_name} 评估失败，已跳过。错误：{e}")
                        continue

                    if method == "ESTD-KD-Auto":
                        selected_alpha = float(np.clip(getattr(self.cfg, "blend_teacher_weight", 0.25), 0.0, 1.0))
                        calibrator = None
                        threshold = 0.5
                        val_score = float("nan")
                        val_metrics: Dict[str, float] = {}

                        try:
                            runner_val = MethodRunner(self.cfg)
                            student_val_prob = runner_val.run_estd_kd_v2(
                                X_sub,
                                y_sub,
                                X_val,
                                ens_val_train_prob,
                                variant="full",
                                teacher_stack_train=ens_val_train_stack,
                            )
                            choice = self._select_auto_blend(y_val, student_val_prob, ens_val_pred_prob)
                            selected_alpha = float(choice["alpha"])
                            calibrator = choice["calibrator"]
                            threshold = float(choice["threshold"])
                            val_score = float(choice["score"])
                            val_metrics = dict(choice["metrics"])
                            print(
                                f"  AutoBlend验证：alpha={selected_alpha:.3f}, "
                                f"score={val_score:.4f}, val_MCC={val_metrics.get('MCC', np.nan):.4f}, "
                                f"val_ECE={val_metrics.get('ECE', np.nan):.4f}, th={threshold:.3f}"
                            )
                        except Exception as e:
                            print(f"[WARN] {method} 验证阶段失败，使用 alpha={selected_alpha:.3f}, threshold=0.5。错误：{e}")

                        try:
                            runner_test = MethodRunner(self.cfg)
                            student_test_prob = runner_test.run_estd_kd_v2(
                                X_train,
                                y_train,
                                X_test,
                                ens_test_train_prob,
                                variant="full",
                                teacher_stack_train=ens_test_train_stack,
                            )
                            test_prob = self._blend_student_teacher(student_test_prob, ens_test_pred_prob, selected_alpha)
                            if getattr(self.cfg, "use_post_calibration", True) and calibrator is not None:
                                test_prob = calibrator.transform(test_prob)

                            metrics = MetricUtils.summarize(y_test, test_prob, threshold=threshold)
                            record = {
                                "dataset": ds_name,
                                "fold": fold_id,
                                "method": method,
                                "blend_alpha": selected_alpha,
                                "blend_val_score": val_score,
                                "blend_val_MCC": val_metrics.get("MCC", np.nan),
                                "blend_val_F1": val_metrics.get("F1", np.nan),
                                "blend_val_GMean": val_metrics.get("GMean", np.nan),
                                "blend_val_ECE": val_metrics.get("ECE", np.nan),
                                "blend_val_Brier": val_metrics.get("Brier", np.nan),
                                **metrics,
                            }
                            all_records.append(record)
                            self._append_pred_records(
                                all_pred_records,
                                ds_name,
                                fold_id,
                                method,
                                test_original_ids,
                                y_test,
                                test_prob,
                                threshold,
                            )

                            elapsed = time.time() - t0
                            self.all_time_records.append({
                                "dataset": ds_name,
                                "fold": fold_id,
                                "method": method,
                                "time_sec": float(elapsed),
                            })

                            print(
                                f"  {method}: MCC={metrics['MCC']:.4f}, F1={metrics['F1']:.4f}, "
                                f"GMean={metrics['GMean']:.4f}, ECE={metrics['ECE']:.4f}, "
                                f"Brier={metrics['Brier']:.4f}, alpha={selected_alpha:.3f}, "
                                f"th={threshold:.3f}, time={elapsed:.1f}s"
                            )

                        except Exception as e:
                            print(f"[ERROR] {method} 测试阶段失败，已跳过。错误：{e}")
                        continue

                    gamma_tau_grid = {
                        "ESTD-KD-RCF-g0p5-t0p4": (0.5, 0.4),
                        "ESTD-KD-RCF-g0p5-t0p6": (0.5, 0.6),
                        "ESTD-KD-RCF-g0p5-t0p7": (0.5, 0.7),
                        "ESTD-KD-RCF-g2p0-t0p4": (2.0, 0.4),
                        "ESTD-KD-RCF-g2p0-t0p6": (2.0, 0.6),
                        "ESTD-KD-RCF-g2p0-t0p7": (2.0, 0.7),
                    }
                    rcf_ablation_specs = {
                        "ESTD-KD-RCF (w/o adaptive temperature)": {"student_variant": "no_adaptive_temp"},
                        "ESTD-KD-RCF (w/o global reliability KD strength)": {"student_variant": "no_global_reliability"},
                        "ESTD-KD-RCF (w/o Brier)": {"student_variant": "no_brier"},
                        "ESTD-KD-RCF (fixed fusion)": {"student_variant": "full", "fixed_only": True},
                        "ESTD-KD-RCF (w/o candidate pool)": {"student_variant": "full", "no_candidate_pool": True},
                        # Legacy diagnostics retained for explicit --methods use only; not included in final paper suite.
                        "ESTD-KD-RCF (legacy sample-wise reliability KD weight)": {"student_variant": "legacy_samplewise_reliability"},
                        "ESTD-KD-RCF (legacy reliable soft target)": {"student_variant": "legacy_reliable_soft_target"},
                    }
                    architecture_specs = {
                        "ESTD-KD-RCF [Linear]": ("linear", 64, 32),
                        "ESTD-KD-RCF [MLP-32]": ("mlp", 32, 0),
                        "ESTD-KD-RCF [MLP-64-32]": ("mlp", 64, 32),
                        "ESTD-KD-RCF [MLP-128-64]": ("mlp", 128, 64),
                    }
                    alpha_bounds_specs: Dict[str, Tuple[float, float]] = {}
                    if method.startswith("ESTD-KD-RCF [a"):
                        match = re.match(r"ESTD-KD-RCF \[a([0-9.]+)-([0-9.]+)\]", method)
                        if match:
                            alpha_bounds_specs[method] = (float(match.group(1)), float(match.group(2)))

                    is_rcf_method = (
                        method == "ESTD-KD-RCF"
                        or method in gamma_tau_grid
                        or method in rcf_ablation_specs
                        or method in architecture_specs
                        or method in alpha_bounds_specs
                    )
                    if is_rcf_method:
                        method_cfg = self.cfg
                        ablation_spec = rcf_ablation_specs.get(method, {})
                        student_variant = str(ablation_spec.get("student_variant", "full"))
                        fixed_only = bool(ablation_spec.get("fixed_only", False))
                        no_candidate_pool = bool(ablation_spec.get("no_candidate_pool", False))
                        # final controlled ablation：只回退单个旧行为，其他配置保持 v7。
                        cfg_overrides = {}
                        if "use_reliable_soft_label" in ablation_spec:
                            cfg_overrides["use_reliable_soft_label"] = bool(ablation_spec["use_reliable_soft_label"])
                        if "lambda_min" in ablation_spec:
                            cfg_overrides["lambda_min"] = float(ablation_spec["lambda_min"])
                        if cfg_overrides:
                            method_cfg = dataclass_replace(method_cfg, **cfg_overrides)
                        if method in gamma_tau_grid:
                            grid_gamma, grid_tau = gamma_tau_grid[method]
                            method_cfg = dataclass_replace(
                                method_cfg, gamma=float(grid_gamma), reliability_tau=float(grid_tau)
                            )
                        if method in architecture_specs:
                            stype, h1, h2 = architecture_specs[method]
                            method_cfg = dataclass_replace(
                                method_cfg, student_type=stype, mlp_hidden1=int(h1), mlp_hidden2=int(h2)
                            )
                        if method in alpha_bounds_specs:
                            amin, amax = alpha_bounds_specs[method]
                            method_cfg = dataclass_replace(
                                method_cfg, rcf_min_alpha=float(amin), rcf_max_alpha=float(amax)
                            )
                        calibrator = None
                        threshold = 0.5
                        val_metrics: Dict[str, float] = {}
                        val_score = float("nan")
                        rcf_head = None
                        val_alpha_mean = float("nan")
                        val_alpha_std = float("nan")
                        selected_candidate = "fixed"
                        selected_calibration_mode = "post"
                        default_alpha = float(np.clip(getattr(self.cfg, "blend_teacher_weight", 0.90), 0.0, 1.0))
                        student_cache_key = (
                            round(float(method_cfg.gamma), 12),
                            round(float(method_cfg.reliability_tau), 12),
                            student_variant,
                            str(method_cfg.student_type),
                            int(method_cfg.mlp_hidden1),
                            int(method_cfg.mlp_hidden2),
                            bool(method_cfg.use_reliable_soft_label),
                            bool(method_cfg.use_reliability_lambda),
                            bool(getattr(method_cfg, "estd_decouple_kd_class_weight", True)),
                            round(float(method_cfg.lambda_min), 12),
                            round(float(method_cfg.lambda_max), 12),
                            bool(method_cfg.use_brier_loss),
                        )

                        try:
                            if student_cache_key in rcf_student_val_cache:
                                student_val_prob = rcf_student_val_cache[student_cache_key].copy()
                            else:
                                runner_val = MethodRunner(method_cfg)
                                student_val_prob = runner_val.run_estd_kd_v2(
                                    X_sub,
                                    y_sub,
                                    X_val,
                                    ens_val_train_prob,
                                    variant=student_variant,
                                )
                                rcf_student_val_cache[student_cache_key] = np.asarray(student_val_prob).copy()
                            X_val_feat = self._rcf_features(
                                student_val_prob,
                                ens_val_pred_prob,
                                ens_val_pred_stack,
                            )
                            rcf_head = self._fit_rcf_head(y_val, X_val_feat, cfg=method_cfg)
                            val_prob, val_alpha, _ = self._predict_rcf_blend(
                                rcf_head, X_val_feat, student_val_prob, ens_val_pred_prob, cfg=method_cfg
                            )
                            val_alpha_mean = float(np.mean(val_alpha))
                            val_alpha_std = float(np.std(val_alpha))

                            fixed_val_prob = self._blend_student_teacher(
                                student_val_prob,
                                ens_val_pred_prob,
                                default_alpha,
                            )
                            candidates = [
                                ("fixed", fixed_val_prob, np.full(len(student_val_prob), default_alpha, dtype=float)),
                            ]
                            guard_min_class = int(getattr(self.cfg, "rcf_teacher_guard_min_class", 40))
                            sub_min_class = int(np.min(np.bincount(np.asarray(y_sub).astype(int), minlength=2)))
                            allow_teacher_candidates = (
                                getattr(self.cfg, "rcf_enable_teacher_candidate", True)
                                and (guard_min_class <= 0 or sub_min_class > guard_min_class)
                                and not fixed_only
                                and not no_candidate_pool
                            )
                            if allow_teacher_candidates:
                                candidates.append((
                                    "teacher",
                                    sanitize_probabilities(ens_val_pred_prob),
                                    np.ones(len(student_val_prob), dtype=float),
                                ))
                                if (
                                    getattr(self.cfg, "rcf_enable_teacher_pool_candidate", True)
                                    and ens_val_pred_stack is not None
                                    and len(ens_val_teacher_names) == np.asarray(ens_val_pred_stack).shape[0]
                                ):
                                    for teacher_idx, teacher_name in enumerate(ens_val_teacher_names):
                                        candidates.append((
                                            f"teacher_pool:{teacher_name}",
                                            sanitize_probabilities(np.asarray(ens_val_pred_stack)[teacher_idx]),
                                            np.ones(len(student_val_prob), dtype=float),
                                        ))
                                if (
                                    getattr(self.cfg, "rcf_include_raw_teacher_candidate", True)
                                    and uncal_val_pred_prob is not None
                                ):
                                    candidates.append((
                                        "teacher_raw",
                                        sanitize_probabilities(uncal_val_pred_prob),
                                        np.ones(len(student_val_prob), dtype=float),
                                    ))
                            if rcf_head is not None and not fixed_only:
                                candidates.append(("rcf", val_prob, val_alpha))
                            if getattr(self.cfg, "rcf_enable_student_guard", False):
                                candidates.append((
                                    "student",
                                    sanitize_probabilities(student_val_prob),
                                    np.zeros(len(student_val_prob), dtype=float),
                                ))

                            if no_candidate_pool and rcf_head is not None:
                                # Keep the reliability fusion head but remove validation-time
                                # selection among fixed/teacher/RCF candidates.
                                candidates = [("rcf", val_prob, val_alpha)]

                            fixed_choice: Optional[Dict[str, Any]] = None
                            evaluated_choices: List[Dict[str, Any]] = []
                            best_choice: Optional[Dict[str, Any]] = None
                            for cand_name, cand_prob, cand_alpha in candidates:
                                choice = self._evaluate_rcf_candidate(
                                    y_val,
                                    cand_name,
                                    cand_prob,
                                    cand_alpha,
                                )
                                evaluated_choices.append(choice)
                                if cand_name == "fixed":
                                    fixed_choice = choice

                            if no_candidate_pool and evaluated_choices:
                                best_choice = evaluated_choices[0]
                            elif fixed_choice is not None:
                                best_choice = fixed_choice
                                fixed_metrics = fixed_choice["metrics"]
                                fixed_score = float(fixed_choice["score"])
                                fixed_mcc = float(fixed_metrics.get("MCC", 0.0))
                                fixed_gmean = float(fixed_metrics.get("GMean", 0.0))
                                fixed_ece = float(fixed_metrics.get("ECE", 0.0))
                                fixed_brier = float(fixed_metrics.get("Brier", 0.0))
                                min_gain = float(getattr(self.cfg, "rcf_candidate_min_gain", 0.0))
                                teacher_gain = float(getattr(self.cfg, "rcf_teacher_min_gain", 0.0))
                                teacher_tie_tol = float(getattr(self.cfg, "rcf_teacher_tie_tol", 0.01))
                                teacher_pool_ece_tol = float(getattr(self.cfg, "rcf_teacher_pool_ece_tol", 0.04))
                                teacher_pool_brier_tol = float(getattr(self.cfg, "rcf_teacher_pool_brier_tol", 0.02))
                                dynamic_ece_tol = float(getattr(self.cfg, "rcf_dynamic_ece_tol", 0.03))
                                dynamic_brier_tol = float(getattr(self.cfg, "rcf_dynamic_brier_tol", 0.015))
                                dynamic_tie_tol = float(getattr(self.cfg, "rcf_dynamic_tie_tol", 0.005))
                                student_min_gain = float(getattr(self.cfg, "rcf_student_min_gain", 0.05))
                                student_mcc_gain = float(getattr(self.cfg, "rcf_student_mcc_gain", 0.05))
                                student_max_fixed_mcc = float(getattr(self.cfg, "rcf_student_max_fixed_mcc", 0.55))
                                use_pareto_guard = bool(getattr(self.cfg, "rcf_pareto_guard", False))
                                pareto_min_gain = float(getattr(self.cfg, "rcf_pareto_min_gain", 0.005))
                                pareto_ece_tol = float(getattr(self.cfg, "rcf_pareto_ece_tol", 0.025))
                                pareto_brier_tol = float(getattr(self.cfg, "rcf_pareto_brier_tol", 0.012))
                                use_dual_pareto = bool(getattr(self.cfg, "rcf_dual_pareto_guard", False))
                                dual_mcc_tol = float(getattr(self.cfg, "rcf_dual_mcc_tol", 0.005))
                                dual_gmean_tol = float(getattr(self.cfg, "rcf_dual_gmean_tol", 0.010))

                                for choice in evaluated_choices:
                                    cand_name = str(choice["name"])
                                    if cand_name == "fixed":
                                        continue
                                    cand_family = "teacher" if cand_name.startswith("teacher") else cand_name
                                    is_teacher_pool = cand_name.startswith("teacher_pool:")
                                    cand_metrics = choice["metrics"]
                                    cand_score = float(choice["score"])
                                    cand_mcc = float(cand_metrics.get("MCC", 0.0))
                                    cand_gmean = float(cand_metrics.get("GMean", 0.0))
                                    cand_ece = float(cand_metrics.get("ECE", 1.0))
                                    cand_brier = float(cand_metrics.get("Brier", 1.0))

                                    allowed = cand_score >= fixed_score + min_gain
                                    if cand_family == "teacher":
                                        allowed = cand_score >= fixed_score + teacher_gain
                                        if is_teacher_pool:
                                            allowed = (
                                                allowed
                                                and cand_ece <= fixed_ece + teacher_pool_ece_tol
                                                and cand_brier <= fixed_brier + teacher_pool_brier_tol
                                            )
                                        if not allowed:
                                            allowed = (
                                                (not is_teacher_pool)
                                                and cand_score >= fixed_score - teacher_tie_tol
                                                and cand_ece <= fixed_ece
                                                and cand_brier <= fixed_brier + dynamic_brier_tol
                                            )
                                    elif cand_family == "rcf":
                                        allowed = (
                                            allowed
                                            and cand_ece <= fixed_ece + dynamic_ece_tol
                                            and cand_brier <= fixed_brier + dynamic_brier_tol
                                        )
                                    elif cand_family == "student":
                                        allowed = (
                                            fixed_mcc <= student_max_fixed_mcc
                                            and cand_score >= fixed_score + student_min_gain
                                            and cand_mcc >= fixed_mcc + student_mcc_gain
                                        )
                                    if use_pareto_guard:
                                        allowed = (
                                            allowed
                                            and cand_score >= fixed_score + pareto_min_gain
                                            and cand_ece <= fixed_ece + pareto_ece_tol
                                            and cand_brier <= fixed_brier + pareto_brier_tol
                                        )
                                    if use_dual_pareto:
                                        allowed = (
                                            allowed
                                            and cand_mcc >= fixed_mcc - dual_mcc_tol
                                            and cand_gmean >= fixed_gmean - dual_gmean_tol
                                            and cand_ece <= fixed_ece + pareto_ece_tol
                                            and cand_brier <= fixed_brier + pareto_brier_tol
                                        )

                                    best_metrics = best_choice["metrics"]
                                    best_score = float(best_choice["score"])
                                    best_name = str(best_choice["name"])
                                    best_family = "teacher" if best_name.startswith("teacher") else best_name
                                    best_ece = float(best_metrics.get("ECE", 1.0))
                                    best_brier = float(best_metrics.get("Brier", 1.0))
                                    replace = cand_score > best_score
                                    if cand_family == "teacher":
                                        replace = (
                                            cand_score > best_score + teacher_gain
                                            or (
                                                (not is_teacher_pool)
                                                and cand_score >= best_score - teacher_tie_tol
                                                and cand_ece <= best_ece
                                                and cand_brier <= best_brier + dynamic_brier_tol
                                            )
                                        )
                                        if is_teacher_pool:
                                            replace = (
                                                replace
                                                and cand_ece <= best_ece + teacher_pool_ece_tol
                                                and cand_brier <= best_brier + teacher_pool_brier_tol
                                            )
                                    elif cand_family == "rcf" and best_family == "teacher":
                                        replace = (
                                            cand_score > best_score
                                            or (
                                                cand_score >= best_score - dynamic_tie_tol
                                                and cand_ece <= best_ece
                                                and cand_brier <= best_brier + dynamic_brier_tol
                                            )
                                        )

                                    if allowed and replace:
                                        best_choice = choice

                            if best_choice is not None:
                                selected_candidate = str(best_choice["name"])
                                selected_calibration_mode = str(best_choice.get("calibration_mode", "post"))
                                calibrator = best_choice["calibrator"]
                                threshold = float(best_choice["threshold"])
                                val_metrics = dict(best_choice["metrics"])
                                val_score = float(best_choice["score"])
                                val_alpha_mean = float(best_choice["alpha_mean"])
                                val_alpha_std = float(best_choice["alpha_std"])
                            print(
                                f"  RCF验证：selected={selected_candidate}, alpha_mean={val_alpha_mean:.3f}, "
                                f"alpha_std={val_alpha_std:.3f}, cal={selected_calibration_mode}, "
                                f"score={val_score:.4f}, val_MCC={val_metrics.get('MCC', np.nan):.4f}, "
                                f"val_ECE={val_metrics.get('ECE', np.nan):.4f}, th={threshold:.3f}"
                            )
                        except Exception as e:
                            print(f"[WARN] {method} 验证阶段失败，使用固定 alpha 融合兜底。错误：{e}")
                            rcf_head = None

                        try:
                            if student_cache_key in rcf_student_test_cache:
                                student_test_prob = rcf_student_test_cache[student_cache_key].copy()
                            else:
                                runner_test = MethodRunner(method_cfg)
                                student_test_prob = runner_test.run_estd_kd_v2(
                                    X_train,
                                    y_train,
                                    X_test,
                                    ens_test_train_prob,
                                    variant=student_variant,
                                )
                                rcf_student_test_cache[student_cache_key] = np.asarray(student_test_prob).copy()
                            X_test_feat = self._rcf_features(
                                student_test_prob,
                                ens_test_pred_prob,
                                ens_test_pred_stack,
                            )
                            if selected_candidate == "student":
                                test_prob = sanitize_probabilities(student_test_prob)
                                test_alpha = np.zeros(len(student_test_prob), dtype=float)
                            elif selected_candidate.startswith("teacher_pool:") and ens_test_pred_stack is not None:
                                teacher_name = selected_candidate.split(":", 1)[1]
                                try:
                                    teacher_idx = ens_test_teacher_names.index(teacher_name)
                                except ValueError:
                                    teacher_idx = 0
                                test_prob = sanitize_probabilities(np.asarray(ens_test_pred_stack)[teacher_idx])
                                test_alpha = np.ones(len(student_test_prob), dtype=float)
                            elif selected_candidate == "teacher":
                                test_prob = sanitize_probabilities(ens_test_pred_prob)
                                test_alpha = np.ones(len(student_test_prob), dtype=float)
                            elif selected_candidate == "teacher_raw":
                                raw_teacher_prob = uncal_test_pred_prob if uncal_test_pred_prob is not None else ens_test_pred_prob
                                test_prob = sanitize_probabilities(raw_teacher_prob)
                                test_alpha = np.ones(len(student_test_prob), dtype=float)
                            elif selected_candidate == "fixed":
                                test_prob = self._blend_student_teacher(
                                    student_test_prob,
                                    ens_test_pred_prob,
                                    default_alpha,
                                )
                                test_alpha = np.full(len(student_test_prob), default_alpha, dtype=float)
                            else:
                                test_prob, test_alpha, _ = self._predict_rcf_blend(
                                    rcf_head, X_test_feat, student_test_prob, ens_test_pred_prob, cfg=method_cfg
                                )
                            if getattr(self.cfg, "use_post_calibration", True) and calibrator is not None:
                                test_prob = calibrator.transform(test_prob)

                            metrics = MetricUtils.summarize(y_test, test_prob, threshold=threshold)
                            record = {
                                "dataset": ds_name,
                                "fold": fold_id,
                                "method": method,
                                "gamma": float(method_cfg.gamma),
                                "reliability_tau": float(method_cfg.reliability_tau),
                                "student_type": str(method_cfg.student_type),
                                "mlp_hidden1": int(method_cfg.mlp_hidden1),
                                "mlp_hidden2": int(method_cfg.mlp_hidden2),
                                "rcf_min_alpha": float(method_cfg.rcf_min_alpha),
                                "rcf_max_alpha": float(method_cfg.rcf_max_alpha),
                                "ablation_variant": student_variant,
                                "rcf_selected_candidate": selected_candidate,
                                "rcf_calibration_mode": selected_calibration_mode,
                                "rcf_alpha_mean": float(np.mean(test_alpha)),
                                "rcf_alpha_std": float(np.std(test_alpha)),
                                "rcf_val_alpha_mean": val_alpha_mean,
                                "rcf_val_alpha_std": val_alpha_std,
                                "blend_val_score": val_score,
                                "blend_val_MCC": val_metrics.get("MCC", np.nan),
                                "blend_val_F1": val_metrics.get("F1", np.nan),
                                "blend_val_GMean": val_metrics.get("GMean", np.nan),
                                "blend_val_ECE": val_metrics.get("ECE", np.nan),
                                "blend_val_Brier": val_metrics.get("Brier", np.nan),
                                **metrics,
                            }
                            all_records.append(record)
                            self._append_pred_records(
                                all_pred_records,
                                ds_name,
                                fold_id,
                                method,
                                test_original_ids,
                                y_test,
                                test_prob,
                                threshold,
                            )

                            elapsed = time.time() - t0
                            self.all_time_records.append({
                                "dataset": ds_name,
                                "fold": fold_id,
                                "method": method,
                                "time_sec": float(elapsed),
                            })

                            print(
                                f"  {method}: MCC={metrics['MCC']:.4f}, F1={metrics['F1']:.4f}, "
                                f"GMean={metrics['GMean']:.4f}, ECE={metrics['ECE']:.4f}, "
                                f"Brier={metrics['Brier']:.4f}, selected={selected_candidate}, "
                                f"alpha={np.mean(test_alpha):.3f}±{np.std(test_alpha):.3f}, "
                                f"th={threshold:.3f}, time={elapsed:.1f}s"
                            )
                        except Exception as e:
                            print(f"[ERROR] {method} 测试阶段失败，已跳过。错误：{e}")
                        continue

                    # 阈值：先在训练折内部验证集上得到验证概率
                    try:
                        shared_full_key = (
                            round(float(self.cfg.gamma), 12),
                            round(float(self.cfg.reliability_tau), 12),
                            "full",
                        )
                        if method == "ESTD-KD-S" and shared_full_key in rcf_student_val_cache:
                            val_prob = rcf_student_val_cache[shared_full_key].copy()
                        else:
                            runner_val = MethodRunner(self.cfg)
                            val_prob = runner_val.predict_method_prepared(
                                method,
                                X_sub,
                                y_sub,
                                X_val,
                                ensemble_teacher_prob=ens_val_train_prob,
                                ensemble_pred_prob=ens_val_pred_prob,
                                single_teacher_prob=single_val_train_prob,
                                uncalibrated_teacher_prob=uncal_val_train_prob,
                                uncalibrated_pred_prob=uncal_val_pred_prob,
                                ensemble_teacher_stack_train=ens_val_train_stack,
                                ensemble_teacher_stack_pred=ens_val_pred_stack,
                            )
                            if method == "ESTD-KD-S":
                                rcf_student_val_cache[shared_full_key] = np.asarray(val_prob).copy()
                        calibrator = None
                        use_post = getattr(self.cfg, "use_post_calibration", True) and method != "TeacherEns"
                        if use_post:
                            calibrator = ProbabilityCalibrator().fit(y_val, val_prob)
                            val_prob_for_threshold = calibrator.transform(val_prob)
                        else:
                            val_prob_for_threshold = val_prob
                        if self.cfg.use_threshold_search:
                            threshold = MetricUtils.find_best_threshold(
                                y_val, val_prob_for_threshold, metric=self.cfg.threshold_metric
                            )
                        else:
                            threshold = 0.5
                    except Exception as e:
                        print(f"[WARN] {method} 验证阶段失败，使用 threshold=0.5。错误：{e}")
                        threshold = 0.5
                        calibrator = None

                    # 测试：在完整训练折上训练并预测测试折
                    try:
                        if method == "ESTD-KD-S" and shared_full_key in rcf_student_test_cache:
                            test_prob = rcf_student_test_cache[shared_full_key].copy()
                        else:
                            runner_test = MethodRunner(self.cfg)
                            test_prob = runner_test.predict_method_prepared(
                                method,
                                X_train,
                                y_train,
                                X_test,
                                ensemble_teacher_prob=ens_test_train_prob,
                                ensemble_pred_prob=ens_test_pred_prob,
                                single_teacher_prob=single_test_train_prob,
                                uncalibrated_teacher_prob=uncal_test_train_prob,
                                uncalibrated_pred_prob=uncal_test_pred_prob,
                                ensemble_teacher_stack_train=ens_test_train_stack,
                                ensemble_teacher_stack_pred=ens_test_pred_stack,
                            )
                            if method == "ESTD-KD-S":
                                rcf_student_test_cache[shared_full_key] = np.asarray(test_prob).copy()
                        use_post = getattr(self.cfg, "use_post_calibration", True) and method != "TeacherEns"
                        if use_post and calibrator is not None:
                            test_prob = calibrator.transform(test_prob)

                        metrics = MetricUtils.summarize(y_test, test_prob, threshold=threshold)
                        record = {
                            "dataset": ds_name,
                            "fold": fold_id,
                            "method": method,
                            **metrics,
                        }
                        all_records.append(record)
                        self._append_pred_records(
                            all_pred_records,
                            ds_name,
                            fold_id,
                            method,
                            test_original_ids,
                            y_test,
                            test_prob,
                            threshold,
                        )

                        elapsed = time.time() - t0
                        self.all_time_records.append({
                            "dataset": ds_name,
                            "fold": fold_id,
                            "method": method,
                            "time_sec": float(elapsed),
                        })

                        print(
                            f"  {method}: MCC={metrics['MCC']:.4f}, F1={metrics['F1']:.4f}, "
                            f"GMean={metrics['GMean']:.4f}, ECE={metrics['ECE']:.4f}, "
                            f"Brier={metrics['Brier']:.4f}, th={threshold:.3f}, time={elapsed:.1f}s"
                        )

                    except Exception as e:
                        print(f"[ERROR] {method} 测试阶段失败，已跳过。错误：{e}")
                        continue

                trace_fields = {
                    "repeat_id": int(repeat_id),
                    "fold_id": int(fold_in_repeat),
                    "seed": int(repeat_seed),
                }
                for record in all_records[fold_record_start:]:
                    record.update(trace_fields)
                for record in all_pred_records[fold_pred_start:]:
                    record.update(trace_fields)
                for record in self.all_teacher_records[fold_teacher_start:]:
                    record.update(trace_fields)
                for record in self.all_time_records[fold_time_start:]:
                    record.update(trace_fields)
                for record in self.all_teacher_state_records[fold_state_start:]:
                    record.update(trace_fields)

                pd.DataFrame(all_records).to_csv(
                    self.metrics_dir / "all_metrics.partial.csv", index=False, encoding="utf-8-sig"
                )
                pd.DataFrame(all_pred_records).to_csv(
                    self.preds_dir / "all_predictions.partial.csv", index=False, encoding="utf-8-sig"
                )
                if self.all_teacher_records:
                    pd.DataFrame(self.all_teacher_records).to_csv(
                        self.metrics_dir / "teacher_quality.partial.csv", index=False, encoding="utf-8-sig"
                    )
                if self.all_time_records:
                    pd.DataFrame(self.all_time_records).to_csv(
                        self.metrics_dir / "runtime_summary.partial.csv", index=False, encoding="utf-8-sig"
                    )
                if self.all_teacher_state_records:
                    pd.DataFrame(self.all_teacher_state_records).to_csv(
                        self.metrics_dir / "teacher_state_samples.partial.csv", index=False, encoding="utf-8-sig"
                    )

        self.cfg.random_state = base_random_state
        metric_df = pd.DataFrame(all_records)
        pred_df = pd.DataFrame(all_pred_records)

        metric_df.to_csv(self.metrics_dir / "all_metrics.csv", index=False, encoding="utf-8-sig")
        pred_df.to_csv(self.preds_dir / "all_predictions.csv", index=False, encoding="utf-8-sig")

        if self.all_teacher_records:
            pd.DataFrame(self.all_teacher_records).to_csv(
                self.metrics_dir / "teacher_quality.csv", index=False, encoding="utf-8-sig"
            )
        if self.all_time_records:
            pd.DataFrame(self.all_time_records).to_csv(
                self.metrics_dir / "runtime_summary.csv", index=False, encoding="utf-8-sig"
            )

        self._save_mean_std_table(metric_df)
        self._save_paper_split_tables(metric_df)
        self._save_best_counts(metric_df)
        self._save_revision_tables(metric_df)
        self._save_teacher_state_outputs()
        if getattr(self.cfg, "enable_statistical_tests", True):
            self._save_statistical_tests(metric_df)
        if getattr(self.cfg, "enable_diagnostic_outputs", True):
            self._save_uncertainty_and_reliability_outputs(pred_df, metric_df)


    @staticmethod
    def _subset_summary(metric_df: pd.DataFrame, methods: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        sub = metric_df[metric_df["method"].isin(methods)].copy()
        if sub.empty:
            return pd.DataFrame(), pd.DataFrame()
        metrics = [m for m in ["MCC", "F1", "GMean", "PRAUC", "AUC", "ECE", "Brier"] if m in sub.columns]
        by_dataset = sub.groupby(["dataset", "method"])[metrics].mean().reset_index()
        overall = by_dataset.groupby("method")[metrics].agg(["mean", "std"]).reset_index()
        overall.columns = [
            "_".join(str(part) for part in col if str(part)) if isinstance(col, tuple) else str(col)
            for col in overall.columns
        ]
        return by_dataset, overall

    def _save_revision_tables(self, metric_df: pd.DataFrame):
        if metric_df.empty:
            return
        suites = {
            "revision_stage_contribution": [
                "ESTD-Student", "ESTD-KD-S", "ESTD-KD-RCF (fixed fusion)", "ESTD-KD-RCF"
            ],
            "revision_ablation": [
                "ESTD-KD-RCF",
                "ESTD-KD-RCF (w/o adaptive temperature)",
                "ESTD-KD-RCF (w/o global reliability KD strength)",
                "ESTD-KD-RCF (w/o Brier)",
                "ESTD-KD-RCF (fixed fusion)",
                "ESTD-KD-RCF (w/o candidate pool)",
            ],
            "revision_direct_comparison": [
                "KD", "Yang-ATD-adapted", "UMKD-UDD-adapted", "ESTD-KD-S", "ESTD-KD-RCF"
            ],
            "revision_student_architecture": [
                "ESTD-KD-RCF [Linear]", "ESTD-KD-RCF [MLP-32]",
                "ESTD-KD-RCF [MLP-64-32]", "ESTD-KD-RCF [MLP-128-64]"
            ],
        }
        alpha_methods = sorted([m for m in metric_df["method"].astype(str).unique() if m.startswith("ESTD-KD-RCF [a")])
        if alpha_methods:
            suites["revision_alpha_bounds"] = alpha_methods
        for prefix, methods in suites.items():
            by_dataset, overall = self._subset_summary(metric_df, methods)
            if not by_dataset.empty:
                by_dataset.to_csv(self.metrics_dir / f"{prefix}_by_dataset.csv", index=False, encoding="utf-8-sig")
                overall.to_csv(self.metrics_dir / f"{prefix}_overall.csv", index=False, encoding="utf-8-sig")

    @staticmethod
    def _holm_adjust(p_values: List[float]) -> List[float]:
        if not p_values:
            return []
        p = np.asarray(p_values, dtype=float)
        order = np.argsort(p)
        m = len(p)
        adjusted = np.empty(m, dtype=float)
        running = 0.0
        for rank, idx in enumerate(order):
            val = min(1.0, (m - rank) * p[idx])
            running = max(running, val)
            adjusted[idx] = running
        return adjusted.tolist()

    def _save_statistical_tests(self, metric_df: pd.DataFrame):
        if metric_df.empty or not SCIPY_OK:
            if not SCIPY_OK:
                print("[WARN] scipy 不可用，跳过 Friedman/Wilcoxon-Holm 自动统计。")
            return
        metrics = [m for m in ["MCC", "F1", "GMean", "ECE", "Brier"] if m in metric_df.columns]
        # 正式统计只使用主对比算法，避免把消融/参数/结构敏感性方法混进 Friedman。
        stat_df = metric_df.copy()
        stat_df["paper_table_type"] = stat_df["method"].apply(self._paper_table_type)
        stat_df = stat_df[stat_df["paper_table_type"] == "main_comparison"].copy()
        if stat_df.empty:
            return
        ds_mean = stat_df.groupby(["dataset", "method"])[metrics].mean().reset_index()

        # 平均排名：分类指标越大越好，概率误差越小越好。
        rank_rows = []
        for metric in metrics:
            ascending = metric in {"ECE", "Brier"}
            pivot = ds_mean.pivot(index="dataset", columns="method", values=metric).dropna(axis=0, how="any")
            if pivot.empty:
                continue
            ranks = pivot.rank(axis=1, ascending=ascending, method="average")
            for method, value in ranks.mean(axis=0).items():
                rank_rows.append({"metric": metric, "method": method, "average_rank": float(value), "n_datasets": int(len(pivot))})
        if rank_rows:
            pd.DataFrame(rank_rows).to_csv(self.metrics_dir / "statistical_average_ranks.csv", index=False, encoding="utf-8-sig")

        friedman_rows = []
        wilcoxon_rows = []
        ours = "ESTD-KD-RCF"
        for metric in metrics:
            pivot = ds_mean.pivot(index="dataset", columns="method", values=metric).dropna(axis=0, how="any")
            if pivot.shape[0] >= 3 and pivot.shape[1] >= 3:
                try:
                    stat, pval = friedmanchisquare(*[pivot[c].values for c in pivot.columns])
                    friedman_rows.append({
                        "metric": metric, "statistic": float(stat), "p_value": float(pval),
                        "n_datasets": int(pivot.shape[0]), "n_methods": int(pivot.shape[1]),
                    })
                except Exception:
                    pass

            if ours not in pivot.columns or pivot.shape[0] < 3:
                continue
            raw_rows = []
            raw_p = []
            higher_better = metric not in {"ECE", "Brier"}
            for method in pivot.columns:
                if method == ours:
                    continue
                a = pivot[ours].values
                b = pivot[method].values
                diff = a - b if higher_better else b - a
                wins = int(np.sum(diff > 1e-12))
                ties = int(np.sum(np.abs(diff) <= 1e-12))
                losses = int(np.sum(diff < -1e-12))
                try:
                    if np.allclose(a, b):
                        stat, pval = 0.0, 1.0
                    else:
                        stat, pval = wilcoxon(a, b, alternative="two-sided", zero_method="wilcox")
                    stat, pval = float(stat), float(pval)
                except Exception:
                    stat, pval = float("nan"), 1.0
                raw_rows.append({
                    "metric": metric, "ours": ours, "comparison": method,
                    "wilcoxon_statistic": stat, "p_raw": pval,
                    "wins": wins, "ties": ties, "losses": losses,
                    "n_datasets": int(len(a)),
                })
                raw_p.append(pval)
            adjusted = self._holm_adjust(raw_p)
            for row, p_adj in zip(raw_rows, adjusted):
                row["p_holm"] = float(p_adj)
                row["significant_0p05"] = bool(p_adj < 0.05)
                wilcoxon_rows.append(row)

        if friedman_rows:
            pd.DataFrame(friedman_rows).to_csv(self.metrics_dir / "statistical_friedman.csv", index=False, encoding="utf-8-sig")
        if wilcoxon_rows:
            pd.DataFrame(wilcoxon_rows).to_csv(self.metrics_dir / "statistical_wilcoxon_holm_vs_ours.csv", index=False, encoding="utf-8-sig")

    def _save_uncertainty_and_reliability_outputs(self, pred_df: pd.DataFrame, metric_df: pd.DataFrame):
        """
        输出审稿人常关注的定性/可靠性分析数据：
        1) uncertainty_summary.csv：各方法置信度、熵、不确定性均值；
        2) reliability_bins.csv：可靠性曲线分箱数据；
        3) 可选输出可靠性曲线和不确定性箱线图，便于论文补图。
        """
        if pred_df is None or pred_df.empty:
            return
        df = pred_df.copy()
        df["correct"] = (df["y_pred"] == df["y_true"]).astype(int)
        summary = df.groupby(["dataset", "method"]).agg(
            mean_confidence=("confidence", "mean"),
            mean_entropy=("uncertainty_entropy", "mean"),
            mean_margin=("prob_margin", "mean"),
            accuracy=("correct", "mean"),
            n=("y_true", "count"),
        ).reset_index()
        summary.to_csv(self.metrics_dir / "uncertainty_summary.csv", index=False, encoding="utf-8-sig")

        bin_rows = []
        bins = np.linspace(0.0, 1.0, 11)
        for (ds, method), sub in df.groupby(["dataset", "method"]):
            probs = sanitize_probabilities(sub["y_prob"].values)
            y_true = sub["y_true"].values.astype(int)
            bin_ids = np.digitize(probs, bins) - 1
            bin_ids = np.clip(bin_ids, 0, 9)
            for b in range(10):
                mask = bin_ids == b
                if not np.any(mask):
                    continue
                bin_rows.append({
                    "dataset": ds,
                    "method": method,
                    "bin": b,
                    "bin_left": float(bins[b]),
                    "bin_right": float(bins[b + 1]),
                    "count": int(np.sum(mask)),
                    "avg_confidence": float(np.mean(probs[mask])),
                    "empirical_positive_rate": float(np.mean(y_true[mask])),
                    "abs_gap": float(abs(np.mean(probs[mask]) - np.mean(y_true[mask]))),
                })
        rel_df = pd.DataFrame(bin_rows)
        rel_df.to_csv(self.metrics_dir / "reliability_bins.csv", index=False, encoding="utf-8-sig")

        if bool(getattr(self.cfg, "make_xai_plots", False)):
            try:
                self._plot_reliability_from_bins(rel_df)
                self._plot_uncertainty_boxplot(df)
            except Exception as e:
                if getattr(self.cfg, "verbose", 1) >= 1:
                    print(f"[WARN] 定性/可解释性图生成失败：{str(e)[:120]}")

    def _plot_reliability_from_bins(self, rel_df: pd.DataFrame):
        if rel_df is None or rel_df.empty:
            return
        methods_to_plot = ["ESTD-KD-RCF", "UMKD-UDD-adapted", "CB-DistillGrad", "CalibKD", "BKD", "CBD"]
        sub = rel_df[rel_df["method"].isin(methods_to_plot)].copy()
        if sub.empty:
            return
        # 汇总所有数据集后的可靠性曲线，用于论文总体趋势图。
        agg = sub.groupby(["method", "bin"]).agg(
            avg_confidence=("avg_confidence", "mean"),
            empirical_positive_rate=("empirical_positive_rate", "mean"),
            count=("count", "sum"),
        ).reset_index()
        plt.figure(figsize=(7, 6))
        plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="Ideal")
        for method, g in agg.groupby("method"):
            g = g.sort_values("avg_confidence")
            plt.plot(g["avg_confidence"], g["empirical_positive_rate"], marker="o", label=method)
        plt.xlabel("Mean predicted probability")
        plt.ylabel("Empirical positive rate")
        plt.title("Reliability Diagram")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.figures_dir / "reliability_overall.png", dpi=300, bbox_inches="tight")
        plt.close()

    def _plot_uncertainty_boxplot(self, pred_df: pd.DataFrame):
        if pred_df is None or pred_df.empty:
            return
        methods_to_plot = ["ESTD-KD-RCF", "UMKD-UDD-adapted", "CB-DistillGrad", "CalibKD", "BKD", "CBD"]
        sub = pred_df[pred_df["method"].isin(methods_to_plot)].copy()
        if sub.empty:
            return
        data = [sub[sub["method"] == m]["uncertainty_entropy"].values for m in methods_to_plot if m in set(sub["method"])]
        labels = [m for m in methods_to_plot if m in set(sub["method"])]
        if not data:
            return
        plt.figure(figsize=(9, 5))
        try:
            plt.boxplot(data, tick_labels=labels, showfliers=False)
        except TypeError:
            plt.boxplot(data, labels=labels, showfliers=False)
        plt.ylabel("Prediction entropy")
        plt.title("Prediction Uncertainty Distribution")
        plt.xticks(rotation=30, ha="right")
        plt.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.figures_dir / "uncertainty_entropy_boxplot.png", dpi=300, bbox_inches="tight")
        plt.close()

    @staticmethod
    def _paper_table_type(method: str) -> str:
        main_methods = {
            "LR", "SVM", "RF", "XGBoost", "BalancedRF", "EasyEnsemble",
            "KD", "Yang-ATD-adapted", "DTKD", "CalibKD", "BKD", "CBD", "KRDistill", "BiKD",
            "CCKD", "ACCKD", "SRW-KD", "UMKD-UDD-adapted", "CB-DistillGrad",
            "ESTD-KD-RCF",
        }
        internal_prefixes = (
            "ESTD-KD (w/o", "ESTD-KD (coupled", "ESTD-KD (lambda_min",
            "ESTD-KD-RCF (w/o", "ESTD-KD-RCF (coupled", "ESTD-KD-RCF (lambda_min",
            "ESTD-KD-RCF [", "ESTD-KD+a"
        )
        if method in main_methods:
            return "main_comparison"
        if method.startswith(internal_prefixes) or method in {
            "Student", "ESTD-Student", "TeacherEns", "CalibratedTeacherEns", "ESTD-KD",
            "ESTD-KD-S", "ESTD-KD+", "ESTD-KD-Auto", "ESTD-KD-Sweep",
        }:
            return "ablation_or_internal"
        return "other"

    def _save_paper_split_tables(self, metric_df: pd.DataFrame):
        if metric_df.empty:
            return
        df = metric_df.copy()
        df["paper_table_type"] = df["method"].apply(self._paper_table_type)
        df.to_csv(self.metrics_dir / "all_metrics_with_table_type.csv", index=False, encoding="utf-8-sig")

        metrics = ["MCC", "F1", "GMean", "PRAUC", "AUC", "ECE", "Brier"]
        for table_type, filename in [
            ("main_comparison", "paper_main_comparison_mean_std.csv"),
            ("ablation_or_internal", "paper_ablation_internal_mean_std.csv"),
        ]:
            sub = df[df["paper_table_type"] == table_type].copy()
            if sub.empty:
                continue
            grouped = sub.groupby(["method"])[metrics].agg(["mean", "std"]).reset_index()
            grouped.columns = [
                "_".join(str(part) for part in col if str(part))
                if isinstance(col, tuple) else str(col)
                for col in grouped.columns
            ]
            rows = []
            for _, row in grouped.iterrows():
                out = {"method": row["method"]}
                for m in metrics:
                    mean_val = row[f"{m}_mean"]
                    std_val = row[f"{m}_std"]
                    if pd.isna(std_val):
                        std_val = 0.0
                    out[m] = f"{mean_val:.4f}±{std_val:.4f}"
                rows.append(out)
            pd.DataFrame(rows).to_csv(self.metrics_dir / filename, index=False, encoding="utf-8-sig")

    def _save_mean_std_table(self, metric_df: pd.DataFrame):
        if metric_df.empty:
            print("[WARN] 没有指标结果可汇总。")
            return
        metrics = ["MCC", "F1", "GMean", "PRAUC", "AUC", "ECE", "Brier", "threshold"]
        grouped = metric_df.groupby(["dataset", "method"])[metrics].agg(["mean", "std"]).reset_index()
        grouped.columns = [
            "_".join(str(part) for part in col if str(part))
            if isinstance(col, tuple) else str(col)
            for col in grouped.columns
        ]

        table_rows = []
        for _, row in grouped.iterrows():
            out = {
                "dataset": row["dataset"],
                "method": row["method"],
            }
            for m in metrics:
                mean_val = row[f"{m}_mean"]
                std_val = row[f"{m}_std"]
                if pd.isna(std_val):
                    std_val = 0.0
                out[m] = f"{mean_val:.4f}±{std_val:.4f}"
            table_rows.append(out)

        final_table = pd.DataFrame(table_rows)
        final_table.to_csv(self.metrics_dir / "final_results_mean_std.csv", index=False, encoding="utf-8-sig")
        print("\n[OK] 均值±标准差表格已保存：final_results_mean_std.csv")

    def _save_best_counts(self, metric_df: pd.DataFrame):
        if metric_df.empty:
            return
        rows = []
        high_metrics = ["MCC", "F1", "GMean", "PRAUC", "AUC"]
        low_metrics = ["ECE", "Brier"]
        mean_df = metric_df.groupby(["dataset", "method"])[high_metrics + low_metrics].mean().reset_index()

        for ds, sub in mean_df.groupby("dataset"):
            for m in high_metrics:
                idx = sub[m].idxmax()
                rows.append({"dataset": ds, "metric": m, "best_method": sub.loc[idx, "method"], "best_value": sub.loc[idx, m]})
            for m in low_metrics:
                idx = sub[m].idxmin()
                rows.append({"dataset": ds, "metric": m, "best_method": sub.loc[idx, "method"], "best_value": sub.loc[idx, m]})

        pd.DataFrame(rows).to_csv(self.metrics_dir / "best_method_by_metric.csv", index=False, encoding="utf-8-sig")


# ============================================================
# 九、命令行入口
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="ESTD-KD-RCF Scientific Reports FINAL experiment runner")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--datasets", type=str, default="all",
                        help="all 或逗号分隔数据集名，如 WDBC,BCWO,Pima,Heart")
    parser.add_argument("--methods", type=str, default="",
                        help="可选：逗号分隔方法过滤，如 LR,RF,XGBoost,ESTD-KD；默认运行全部启用方法")

    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--n_repeats", type=int, default=1)
    parser.add_argument("--formal", action="store_true", help="正式实验：自动设置 n_repeats=5")

    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--lambda_kd", type=float, default=0.7)
    parser.add_argument("--tmax", type=float, default=2.0)
    parser.add_argument("--mu_brier", type=float, default=0.05)
    parser.add_argument("--focal_gamma", type=float, default=0.5)
    parser.add_argument("--cb_beta", type=float, default=0.999)
    parser.add_argument("--lambda_min", type=float, default=0.05, help="最终可靠性映射下界；最终方法只用其计算训练折全局 lambda_bar")
    parser.add_argument("--lambda_max", type=float, default=0.50)
    parser.add_argument("--reliability_tau", type=float, default=0.50)
    parser.add_argument("--reliability_power", type=float, default=1.0)
    parser.add_argument("--label_smoothing_eps", type=float, default=0.05)
    parser.add_argument("--weighted_brier", action="store_true", help="对 Brier 校准项使用类别权重")
    parser.add_argument("--global_reliability_scale", action="store_true", help="用训练折教师整体可靠性缩放 KD 强度")
    parser.add_argument("--global_reliability_power", type=float, default=1.0)
    parser.add_argument("--use_margin_loss", action="store_true", help="加入轻量 margin loss 以稳定分类边界")
    parser.add_argument("--margin_weight", type=float, default=0.0)
    parser.add_argument("--margin_value", type=float, default=1.0)
    parser.add_argument("--hard_reliability_boost", action="store_true", help="教师不可靠样本上增强 hard-label BCE")
    parser.add_argument("--hard_boost_weight", type=float, default=0.0)
    parser.add_argument("--hard_boost_power", type=float, default=1.0)
    parser.add_argument("--use_pairwise_rank_loss", action="store_true", help="加入轻量 pairwise ranking loss 以提升排序/判别边界")
    parser.add_argument("--rank_loss_weight", type=float, default=0.0)
    parser.add_argument("--rank_loss_margin", type=float, default=1.0)

    parser.add_argument("--student_epochs", type=int, default=200)
    parser.add_argument("--student_lr", type=float, default=1e-2)
    parser.add_argument("--grad_clip_norm", type=float, default=5.0)
    parser.add_argument("--student_type", type=str, default="mlp", choices=["linear", "mlp"])
    parser.add_argument("--mlp_hidden1", type=int, default=64)
    parser.add_argument("--mlp_hidden2", type=int, default=32)
    parser.add_argument("--mlp_dropout", type=float, default=0.10)

    parser.add_argument("--no_xgb", action="store_true")
    parser.add_argument("--with_ablation", action="store_true", help="启用消融实验；默认只跑主实验以节省时间")
    parser.add_argument("--no_ablation", action="store_true", help="兼容旧参数：关闭消融实验")
    parser.add_argument("--no_oof", action="store_true")
    parser.add_argument("--no_teacher_calibration", action="store_true")
    parser.add_argument("--no_threshold_search", action="store_true")
    parser.add_argument("--threshold_metric", type=str, default="MCC", choices=["MCC", "F1", "GMean", "Balanced"], help="验证集阈值搜索目标")
    parser.add_argument("--disable_svm", action="store_true", help="为了快速测试可关闭 SVM 基线")
    parser.add_argument("--with_balanced_rf", action="store_true", help="启用 BalancedRandomForest 不平衡集成基线")
    parser.add_argument("--with_easy_ensemble", action="store_true", help="启用 EasyEnsemble 不平衡集成基线")
    parser.add_argument("--with_teacher_svm", action="store_true", help="启用 SVM 教师；默认关闭以提升速度")
    parser.add_argument("--with_public_kd_methods", action="store_true", help="启用公开论文适配版对比：DTKD、BKD、CBD、KRDistill、BiKD、CCKD、SRW-KD、UMKD、CB-DistillGrad")
    parser.add_argument("--bkd_weight_beta", type=float, default=0.999, help="BKD/CBD/BiKD/CCKD 有效样本数类别权重 beta")
    parser.add_argument("--cbd_resample_ratio", type=float, default=1.0, help="CBD-style 类别均衡重采样比例")
    parser.add_argument("--bikd_lambda_min", type=float, default=0.0, help="BiKD-style 样本级 KD 权重下界")
    parser.add_argument("--bikd_lambda_max", type=float, default=0.6, help="BiKD-style 样本级 KD 权重上界")
    parser.add_argument("--bikd_student_hardness_power", type=float, default=1.0, help="BiKD-style 学生困难度门控幂次")
    parser.add_argument("--bikd_teacher_reliability_power", type=float, default=1.0, help="BiKD-style 教师可靠性门控幂次")
    parser.add_argument("--cckd_prior_power", type=float, default=1.0, help="CCKD/ACCKD-style 类别先验校正强度")
    parser.add_argument("--srw_k_neighbors", type=int, default=10, help="SRW-KD-style 同类邻域样本数")
    parser.add_argument("--srw_weight_strength", type=float, default=1.0, help="SRW-KD-style 样本级重加权强度")
    parser.add_argument("--umkd_uncertainty_power", type=float, default=3.0, help="UMKD-style 专家熵加权强度")
    parser.add_argument("--cbdg_conflict_floor", type=float, default=0.25, help="CB-DistillGrad-style 教师冲突样本 KD 权重下界")
    parser.add_argument("--no_diagnostic_outputs", action="store_true", help="关闭 uncertainty/reliability 诊断 CSV 输出")
    parser.add_argument("--make_xai_plots", action="store_true", help="根据预测结果生成可靠性曲线和不确定性箱线图")
    parser.add_argument("--teacher_weight_power", type=float, default=1.0, help="教师质量权重幂次；>1 会更强调 OOF/Brier 更好的教师")
    parser.add_argument("--teacher_weight_metric", type=str, default="brier", choices=["brier", "balanced", "mcc"], help="教师集成权重评分：brier 只看校准；balanced 同时看 OOF MCC/F1/GMean/ECE/Brier")
    parser.add_argument("--samplewise_teacher_weighting", action="store_true", help="enable sample-wise teacher reliability weighting")
    parser.add_argument("--samplewise_teacher_certainty_power", type=float, default=1.0, help="power for per-sample teacher certainty weights")
    parser.add_argument("--disable_student", action="store_true")
    parser.add_argument("--with_estd_blend", action="store_true", help="启用 ESTD-KD+：学生概率与教师集成概率融合")
    parser.add_argument("--blend_teacher_weight", type=float, default=0.25)
    parser.add_argument("--with_estd_autoblend", action="store_true", help="启用 ESTD-KD-Auto：在验证集上自动选择学生/教师融合比例")
    parser.add_argument("--with_estd_alpha_sweep", action="store_true", help="启用 alpha sweep：一次训练评估多个固定融合比例")
    parser.add_argument("--blend_alpha_grid", type=str, default="0.0,0.25,0.5,0.75,1.0")
    parser.add_argument(
        "--blend_selection_metric",
        type=str,
        default="balanced",
        choices=["balanced", "mcc", "f1", "gmean", "calibrated"],
        help="AutoBlend 验证集选择目标；balanced 同时考虑分类与校准",
    )
    parser.add_argument("--blend_space", type=str, default="prob", choices=["prob", "logit"], help="学生/教师概率融合空间")
    parser.add_argument("--with_rcf_methods", action="store_true", help="启用 TeacherEns、CalibratedTeacherEns、ESTD-KD-S、ESTD-KD-RCF 对照")
    parser.add_argument("--with_gamma_tau_grid", action="store_true", help="共享每折教师概率运行 gamma×tau 参数复核网格")
    parser.add_argument("--with_rcf_ablation_suite", action="store_true", help="最终消融：自适应温度、全局可靠性KD强度、Brier、融合、候选池")
    parser.add_argument("--with_stage_contribution", action="store_true", help="阶段贡献：ESTD-Student -> ESTD-KD-S -> Fixed fusion -> Full RCF")
    parser.add_argument("--with_student_arch_sensitivity", action="store_true", help="学生结构敏感性：Linear/MLP-32/MLP-64-32/MLP-128-64")
    parser.add_argument("--with_alpha_bounds_sensitivity", action="store_true", help="RCF alpha_min/alpha_max 边界敏感性网格")
    parser.add_argument("--alpha_min_grid", type=str, default="0.10,0.30,0.50")
    parser.add_argument("--alpha_max_grid", type=str, default="0.70,0.90,0.95")
    parser.add_argument("--teacher_state_quantile", type=float, default=0.25, help="高/低熵状态分组分位数，默认上下四分位")
    parser.add_argument("--no_teacher_state_analysis", action="store_true", help="关闭 OOF 教师高熵正确/低熵错误样本分析")
    parser.add_argument("--no_statistical_tests", action="store_true", help="关闭 Friedman/Wilcoxon-Holm 自动统计")
    parser.add_argument("--rcf_min_alpha", type=float, default=0.50, help="RCF 样本级教师融合权重下界")
    parser.add_argument("--rcf_max_alpha", type=float, default=0.95, help="RCF 样本级教师融合权重上界")
    parser.add_argument("--rcf_C", type=float, default=0.25, help="RCF LogisticRegression 融合头正则强度")
    parser.add_argument("--rcf_min_val_samples", type=int, default=60, help="验证集少于该样本数时跳过 RCF 融合头并退回固定融合")
    parser.add_argument("--no_rcf_teacher_candidate", action="store_true", help="RCF 候选中不加入 teacher-only 兜底")
    parser.add_argument("--no_rcf_teacher_pool_candidate", action="store_true", help="RCF 候选中不加入单教师池可靠锚点")
    parser.add_argument("--rcf_teacher_guard_min_class", type=int, default=40, help="少数类样本数不超过该值时禁用纯教师/单教师锚点；0 表示关闭")
    parser.add_argument("--with_rcf_raw_teacher_candidate", action="store_true", help="RCF 候选中加入未校准 teacher-only 兜底（默认关闭，避免退化为教师集成）")
    parser.add_argument("--no_rcf_raw_teacher_candidate", action="store_true", help="兼容旧参数：关闭未校准 teacher-only 兜底")
    parser.add_argument("--no_rcf_adaptive_post_calibration", action="store_true", help="RCF 候选固定使用后校准，不在 raw/post 之间自适应选择")
    parser.add_argument("--rcf_enable_student_guard", action="store_true", help="RCF 在 fixed 验证性能较差时允许选择 student-only 兜底")
    parser.add_argument("--rcf_candidate_min_gain", type=float, default=0.0, help="非 teacher 的 RCF 候选替代 fixed 所需最小综合分提升")
    parser.add_argument("--rcf_teacher_min_gain", type=float, default=0.0, help="teacher 候选替代 fixed 所需最小综合分提升")
    parser.add_argument("--rcf_teacher_tie_tol", type=float, default=0.005, help="teacher 候选与当前最佳分数接近且校准更好时允许替代的容忍范围")
    parser.add_argument("--rcf_teacher_pool_ece_tol", type=float, default=0.04, help="单教师池锚点相对 fixed 允许的验证 ECE 最大退化")
    parser.add_argument("--rcf_teacher_pool_brier_tol", type=float, default=0.02, help="单教师池锚点相对 fixed 允许的验证 Brier 最大退化")
    parser.add_argument("--rcf_dynamic_ece_tol", type=float, default=0.03, help="动态 RCF 相对 fixed 允许的验证 ECE 最大劣化")
    parser.add_argument("--rcf_dynamic_brier_tol", type=float, default=0.015, help="动态 RCF 相对 fixed 允许的验证 Brier 最大劣化")
    parser.add_argument("--rcf_dynamic_tie_tol", type=float, default=0.005, help="teacher 与动态 RCF 分数接近时偏向校准更好的动态 RCF 的容忍范围")
    parser.add_argument("--rcf_student_min_gain", type=float, default=0.05, help="student 兜底相对 fixed 所需最小综合分提升")
    parser.add_argument("--rcf_student_mcc_gain", type=float, default=0.05, help="student 兜底相对 fixed 所需最小验证 MCC 提升")
    parser.add_argument("--rcf_student_max_fixed_mcc", type=float, default=0.55, help="fixed 验证 MCC 低于该值时才允许 student 兜底")
    parser.add_argument("--rcf_pareto_guard", action="store_true", help="enable Pareto guard for RCF candidate selection")
    parser.add_argument("--rcf_pareto_min_gain", type=float, default=0.005, help="minimum validation score gain under Pareto guard")
    parser.add_argument("--rcf_pareto_ece_tol", type=float, default=0.025, help="maximum ECE degradation under Pareto guard")
    parser.add_argument("--rcf_pareto_brier_tol", type=float, default=0.012, help="maximum Brier degradation under Pareto guard")
    parser.add_argument("--rcf_calibrator_pool", action="store_true", help="enable raw/Platt/temperature/isotonic calibrator selection for RCF")
    parser.add_argument("--rcf_isotonic_min_class", type=int, default=20, help="minimum validation minority class size for isotonic calibration")
    parser.add_argument("--rcf_threshold_metric_pool", type=str, default="", help="comma-separated threshold metrics for RCF validation, e.g. MCC,F1,GMean,Balanced")
    parser.add_argument("--rcf_dual_pareto_guard", action="store_true", help="guard RCF candidates by MCC/GMean and calibration simultaneously")
    parser.add_argument("--rcf_dual_mcc_tol", type=float, default=0.005, help="allowed MCC drop under dual Pareto guard")
    parser.add_argument("--rcf_dual_gmean_tol", type=float, default=0.010, help="allowed GMean drop under dual Pareto guard")
    parser.add_argument("--rcf_safe_calibrator_pool", action="store_true", help="enable safe raw/Platt/temperature calibrator pool without isotonic")
    parser.add_argument("--rcf_safe_cal_ece_tol", type=float, default=0.002, help="max validation ECE increase allowed for safe calibrator variants")
    parser.add_argument("--rcf_safe_cal_brier_tol", type=float, default=0.001, help="max validation Brier increase allowed for safe calibrator variants")
    parser.add_argument("--teacher_disagreement_gate", action="store_true", help="reduce KD weight when calibrated teachers strongly disagree")
    parser.add_argument("--teacher_entropy_gate_power", type=float, default=1.0, help="entropy gate power for teacher disagreement KD weighting")
    parser.add_argument("--teacher_disagreement_gate_power", type=float, default=1.0, help="disagreement gate power for teacher disagreement KD weighting")
    parser.add_argument("--use_focal", action="store_true", help="启用 Focal BCE；默认关闭，避免损害概率校准")
    parser.add_argument("--no_reliability_lambda", action="store_true", help="关闭教师可靠性门控 lambda，退回旧版熵驱动 lambda")
    parser.add_argument("--legacy_reliable_soft_label", action="store_true", help="legacy diagnostic only; final ESTD-KD-RCF always disables reliable soft targets")
    parser.add_argument("--no_reliable_soft_label", action="store_true", help="兼容旧命令；SCIENTIFIC-REPORTS-FINAL默认已关闭可靠性修正soft target")
    parser.add_argument("--coupled_cb_kd_weight", action="store_true", help="legacy compatibility flag; final method already shares class-balanced weights")
    parser.add_argument("--decoupled_cb_kd_weight", action="store_true", help="legacy diagnostic only; final paper runs keep coupled class-balanced weighting")
    parser.add_argument("--no_post_calibration", action="store_true", help="关闭验证集 Platt 后校准")
    parser.add_argument("--torch_num_threads", type=int, default=1, help="PyTorch CPU threads for the small tabular student")
    parser.add_argument("--resume", action="store_true", help="resume from complete folds in *.partial.csv checkpoints")
    parser.add_argument("--verbose", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()

    torch_threads = max(1, int(args.torch_num_threads))
    torch.set_num_threads(torch_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    n_repeats = 5 if args.formal else args.n_repeats

    cfg = ExperimentConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        datasets=args.datasets,
        methods_filter=args.methods,
        torch_num_threads=torch_threads,
        resume_from_partial=args.resume,
        n_splits=args.n_splits,
        n_repeats=n_repeats,
        gamma=args.gamma,
        lambda_kd=args.lambda_kd,
        t_max=args.tmax,
        mu_brier=args.mu_brier,
        focal_gamma=args.focal_gamma,
        cb_beta=args.cb_beta,
        lambda_min=args.lambda_min,
        lambda_max=args.lambda_max,
        reliability_tau=args.reliability_tau,
        reliability_power=args.reliability_power,
        label_smoothing_eps=args.label_smoothing_eps,
        use_weighted_brier=args.weighted_brier,
        use_global_reliability_scale=args.global_reliability_scale,
        global_reliability_power=args.global_reliability_power,
        use_margin_loss=args.use_margin_loss,
        margin_weight=args.margin_weight,
        margin_value=args.margin_value,
        use_reliability_hard_boost=args.hard_reliability_boost,
        hard_boost_weight=args.hard_boost_weight,
        hard_boost_power=args.hard_boost_power,
        use_pairwise_rank_loss=args.use_pairwise_rank_loss,
        rank_loss_weight=args.rank_loss_weight,
        rank_loss_margin=args.rank_loss_margin,
        student_epochs=args.student_epochs,
        student_lr=args.student_lr,
        grad_clip_norm=args.grad_clip_norm,
        student_type=args.student_type,
        mlp_hidden1=args.mlp_hidden1,
        mlp_hidden2=args.mlp_hidden2,
        mlp_dropout=args.mlp_dropout,
        use_focal=args.use_focal,
        use_reliability_lambda=(not args.no_reliability_lambda),
        use_reliable_soft_label=False,
        estd_decouple_kd_class_weight=False,
        use_post_calibration=(not args.no_post_calibration),
        enable_xgb=(not args.no_xgb),
        enable_balanced_rf=args.with_balanced_rf,
        enable_easy_ensemble=args.with_easy_ensemble,
        enable_public_kd_methods=args.with_public_kd_methods,
        bkd_weight_beta=args.bkd_weight_beta,
        cbd_resample_ratio=args.cbd_resample_ratio,
        bikd_lambda_min=args.bikd_lambda_min,
        bikd_lambda_max=args.bikd_lambda_max,
        bikd_student_hardness_power=args.bikd_student_hardness_power,
        bikd_teacher_reliability_power=args.bikd_teacher_reliability_power,
        cckd_prior_power=args.cckd_prior_power,
        srw_k_neighbors=args.srw_k_neighbors,
        srw_weight_strength=args.srw_weight_strength,
        umkd_uncertainty_power=args.umkd_uncertainty_power,
        cbdg_conflict_floor=args.cbdg_conflict_floor,
        enable_diagnostic_outputs=(not args.no_diagnostic_outputs),
        make_xai_plots=args.make_xai_plots,
        enable_ablation=(args.with_ablation and not args.no_ablation),
        use_teacher_oof=(not args.no_oof),
        use_teacher_calibration=(not args.no_teacher_calibration),
        use_threshold_search=(not args.no_threshold_search),
        threshold_metric=args.threshold_metric,
        enable_svm=(not args.disable_svm),
        enable_student=(not args.disable_student),
        enable_ours_blend=args.with_estd_blend,
        blend_teacher_weight=args.blend_teacher_weight,
        enable_ours_auto_blend=args.with_estd_autoblend,
        enable_ours_alpha_sweep=args.with_estd_alpha_sweep,
        blend_alpha_grid=args.blend_alpha_grid,
        blend_selection_metric=args.blend_selection_metric,
        blend_space=args.blend_space,
        enable_rcf_methods=args.with_rcf_methods,
        enable_gamma_tau_grid=args.with_gamma_tau_grid,
        enable_rcf_ablation_suite=args.with_rcf_ablation_suite,
        enable_stage_contribution_suite=args.with_stage_contribution,
        enable_student_arch_sensitivity=args.with_student_arch_sensitivity,
        enable_alpha_bounds_sensitivity=args.with_alpha_bounds_sensitivity,
        enable_teacher_state_analysis=(not args.no_teacher_state_analysis),
        enable_statistical_tests=(not args.no_statistical_tests),
        teacher_state_quantile=args.teacher_state_quantile,
        alpha_min_grid=args.alpha_min_grid,
        alpha_max_grid=args.alpha_max_grid,
        rcf_min_alpha=args.rcf_min_alpha,
        rcf_max_alpha=args.rcf_max_alpha,
        rcf_C=args.rcf_C,
        rcf_min_val_samples=args.rcf_min_val_samples,
        rcf_enable_teacher_candidate=(not args.no_rcf_teacher_candidate),
        rcf_enable_teacher_pool_candidate=(not args.no_rcf_teacher_pool_candidate),
        rcf_teacher_guard_min_class=args.rcf_teacher_guard_min_class,
        rcf_include_raw_teacher_candidate=(
            bool(args.with_rcf_raw_teacher_candidate) and not bool(args.no_rcf_raw_teacher_candidate)
        ),
        rcf_adaptive_post_calibration=(not args.no_rcf_adaptive_post_calibration),
        rcf_enable_student_guard=args.rcf_enable_student_guard,
        rcf_candidate_min_gain=args.rcf_candidate_min_gain,
        rcf_teacher_min_gain=args.rcf_teacher_min_gain,
        rcf_teacher_tie_tol=args.rcf_teacher_tie_tol,
        rcf_teacher_pool_ece_tol=args.rcf_teacher_pool_ece_tol,
        rcf_teacher_pool_brier_tol=args.rcf_teacher_pool_brier_tol,
        rcf_dynamic_ece_tol=args.rcf_dynamic_ece_tol,
        rcf_dynamic_brier_tol=args.rcf_dynamic_brier_tol,
        rcf_dynamic_tie_tol=args.rcf_dynamic_tie_tol,
        rcf_student_min_gain=args.rcf_student_min_gain,
        rcf_student_mcc_gain=args.rcf_student_mcc_gain,
        rcf_student_max_fixed_mcc=args.rcf_student_max_fixed_mcc,
        enable_teacher_svm=args.with_teacher_svm,
        teacher_weight_power=args.teacher_weight_power,
        teacher_weight_metric=args.teacher_weight_metric,
        use_samplewise_teacher_weighting=args.samplewise_teacher_weighting,
        samplewise_teacher_certainty_power=args.samplewise_teacher_certainty_power,
        rcf_pareto_guard=args.rcf_pareto_guard,
        rcf_pareto_min_gain=args.rcf_pareto_min_gain,
        rcf_pareto_ece_tol=args.rcf_pareto_ece_tol,
        rcf_pareto_brier_tol=args.rcf_pareto_brier_tol,
        rcf_enable_calibrator_pool=args.rcf_calibrator_pool,
        rcf_isotonic_min_class=args.rcf_isotonic_min_class,
        rcf_threshold_metric_pool=args.rcf_threshold_metric_pool,
        rcf_dual_pareto_guard=args.rcf_dual_pareto_guard,
        rcf_dual_mcc_tol=args.rcf_dual_mcc_tol,
        rcf_dual_gmean_tol=args.rcf_dual_gmean_tol,
        rcf_safe_calibrator_pool=args.rcf_safe_calibrator_pool,
        rcf_safe_cal_ece_tol=args.rcf_safe_cal_ece_tol,
        rcf_safe_cal_brier_tol=args.rcf_safe_cal_brier_tol,
        teacher_disagreement_gate=args.teacher_disagreement_gate,
        teacher_entropy_gate_power=args.teacher_entropy_gate_power,
        teacher_disagreement_gate_power=args.teacher_disagreement_gate_power,
        verbose=args.verbose,
    )

    print("\n========== ESTD-KD-RCF v7 定向修复实验开始 ==========")
    print(json.dumps(asdict(cfg), indent=2, ensure_ascii=False))

    loader = DatasetLoader(cfg.data_dir)
    datasets = loader.load_selected(cfg.datasets)

    runner = ExperimentRunner(cfg)
    runner.run(datasets)

    print("\n[OK] 实验完成！")
    print(f"结果目录：{cfg.output_dir}")
    print(f"主结果表：{Path(cfg.output_dir) / 'metrics' / 'final_results_mean_std.csv'}")
    print(f"逐折结果：{Path(cfg.output_dir) / 'metrics' / 'all_metrics.csv'}")
    print(f"教师质量：{Path(cfg.output_dir) / 'metrics' / 'teacher_quality.csv'}")


if __name__ == "__main__":
    main()
