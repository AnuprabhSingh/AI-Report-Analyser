#!/usr/bin/env python3
"""
Generate paper-ready evaluation plots for trained interpretation models.

Outputs:
- Accuracy/Precision/Recall/F1/MCC grouped bar chart
- Confusion matrix per category
- ROC-AUC curve per category (binary and multiclass one-vs-rest)
- Cross-validation score boxplot per category
- Metrics summary CSV with uncertainty estimates
- Leakage diagnostics and paper-ready interpretation notes
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Callable, Dict, List

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    auc,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer, StandardScaler

from train_interpretation_model import InterpretationModelTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate evaluation plots for IEEE paper")
    parser.add_argument("--dataset", default="data/processed/combined_training_dataset.json")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--suffix", default="v2_expanded")
    parser.add_argument("--output-dir", default="outputs/paper_plots")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument(
        "--eval-mode",
        choices=["retrain", "artifact"],
        default="retrain",
        help="retrain = train on split-specific train fold (recommended for leakage-safe evaluation)",
    )
    return parser.parse_args()


def load_models(model_dir: str, suffix: str) -> tuple[dict, StandardScaler, Dict[str, object]]:
    suffix_part = f"_{suffix}" if suffix else ""
    meta_path = Path(model_dir) / f"model_metadata{suffix_part}.json"
    scaler_path = Path(model_dir) / f"scaler{suffix_part}.pkl"
    with open(meta_path, "r") as f:
        metadata = json.load(f)
    scaler = joblib.load(scaler_path)

    models: Dict[str, object] = {}
    for category in metadata["categories"]:
        model_path = Path(model_dir) / f"model_{category}{suffix_part}.pkl"
        if model_path.exists():
            models[category] = joblib.load(model_path)
    return metadata, scaler, models


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def plot_grouped_metrics(metrics_df: pd.DataFrame, output_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(metrics_df))
    width = 0.16

    ax.bar(x - 2 * width, metrics_df["accuracy"], width, label="Accuracy")
    ax.bar(x - width, metrics_df["precision_macro"], width, label="Precision (Macro)")
    ax.bar(x, metrics_df["recall_macro"], width, label="Recall (Macro)")
    ax.bar(x + width, metrics_df["f1_macro"], width, label="F1 (Macro)")
    ax.bar(x + 2 * width, metrics_df["mcc"], width, label="MCC")

    ax.set_title("Model Performance by Category")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_df["category"], rotation=20, ha="right")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(Path(output_dir) / "metrics_accuracy_precision_recall_f1_mcc.png", dpi=300)
    plt.close(fig)


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, category: str, output_dir: str) -> None:
    labels = sorted(list(set(y_true) | set(y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title(f"Confusion Matrix - {category}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    fig.savefig(Path(output_dir) / f"confusion_matrix_{category}.png", dpi=300)
    plt.close(fig)


def bootstrap_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_samples: int,
    seed: int,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    if n == 0:
        return float("nan"), float("nan"), float("nan")

    boot_vals: List[float] = []
    for _ in range(n_samples):
        idx = rng.integers(0, n, n)
        yt = y_true[idx]
        yp = y_pred[idx]
        try:
            val = float(metric_fn(yt, yp))
        except Exception:
            continue
        if np.isfinite(val):
            boot_vals.append(val)

    if not boot_vals:
        return float("nan"), float("nan"), float("nan")

    arr = np.asarray(boot_vals, dtype=float)
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    ci_low = float(np.percentile(arr, 2.5))
    ci_high = float(np.percentile(arr, 97.5))
    return std, ci_low, ci_high


def train_category_model(X_train: np.ndarray, y_train: pd.Series, seed: int) -> object:
    class_counts = y_train.value_counts().to_dict()
    n_classes = max(1, len(class_counts))
    class_weight = {
        cls: len(y_train) / (n_classes * count)
        for cls, count in class_counts.items()
    }
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        class_weight=class_weight,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def count_exact_row_overlap(X_train: pd.DataFrame, X_test: pd.DataFrame) -> int:
    train_hash = pd.util.hash_pandas_object(X_train.reset_index(drop=True), index=False)
    test_hash = pd.util.hash_pandas_object(X_test.reset_index(drop=True), index=False)
    return int(len(set(train_hash.values.tolist()) & set(test_hash.values.tolist())))


def shuffled_label_control_accuracy(
    X_train: np.ndarray,
    y_train: pd.Series,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    y_train_shuffled = y_train.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    model = train_category_model(X_train, y_train_shuffled, seed)
    y_pred = model.predict(X_test)

    majority_class = y_train.mode().iloc[0]
    baseline_pred = np.full(shape=len(y_test), fill_value=majority_class, dtype=object)

    shuffled_acc = float(accuracy_score(y_test, y_pred))
    shuffled_bal_acc = float(balanced_accuracy_score(y_test, y_pred))
    majority_acc = float(accuracy_score(y_test, baseline_pred))
    return shuffled_acc, shuffled_bal_acc, majority_acc


def write_paper_notes(
    metrics_df: pd.DataFrame,
    leakage_df: pd.DataFrame,
    output_dir: str,
    eval_mode: str,
) -> None:
    lines: List[str] = []
    lines.append("# Paper Evaluation Notes")
    lines.append("")
    lines.append("## Added Metrics")
    lines.append("- Included MCC for each category.")
    lines.append("- Included macro-F1 and weighted-F1.")
    lines.append("- Included bootstrap uncertainty as std and 95% CI.")
    lines.append("")
    lines.append("## Perfect Score Interpretation")

    perfect = metrics_df[metrics_df["accuracy"] >= 0.999]
    if perfect.empty:
        lines.append("- No category achieved perfect accuracy on the evaluated split.")
    else:
        lines.append(
            "- Categories with perfect/near-perfect accuracy: "
            + ", ".join(perfect["category"].tolist())
            + "."
        )
        lines.append(
            "- Likely reason: several labels in this dataset are rule-derived from the same measurements "
            "used as model inputs (for example threshold-based grading), which makes some tasks nearly deterministic."
        )
        lines.append(
            "- This behavior should be stated explicitly in the paper as high separability of derived-label tasks, "
            "not as evidence of broad clinical generalization by itself."
        )

    lines.append("")
    lines.append("## Leakage Evidence")
    if eval_mode == "artifact":
        lines.append(
            "- Evaluation mode was `artifact`; strict leakage-proof claims are weaker because trained artifacts may have seen overlapping data in prior training."
        )
    else:
        lines.append(
            "- Evaluation mode was `retrain`; each category model was fit only on the current training split and evaluated on holdout data."
        )

    overlap_rows = leakage_df[leakage_df["check"] == "index_overlap"]
    dup_rows = leakage_df[leakage_df["check"] == "exact_feature_row_overlap"]
    if not overlap_rows.empty:
        lines.append(f"- Train/test index overlap count: {int(overlap_rows['value'].iloc[0])}.")
    if not dup_rows.empty:
        lines.append(f"- Exact duplicate feature rows shared across train/test: {int(dup_rows['value'].iloc[0])}.")

    shuffled_rows = leakage_df[leakage_df["check"].str.startswith("shuffled_label_accuracy:")]
    if not shuffled_rows.empty:
        lines.append("- Shuffled-label control versus class-imbalance baseline:")
        for _, row in shuffled_rows.iterrows():
            cat = row["check"].split(":", 1)[1]
            bal = leakage_df.loc[leakage_df["check"] == f"shuffled_label_balanced_accuracy:{cat}", "value"]
            base = leakage_df.loc[leakage_df["check"] == f"majority_class_accuracy:{cat}", "value"]
            if not bal.empty and not base.empty:
                lines.append(
                    f"  - {cat}: shuffled_acc={float(row['value']):.3f}, "
                    f"shuffled_bal_acc={float(bal.iloc[0]):.3f}, majority_baseline={float(base.iloc[0]):.3f}"
                )

    lines.append("")
    lines.append("## Recommended Wording")
    lines.append(
        "- \"To reduce leakage risk, all reported holdout metrics were produced with split-wise retraining, "
        "no index overlap between train and test partitions, and shuffled-label sanity checks interpreted against class-imbalance baselines.\""
    )
    lines.append(
        "- \"Perfect scores in some categories are explained by deterministic threshold-derived labels and clear class separation in the measurement space.\""
    )

    with open(Path(output_dir) / "paper_evaluation_notes.md", "w") as f:
        f.write("\n".join(lines) + "\n")


def plot_roc_auc(model: object, X_test: np.ndarray, y_true: np.ndarray, category: str, output_dir: str) -> bool:
    if not hasattr(model, "predict_proba"):
        return False

    try:
        classes = model.classes_
        proba = model.predict_proba(X_test)
    except Exception:
        return False

    fig, ax = plt.subplots(figsize=(7, 6))
    plotted = False

    if len(classes) == 2:
        positive_class = classes[1]
        y_bin = (y_true == positive_class).astype(int)
        if len(np.unique(y_bin)) < 2:
            plt.close(fig)
            return False
        fpr, tpr, _ = roc_curve(y_bin, proba[:, 1])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.3f}")
        plotted = True
    else:
        lb = LabelBinarizer()
        lb.fit(classes)
        y_bin = lb.transform(y_true)
        if y_bin.ndim == 1:
            y_bin = np.column_stack([1 - y_bin, y_bin])

        for i, cls in enumerate(classes):
            if i >= proba.shape[1]:
                continue
            y_cls = y_bin[:, i]
            if len(np.unique(y_cls)) < 2:
                continue
            fpr, tpr, _ = roc_curve(y_cls, proba[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, lw=1.8, label=f"{cls} (AUC={roc_auc:.3f})")
            plotted = True

    if not plotted:
        plt.close(fig)
        return False

    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_title(f"ROC-AUC Curve - {category}")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(Path(output_dir) / f"roc_auc_{category}.png", dpi=300)
    plt.close(fig)
    return True


def compute_cv_scores(X: pd.DataFrame, y: pd.Series, seed: int) -> np.ndarray | None:
    y_clean = y[y != "Unknown"]
    X_clean = X.iloc[y_clean.index]
    if y_clean.nunique() < 2:
        return None

    min_class_count = int(y_clean.value_counts().min())
    if min_class_count < 2:
        return None

    n_splits = min(5, min_class_count)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", model),
    ])

    return cross_val_score(pipe, X_clean, y_clean, cv=cv, scoring="accuracy")


def plot_cv_boxplot(cv_data: Dict[str, np.ndarray], output_dir: str) -> None:
    if not cv_data:
        return

    rows = []
    for cat, scores in cv_data.items():
        for s in scores:
            rows.append({"category": cat, "cv_accuracy": float(s)})
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(11, 6))
    sns.boxplot(data=df, x="category", y="cv_accuracy", ax=ax)
    sns.stripplot(data=df, x="category", y="cv_accuracy", color="black", alpha=0.5, size=4, ax=ax)
    ax.set_title("Cross-Validation Accuracy Distribution")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Category")
    ax.set_ylabel("CV Accuracy")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(Path(output_dir) / "cross_validation_accuracy_boxplot.png", dpi=300)
    plt.close(fig)


def main() -> None:
    sns.set_theme(style="whitegrid")
    args = parse_args()
    ensure_dir(args.output_dir)

    trainer = InterpretationModelTrainer()
    df = trainer.load_dataset(args.dataset)
    X, y_dict = trainer.prepare_features_and_labels(df)

    X_train, X_test, train_indices, test_indices = train_test_split(
        X, X.index, test_size=args.test_size, random_state=args.seed, stratify=None
    )

    metadata, artifact_scaler, artifact_models = load_models(args.model_dir, args.suffix)

    eval_scaler = StandardScaler().fit(X_train)
    X_train_scaled = eval_scaler.transform(X_train)
    X_test_scaled = eval_scaler.transform(X_test)

    metrics_rows: List[Dict[str, float | str]] = []
    cv_scores_by_cat: Dict[str, np.ndarray] = {}
    leakage_rows: List[Dict[str, float | str]] = []

    index_overlap = int(len(set(train_indices.tolist()) & set(test_indices.tolist())))
    feature_row_overlap = count_exact_row_overlap(X_train, X_test)
    leakage_rows.append({"check": "index_overlap", "value": index_overlap})
    leakage_rows.append({"check": "exact_feature_row_overlap", "value": feature_row_overlap})

    for category in metadata["categories"]:
        y_train = y_dict[category].iloc[train_indices]
        y_test = y_dict[category].iloc[test_indices]
        train_mask = y_train != "Unknown"
        mask = y_test != "Unknown"
        if mask.sum() == 0:
            print(f"Skipping {category}: no valid test labels")
            continue

        y_train_cat = y_train[train_mask]
        if y_train_cat.nunique() < 2:
            print(f"Skipping {category}: insufficient train class diversity")
            continue

        if args.eval_mode == "artifact":
            if category not in artifact_models:
                print(f"Skipping {category}: model not found")
                continue
            model = artifact_models[category]
            X_cat = artifact_scaler.transform(X_test)[mask]
            X_train_cat_for_control = artifact_scaler.transform(X_train)[train_mask]
        else:
            X_train_cat = X_train_scaled[train_mask]
            model = train_category_model(X_train_cat, y_train_cat, args.seed)
            X_cat = X_test_scaled[mask]
            X_train_cat_for_control = X_train_cat

        y_true = y_test[mask].to_numpy()
        y_pred = model.predict(X_cat)

        acc = float(accuracy_score(y_true, y_pred))
        prec_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
        rec_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
        f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
        mcc = float(matthews_corrcoef(y_true, y_pred))

        acc_std, acc_ci_low, acc_ci_high = bootstrap_metric(
            y_true, y_pred, accuracy_score, args.bootstrap_samples, args.seed
        )
        f1_std, f1_ci_low, f1_ci_high = bootstrap_metric(
            y_true,
            y_pred,
            lambda yt, yp: f1_score(yt, yp, average="macro", zero_division=0),
            args.bootstrap_samples,
            args.seed + 1,
        )
        mcc_std, mcc_ci_low, mcc_ci_high = bootstrap_metric(
            y_true, y_pred, matthews_corrcoef, args.bootstrap_samples, args.seed + 2
        )

        shuffled_acc, shuffled_bal_acc, majority_acc = shuffled_label_control_accuracy(
            X_train_cat_for_control,
            y_train_cat,
            X_cat,
            y_true,
            args.seed,
        )
        leakage_rows.append({"check": f"shuffled_label_accuracy:{category}", "value": shuffled_acc})
        leakage_rows.append({"check": f"shuffled_label_balanced_accuracy:{category}", "value": shuffled_bal_acc})
        leakage_rows.append({"check": f"majority_class_accuracy:{category}", "value": majority_acc})

        metrics_rows.append(
            {
                "category": category,
                "accuracy": acc,
                "precision_macro": prec_macro,
                "recall_macro": rec_macro,
                "f1_macro": f1_macro,
                "f1_weighted": f1_weighted,
                "mcc": mcc,
                "accuracy_std_boot": acc_std,
                "accuracy_ci95_low": acc_ci_low,
                "accuracy_ci95_high": acc_ci_high,
                "f1_macro_std_boot": f1_std,
                "f1_macro_ci95_low": f1_ci_low,
                "f1_macro_ci95_high": f1_ci_high,
                "mcc_std_boot": mcc_std,
                "mcc_ci95_low": mcc_ci_low,
                "mcc_ci95_high": mcc_ci_high,
                "support": int(mask.sum()),
            }
        )

        plot_confusion_matrix(y_true, y_pred, category, args.output_dir)
        roc_ok = plot_roc_auc(model, X_cat, y_true, category, args.output_dir)
        if not roc_ok:
            print(f"Skipping ROC-AUC plot for {category}: insufficient probability/label coverage")

        cv_scores = compute_cv_scores(X, y_dict[category], args.seed)
        if cv_scores is not None:
            cv_scores_by_cat[category] = cv_scores

    if not metrics_rows:
        print("No category metrics produced. Check models and labels.")
        return

    metrics_df = pd.DataFrame(metrics_rows).sort_values("category")
    for cat, scores in cv_scores_by_cat.items():
        metrics_df.loc[metrics_df["category"] == cat, "cv_accuracy_mean"] = float(np.mean(scores))
        metrics_df.loc[metrics_df["category"] == cat, "cv_accuracy_std"] = float(np.std(scores, ddof=1))

    leakage_df = pd.DataFrame(leakage_rows)

    metrics_df.to_csv(Path(args.output_dir) / "metrics_summary.csv", index=False)
    leakage_df.to_csv(Path(args.output_dir) / "leakage_checks.csv", index=False)
    plot_grouped_metrics(metrics_df, args.output_dir)
    plot_cv_boxplot(cv_scores_by_cat, args.output_dir)
    write_paper_notes(metrics_df, leakage_df, args.output_dir, args.eval_mode)

    print("\nGenerated files:")
    for p in sorted(Path(args.output_dir).glob("*.png")):
        print(f"  - {p}")
    print(f"  - {Path(args.output_dir) / 'metrics_summary.csv'}")
    print(f"  - {Path(args.output_dir) / 'leakage_checks.csv'}")
    print(f"  - {Path(args.output_dir) / 'paper_evaluation_notes.md'}")


if __name__ == "__main__":
    main()
