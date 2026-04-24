#!/usr/bin/env python3
"""
Compare Selective Prediction Methods for IEEE Paper Novelty

This script implements multiple selective prediction (abstention) methods
and compares them to demonstrate the novelty of our adaptive routing approach.

Methods compared:
1. Max Confidence Threshold (baseline) - abstain if max_prob < threshold
2. Entropy Threshold - abstain if prediction entropy > threshold
3. Ensemble Disagreement - train multiple models, abstain if they disagree
4. Our Method: Confidence + Rule-ML Disagreement (adaptive routing)

Additional analyses:
- Distribution shift robustness
- Abstained case analysis
- Expected Calibration Error (ECE)
- Per-category routing breakdown
- Confidence distribution visualization
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.calibration import calibration_curve
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, brier_score_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize

from train_interpretation_model import InterpretationModelTrainer
from src.predictor import ClinicalPredictor

warnings.filterwarnings('ignore')

# ============================================================================
# SELECTIVE PREDICTION METHODS
# ============================================================================

def max_confidence_selection(
    ml_conf: float,
    threshold: float,
    **kwargs
) -> Tuple[str, str]:
    """Baseline: abstain if max confidence < threshold."""
    if ml_conf >= threshold:
        return "Predict", "high_confidence"
    return "Abstain", "low_confidence"


def entropy_selection(
    proba: np.ndarray,
    threshold: float,
    **kwargs
) -> Tuple[str, str]:
    """Abstain if prediction entropy > threshold."""
    # Entropy: -sum(p * log(p))
    proba_safe = np.clip(proba, 1e-10, 1.0)
    entropy = -np.sum(proba_safe * np.log(proba_safe))
    max_entropy = np.log(len(proba))  # Normalize by max possible entropy
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

    if normalized_entropy <= threshold:
        return "Predict", f"low_entropy_{normalized_entropy:.3f}"
    return "Abstain", f"high_entropy_{normalized_entropy:.3f}"


def ensemble_disagreement_selection(
    ensemble_preds: List[str],
    min_agreement: float = 0.6,
    **kwargs
) -> Tuple[str, str]:
    """Abstain if ensemble models disagree beyond threshold."""
    if not ensemble_preds:
        return "Abstain", "no_predictions"

    # Calculate agreement ratio
    from collections import Counter
    counts = Counter(ensemble_preds)
    most_common_count = counts.most_common(1)[0][1]
    agreement_ratio = most_common_count / len(ensemble_preds)

    if agreement_ratio >= min_agreement:
        return "Predict", f"agreement_{agreement_ratio:.2f}"
    return "Abstain", f"disagreement_{agreement_ratio:.2f}"


def adaptive_routing_selection(
    ml_conf: float,
    rule_pred: str,
    ml_pred: str,
    has_required_features: bool,
    ml_threshold: float = 0.85,
    abstain_threshold: float = 0.55,
    disagreement_margin: float = 0.20,
    **kwargs
) -> Tuple[str, str]:
    """Our method: confidence + rule-ML disagreement."""
    if not has_required_features:
        return "Rule", "missing_features"

    disagree = rule_pred != ml_pred
    high_conf = ml_conf >= ml_threshold
    conflict_low_conf = disagree and ml_conf < (ml_threshold + disagreement_margin)

    if high_conf and not conflict_low_conf:
        return "ML", "high_confidence_agreement"

    if disagree and ml_conf < abstain_threshold:
        return "Abstain", "low_confidence_disagreement"

    return "Rule", "fallback_rule"


# ============================================================================
# ENSEMBLE TRAINING
# ============================================================================

def train_ensemble_models(
    X_train: np.ndarray,
    y_train: pd.Series,
    n_models: int = 5,
    seed: int = 42
) -> List[GradientBoostingClassifier]:
    """Train an ensemble of models with different random seeds."""
    models = []
    for i in range(n_models):
        model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=seed + i
        )
        # Bootstrap sampling for diversity
        np.random.seed(seed + i)
        indices = np.random.choice(len(X_train), size=len(X_train), replace=True)
        model.fit(X_train[indices], y_train.iloc[indices])
        models.append(model)
    return models


def get_ensemble_predictions(
    models: List[GradientBoostingClassifier],
    X: np.ndarray
) -> Tuple[List[str], np.ndarray]:
    """Get predictions from all ensemble members."""
    all_preds = []
    all_probas = []

    for model in models:
        pred = model.predict(X)[0]
        all_preds.append(pred)
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X)[0]
            all_probas.append(proba)

    # Average probabilities
    if all_probas:
        avg_proba = np.mean(all_probas, axis=0)
    else:
        avg_proba = np.array([1.0])

    return all_preds, avg_proba


# ============================================================================
# EVALUATION FRAMEWORK
# ============================================================================

def evaluate_selective_method(
    method_name: str,
    records: List[Dict[str, Any]],
    selection_func,
    **method_kwargs
) -> Dict[str, Any]:
    """Evaluate a selective prediction method."""
    y_true_all = []
    y_pred_all = []
    decisions = []

    for rec in records:
        y_true = rec["y_true"]

        # Get selection decision based on method
        if method_name.startswith("max_conf"):
            decision, reason = selection_func(
                ml_conf=rec["ml_conf"],
                threshold=method_kwargs.get("threshold", 0.85)
            )
            y_pred = rec["ml_pred"] if decision == "Predict" else "ABSTAIN"

        elif method_name.startswith("entropy"):
            decision, reason = selection_func(
                proba=np.array(rec.get("ml_proba", [rec["ml_conf"]])),
                threshold=method_kwargs.get("threshold", 0.5)
            )
            y_pred = rec["ml_pred"] if decision == "Predict" else "ABSTAIN"

        elif method_name.startswith("ensemble"):
            decision, reason = selection_func(
                ensemble_preds=rec.get("ensemble_preds", [rec["ml_pred"]]),
                min_agreement=method_kwargs.get("min_agreement", 0.6)
            )
            # Use majority vote for prediction
            if decision == "Predict" and rec.get("ensemble_preds"):
                from collections import Counter
                y_pred = Counter(rec["ensemble_preds"]).most_common(1)[0][0]
            else:
                y_pred = "ABSTAIN" if decision == "Abstain" else rec["ml_pred"]

        elif method_name == "adaptive_routing":
            decision, reason = selection_func(
                ml_conf=rec["ml_conf"],
                rule_pred=rec["rule_pred"],
                ml_pred=rec["ml_pred"],
                has_required_features=rec["required_ok"],
                **method_kwargs
            )
            if decision == "ML":
                y_pred = rec["ml_pred"]
            elif decision == "Rule":
                y_pred = rec["rule_pred"]
            else:
                y_pred = "ABSTAIN"
        else:
            raise ValueError(f"Unknown method: {method_name}")

        y_true_all.append(y_true)
        y_pred_all.append(y_pred)
        decisions.append(decision)

    # Calculate metrics
    n_total = len(y_true_all)
    n_abstain = sum(1 for d in decisions if d == "Abstain")
    n_covered = n_total - n_abstain

    covered_true = [t for t, p in zip(y_true_all, y_pred_all) if p != "ABSTAIN"]
    covered_pred = [p for p in y_pred_all if p != "ABSTAIN"]

    coverage = n_covered / n_total if n_total else 0.0
    abstain_rate = n_abstain / n_total if n_total else 0.0

    if n_covered > 0:
        covered_acc = accuracy_score(covered_true, covered_pred)
        covered_f1 = f1_score(covered_true, covered_pred, average="macro", zero_division=0)
        risk_at_coverage = 1.0 - covered_acc
    else:
        covered_acc = 0.0
        covered_f1 = 0.0
        risk_at_coverage = 1.0

    # Strict accuracy (abstain counted as error)
    strict_acc = sum(1 for yt, yp in zip(y_true_all, y_pred_all) if yt == yp) / n_total if n_total else 0.0

    return {
        "method": method_name,
        "coverage": coverage,
        "abstain_rate": abstain_rate,
        "covered_accuracy": covered_acc,
        "covered_f1": covered_f1,
        "risk_at_coverage": risk_at_coverage,
        "strict_accuracy": strict_acc,
        "n_total": n_total,
        "n_abstain": n_abstain,
    }


# ============================================================================
# DISTRIBUTION SHIFT ANALYSIS
# ============================================================================

def add_gaussian_noise(X: np.ndarray, sigma: float, seed: int = 42) -> np.ndarray:
    """Add Gaussian noise to features to simulate distribution shift."""
    np.random.seed(seed)
    noise = np.random.normal(0, sigma, X.shape)
    return X + noise


def evaluate_under_distribution_shift(
    records_by_category: Dict[str, List[Dict[str, Any]]],
    X_test_scaled: np.ndarray,
    predictor: ClinicalPredictor,
    trainer: InterpretationModelTrainer,
    noise_levels: List[float] = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0],
    ml_threshold: float = 0.85,
    abstain_threshold: float = 0.55,
) -> pd.DataFrame:
    """Evaluate selective prediction under increasing distribution shift."""
    results = []

    for sigma in noise_levels:
        # Add noise to features
        X_noisy = add_gaussian_noise(X_test_scaled, sigma)

        # Re-compute predictions with noisy features
        category_results = []
        for category, records in records_by_category.items():
            model = predictor.models.get(category)
            if model is None:
                continue

            noisy_records = []
            for i, rec in enumerate(records):
                # Get new prediction with noisy features
                x_row = X_noisy[i:i+1]
                ml_pred = str(model.predict(x_row)[0])
                ml_conf = 0.0
                if hasattr(model, 'predict_proba'):
                    try:
                        proba = model.predict_proba(x_row)[0]
                        ml_conf = float(np.max(proba))
                    except:
                        pass

                noisy_rec = dict(rec)
                noisy_rec["ml_pred"] = ml_pred
                noisy_rec["ml_conf"] = ml_conf
                noisy_records.append(noisy_rec)

            # Evaluate with adaptive routing
            result = evaluate_selective_method(
                "adaptive_routing",
                noisy_records,
                adaptive_routing_selection,
                ml_threshold=ml_threshold,
                abstain_threshold=abstain_threshold,
            )
            result["category"] = category
            result["noise_sigma"] = sigma
            category_results.append(result)

        # Aggregate across categories
        if category_results:
            df_cat = pd.DataFrame(category_results)
            weights = df_cat["n_total"].values / df_cat["n_total"].sum()

            results.append({
                "noise_sigma": sigma,
                "weighted_coverage": float(np.sum(df_cat["coverage"].values * weights)),
                "weighted_abstain_rate": float(np.sum(df_cat["abstain_rate"].values * weights)),
                "weighted_covered_accuracy": float(np.sum(df_cat["covered_accuracy"].values * weights)),
                "weighted_covered_f1": float(np.sum(df_cat["covered_f1"].values * weights)),
                "weighted_risk": float(np.sum(df_cat["risk_at_coverage"].values * weights)),
            })

    return pd.DataFrame(results)


# ============================================================================
# CALIBRATION ANALYSIS (ECE)
# ============================================================================

def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            avg_confidence = y_prob[in_bin].mean()
            avg_accuracy = y_true[in_bin].mean()
            ece += np.abs(avg_confidence - avg_accuracy) * prop_in_bin

    return ece


def calibration_analysis(
    records_by_category: Dict[str, List[Dict[str, Any]]],
    n_bins: int = 10
) -> Dict[str, Any]:
    """Compute calibration metrics per category."""
    results = {}

    for category, records in records_by_category.items():
        y_true = []
        y_conf = []

        for rec in records:
            # Binary: correct (1) or incorrect (0)
            is_correct = 1 if rec["ml_pred"] == rec["y_true"] else 0
            y_true.append(is_correct)
            y_conf.append(rec["ml_conf"])

        y_true = np.array(y_true)
        y_conf = np.array(y_conf)

        ece = compute_ece(y_true, y_conf, n_bins)

        # Reliability curve
        try:
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true, y_conf, n_bins=n_bins, strategy='uniform'
            )
        except:
            fraction_of_positives = np.array([])
            mean_predicted_value = np.array([])

        results[category] = {
            "ece": ece,
            "n_samples": len(records),
            "mean_confidence": float(np.mean(y_conf)),
            "accuracy": float(np.mean(y_true)),
            "fraction_of_positives": fraction_of_positives.tolist(),
            "mean_predicted_value": mean_predicted_value.tolist(),
        }

    return results


# ============================================================================
# ABSTAINED CASE ANALYSIS
# ============================================================================

def analyze_abstained_cases(
    records_by_category: Dict[str, List[Dict[str, Any]]],
    ml_threshold: float = 0.85,
    abstain_threshold: float = 0.55,
) -> Dict[str, Any]:
    """Analyze characteristics of abstained vs non-abstained cases."""
    analysis = {
        "per_category": {},
        "overall": {}
    }

    all_abstained_conf = []
    all_predicted_conf = []
    all_abstained_correct = []
    all_predicted_correct = []

    for category, records in records_by_category.items():
        abstained = []
        predicted = []

        for rec in records:
            decision, _ = adaptive_routing_selection(
                ml_conf=rec["ml_conf"],
                rule_pred=rec["rule_pred"],
                ml_pred=rec["ml_pred"],
                has_required_features=rec["required_ok"],
                ml_threshold=ml_threshold,
                abstain_threshold=abstain_threshold,
            )

            is_ml_correct = rec["ml_pred"] == rec["y_true"]
            is_rule_correct = rec["rule_pred"] == rec["y_true"]
            disagree = rec["ml_pred"] != rec["rule_pred"]

            case_info = {
                "ml_conf": rec["ml_conf"],
                "is_ml_correct": is_ml_correct,
                "is_rule_correct": is_rule_correct,
                "disagreement": disagree,
            }

            if decision == "Abstain":
                abstained.append(case_info)
                all_abstained_conf.append(rec["ml_conf"])
                all_abstained_correct.append(1 if is_ml_correct else 0)
            else:
                predicted.append(case_info)
                all_predicted_conf.append(rec["ml_conf"])
                all_predicted_correct.append(1 if is_ml_correct else 0)

        # Per-category statistics
        if abstained:
            abstained_df = pd.DataFrame(abstained)
            analysis["per_category"][category] = {
                "n_abstained": len(abstained),
                "n_predicted": len(predicted),
                "abstain_rate": len(abstained) / (len(abstained) + len(predicted)),
                "abstained_mean_conf": float(abstained_df["ml_conf"].mean()),
                "abstained_ml_accuracy": float(abstained_df["is_ml_correct"].mean()),
                "abstained_rule_accuracy": float(abstained_df["is_rule_correct"].mean()),
                "abstained_disagreement_rate": float(abstained_df["disagreement"].mean()),
            }
        else:
            analysis["per_category"][category] = {
                "n_abstained": 0,
                "n_predicted": len(predicted),
                "abstain_rate": 0.0,
            }

    # Overall statistics
    if all_abstained_conf:
        analysis["overall"] = {
            "n_abstained": len(all_abstained_conf),
            "n_predicted": len(all_predicted_conf),
            "abstained_mean_conf": float(np.mean(all_abstained_conf)),
            "predicted_mean_conf": float(np.mean(all_predicted_conf)),
            "abstained_would_be_correct": float(np.mean(all_abstained_correct)),
            "predicted_accuracy": float(np.mean(all_predicted_correct)),
            "confidence_gap": float(np.mean(all_predicted_conf) - np.mean(all_abstained_conf)),
        }

    return analysis


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_method_comparison(results_df: pd.DataFrame, out_path: Path) -> None:
    """Plot comparison of selective prediction methods."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    methods = results_df["method"].values
    x = np.arange(len(methods))
    width = 0.35

    # Coverage vs Accuracy
    ax = axes[0]
    ax.bar(x - width/2, results_df["coverage"], width, label="Coverage", color="steelblue")
    ax.bar(x + width/2, results_df["covered_accuracy"], width, label="Covered Accuracy", color="forestgreen")
    ax.set_xlabel("Method")
    ax.set_ylabel("Value")
    ax.set_title("Coverage vs Covered Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3)

    # Risk at Coverage
    ax = axes[1]
    colors = ['coral' if m != 'adaptive_routing' else 'crimson' for m in methods]
    bars = ax.bar(x, results_df["risk_at_coverage"], color=colors, edgecolor='black')
    ax.set_xlabel("Method")
    ax.set_ylabel("Risk (1 - Accuracy)")
    ax.set_title("Risk at Coverage")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha="right")
    ax.grid(axis='y', alpha=0.3)

    # Highlight our method
    for i, m in enumerate(methods):
        if m == 'adaptive_routing':
            bars[i].set_edgecolor('gold')
            bars[i].set_linewidth(3)

    # Abstention Rate
    ax = axes[2]
    ax.bar(x, results_df["abstain_rate"], color="mediumpurple", edgecolor='black')
    ax.set_xlabel("Method")
    ax.set_ylabel("Abstention Rate")
    ax.set_title("Abstention Rate")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha="right")
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_distribution_shift(shift_df: pd.DataFrame, out_path: Path) -> None:
    """Plot performance under distribution shift."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Coverage and Abstention vs Noise
    ax = axes[0]
    ax.plot(shift_df["noise_sigma"], shift_df["weighted_coverage"],
            'b-o', label="Coverage", linewidth=2, markersize=8)
    ax.plot(shift_df["noise_sigma"], shift_df["weighted_abstain_rate"],
            'r--s', label="Abstention Rate", linewidth=2, markersize=8)
    ax.set_xlabel("Noise Level (σ)")
    ax.set_ylabel("Rate")
    ax.set_title("Safety-Aware Behavior Under Distribution Shift")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    # Accuracy vs Noise
    ax = axes[1]
    ax.plot(shift_df["noise_sigma"], shift_df["weighted_covered_accuracy"],
            'g-o', label="Covered Accuracy", linewidth=2, markersize=8)
    ax.plot(shift_df["noise_sigma"], shift_df["weighted_covered_f1"],
            'm--s', label="Covered F1", linewidth=2, markersize=8)
    ax.set_xlabel("Noise Level (σ)")
    ax.set_ylabel("Metric Value")
    ax.set_title("Covered Performance Under Distribution Shift")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_calibration(calibration_results: Dict[str, Any], out_path: Path) -> None:
    """Plot reliability diagrams for calibration analysis."""
    categories = list(calibration_results.keys())
    n_cats = len(categories)

    fig, axes = plt.subplots(1, min(n_cats, 5), figsize=(4*min(n_cats, 5), 4))
    if n_cats == 1:
        axes = [axes]

    for i, (cat, data) in enumerate(calibration_results.items()):
        if i >= 5:
            break
        ax = axes[i]

        # Perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')

        # Actual calibration
        if len(data["fraction_of_positives"]) > 0:
            ax.plot(data["mean_predicted_value"], data["fraction_of_positives"],
                   'b-o', label=f'Model (ECE={data["ece"]:.3f})')

        ax.set_xlabel("Mean Predicted Confidence")
        ax.set_ylabel("Fraction Correct")
        ax.set_title(f"{cat}\nECE: {data['ece']:.4f}")
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_confidence_distribution(
    records_by_category: Dict[str, List[Dict[str, Any]]],
    out_path: Path
) -> None:
    """Plot confidence distributions for correct vs incorrect vs abstained."""
    # Aggregate all records
    correct_conf = []
    incorrect_conf = []

    for category, records in records_by_category.items():
        for rec in records:
            if rec["ml_pred"] == rec["y_true"]:
                correct_conf.append(rec["ml_conf"])
            else:
                incorrect_conf.append(rec["ml_conf"])

    fig, ax = plt.subplots(figsize=(10, 6))

    bins = np.linspace(0, 1, 21)

    ax.hist(correct_conf, bins=bins, alpha=0.6, label=f'Correct (n={len(correct_conf)})',
            color='forestgreen', edgecolor='black')
    ax.hist(incorrect_conf, bins=bins, alpha=0.6, label=f'Incorrect (n={len(incorrect_conf)})',
            color='coral', edgecolor='black')

    # Add vertical lines for thresholds
    ax.axvline(x=0.85, color='blue', linestyle='--', linewidth=2, label='ML Threshold (0.85)')
    ax.axvline(x=0.55, color='red', linestyle='--', linewidth=2, label='Abstain Threshold (0.55)')

    ax.set_xlabel("ML Confidence")
    ax.set_ylabel("Frequency")
    ax.set_title("Confidence Distribution: Correct vs Incorrect Predictions")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_per_category_routing(
    records_by_category: Dict[str, List[Dict[str, Any]]],
    ml_threshold: float,
    abstain_threshold: float,
    out_path: Path
) -> None:
    """Plot routing decisions breakdown by category."""
    categories = []
    ml_rates = []
    rule_rates = []
    abstain_rates = []

    for category, records in records_by_category.items():
        ml_count = 0
        rule_count = 0
        abstain_count = 0

        for rec in records:
            decision, _ = adaptive_routing_selection(
                ml_conf=rec["ml_conf"],
                rule_pred=rec["rule_pred"],
                ml_pred=rec["ml_pred"],
                has_required_features=rec["required_ok"],
                ml_threshold=ml_threshold,
                abstain_threshold=abstain_threshold,
            )
            if decision == "ML":
                ml_count += 1
            elif decision == "Rule":
                rule_count += 1
            else:
                abstain_count += 1

        total = len(records)
        categories.append(category.replace("_", "\n"))
        ml_rates.append(ml_count / total if total else 0)
        rule_rates.append(rule_count / total if total else 0)
        abstain_rates.append(abstain_count / total if total else 0)

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(categories))
    width = 0.25

    ax.bar(x - width, ml_rates, width, label='ML Selected', color='steelblue')
    ax.bar(x, rule_rates, width, label='Rule Selected', color='forestgreen')
    ax.bar(x + width, abstain_rates, width, label='Abstained', color='coral')

    ax.set_xlabel("Category")
    ax.set_ylabel("Rate")
    ax.set_title("Per-Category Routing Decisions")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


# ============================================================================
# MAIN EVALUATION
# ============================================================================

def build_eval_records_with_ensemble(
    trainer: InterpretationModelTrainer,
    predictor: ClinicalPredictor,
    X_train: np.ndarray,
    X_test_scaled: np.ndarray,
    test_indices: List[int],
    raw_items: List[Dict[str, Any]],
    y_dict: Dict[str, pd.Series],
    scaler: StandardScaler,
    n_ensemble: int = 5,
    seed: int = 42,
) -> Dict[str, List[Dict[str, Any]]]:
    """Build evaluation records with ensemble predictions."""
    records_by_category: Dict[str, List[Dict[str, Any]]] = {}
    ensemble_models_by_category: Dict[str, List] = {}

    # Train ensemble models for each category
    for category in trainer.categories:
        y_all = y_dict[category]
        # Get training indices (not in test)
        train_mask = ~y_all.index.isin(test_indices)
        y_train = y_all[train_mask]
        train_mask_arr = train_mask.values if hasattr(train_mask, 'values') else train_mask
        X_train_cat = X_train[train_mask_arr]

        # Filter out Unknown labels
        valid_mask = y_train != "Unknown"
        if valid_mask.sum() < 10:
            continue

        y_train_valid = y_train[valid_mask]
        valid_mask_arr = valid_mask.values if hasattr(valid_mask, 'values') else valid_mask
        X_train_valid = scaler.transform(X_train_cat[valid_mask_arr])

        # Train ensemble
        ensemble = train_ensemble_models(X_train_valid, y_train_valid, n_ensemble, seed)
        ensemble_models_by_category[category] = ensemble

    # Build records
    for category in trainer.categories:
        model = predictor.models.get(category)
        if model is None:
            continue

        y_test = y_dict[category].iloc[test_indices]
        category_records: List[Dict[str, Any]] = []
        ensemble = ensemble_models_by_category.get(category, [])

        for pos, global_idx in enumerate(test_indices):
            y_true = str(y_test.iloc[pos])
            if y_true == "Unknown":
                continue

            item = raw_items[global_idx]
            measurements = item.get("measurements", {})
            patient_info = item.get("patient", {})

            rule_text = predictor.rule_engine.interpret_measurements(measurements, patient_info)
            rule_labels = trainer._extract_labels(rule_text, measurements, patient_info)
            rule_pred = str(rule_labels.get(category, "Unknown"))

            x_row_scaled = X_test_scaled[pos:pos+1]
            ml_pred = str(model.predict(x_row_scaled)[0])

            # Get confidence and probabilities
            ml_conf = 0.0
            ml_proba = []
            if hasattr(model, 'predict_proba'):
                try:
                    proba = model.predict_proba(x_row_scaled)[0]
                    ml_conf = float(np.max(proba))
                    ml_proba = proba.tolist()
                except:
                    pass

            # Get ensemble predictions
            ensemble_preds = []
            if ensemble:
                for ens_model in ensemble:
                    try:
                        ens_pred = str(ens_model.predict(x_row_scaled)[0])
                        ensemble_preds.append(ens_pred)
                    except:
                        pass

            required_features = {
                "LV_FUNCTION": ["EF", "FS"],
                "LV_SIZE": ["LVID_D"],
                "LV_HYPERTROPHY": ["IVS_D"],
                "LA_SIZE": ["LA_DIMENSION"],
                "DIASTOLIC_FUNCTION": ["MV_E_A"],
            }
            feats = required_features.get(category, [])
            required_ok = any((measurements.get(f, 0) or 0) > 0 for f in feats) if feats else True

            category_records.append({
                "y_true": y_true,
                "rule_pred": rule_pred,
                "ml_pred": ml_pred,
                "ml_conf": ml_conf,
                "ml_proba": ml_proba,
                "ensemble_preds": ensemble_preds,
                "required_ok": required_ok,
            })

        if category_records:
            records_by_category[category] = category_records

    return records_by_category


def main():
    parser = argparse.ArgumentParser(description="Compare Selective Prediction Methods")
    parser.add_argument("--dataset", default="data/processed_new/training_dataset_new.json")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--output-dir", default="outputs/selective_prediction_comparison")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ml-threshold", type=float, default=0.85)
    parser.add_argument("--abstain-threshold", type=float, default=0.55)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("SELECTIVE PREDICTION METHOD COMPARISON")
    print("=" * 70)

    # Load data and models
    print("\n1. Loading data and models...")
    trainer = InterpretationModelTrainer()
    df = trainer.load_dataset(args.dataset)
    X, y_dict = trainer.prepare_features_and_labels(df)

    with open(args.dataset, 'r') as f:
        raw_items = json.load(f)

    predictor = ClinicalPredictor(
        model_dir=args.model_dir,
        use_ml=True,
        routing_mode="adaptive",
    )

    if not predictor.ml_available:
        raise RuntimeError("ML models not available")

    # Split data
    X_train, X_test, train_indices, test_indices = train_test_split(
        X, X.index, test_size=args.test_size, random_state=args.seed
    )
    X_test_scaled = predictor.scaler.transform(X_test)
    test_index_list = list(test_indices)

    # Build records with ensemble predictions
    print("\n2. Building evaluation records with ensemble models...")
    records_by_category = build_eval_records_with_ensemble(
        trainer=trainer,
        predictor=predictor,
        X_train=X.values,
        X_test_scaled=X_test_scaled,
        test_indices=test_index_list,
        raw_items=raw_items,
        y_dict=y_dict,
        scaler=predictor.scaler,
        n_ensemble=5,
        seed=args.seed,
    )

    # Flatten records for method comparison
    all_records = []
    for cat, recs in records_by_category.items():
        all_records.extend(recs)

    print(f"   Total test samples: {len(all_records)}")

    # ========================================================================
    # COMPARE SELECTIVE PREDICTION METHODS
    # ========================================================================
    print("\n3. Comparing selective prediction methods...")

    method_results = []

    # Method 1: Max Confidence (baseline)
    for threshold in [0.70, 0.85, 0.90]:
        result = evaluate_selective_method(
            f"max_conf_{threshold}",
            all_records,
            max_confidence_selection,
            threshold=threshold
        )
        result["method_full"] = f"Max Confidence (τ={threshold})"
        method_results.append(result)

    # Method 2: Entropy threshold
    for threshold in [0.3, 0.5, 0.7]:
        result = evaluate_selective_method(
            f"entropy_{threshold}",
            all_records,
            entropy_selection,
            threshold=threshold
        )
        result["method_full"] = f"Entropy (τ={threshold})"
        method_results.append(result)

    # Method 3: Ensemble disagreement
    for min_agree in [0.6, 0.8, 1.0]:
        result = evaluate_selective_method(
            f"ensemble_{min_agree}",
            all_records,
            ensemble_disagreement_selection,
            min_agreement=min_agree
        )
        result["method_full"] = f"Ensemble (agree≥{min_agree})"
        method_results.append(result)

    # Method 4: Our adaptive routing
    result = evaluate_selective_method(
        "adaptive_routing",
        all_records,
        adaptive_routing_selection,
        ml_threshold=args.ml_threshold,
        abstain_threshold=args.abstain_threshold,
    )
    result["method_full"] = "Ours: Adaptive Routing"
    method_results.append(result)

    results_df = pd.DataFrame(method_results)
    results_df.to_csv(out_dir / "method_comparison.csv", index=False)
    print(f"   Saved: {out_dir / 'method_comparison.csv'}")

    # Print comparison table
    print("\n" + "=" * 90)
    print("SELECTIVE PREDICTION METHOD COMPARISON")
    print("=" * 90)
    print(f"{'Method':<30} {'Coverage':>10} {'Abstain':>10} {'Cov.Acc':>10} {'Cov.F1':>10} {'Risk':>10}")
    print("-" * 90)
    for _, row in results_df.iterrows():
        print(f"{row['method_full']:<30} {row['coverage']:>10.3f} {row['abstain_rate']:>10.3f} "
              f"{row['covered_accuracy']:>10.3f} {row['covered_f1']:>10.3f} {row['risk_at_coverage']:>10.4f}")
    print("=" * 90)

    # Plot comparison (select representative methods)
    plot_methods = ["max_conf_0.85", "entropy_0.5", "ensemble_0.8", "adaptive_routing"]
    plot_df = results_df[results_df["method"].isin(plot_methods)].copy()
    plot_df["method"] = plot_df["method"].replace({
        "max_conf_0.85": "Max Confidence",
        "entropy_0.5": "Entropy",
        "ensemble_0.8": "Ensemble",
        "adaptive_routing": "Ours (Adaptive)",
    })
    plot_method_comparison(plot_df, out_dir / "method_comparison_plot.png")
    print(f"   Saved: {out_dir / 'method_comparison_plot.png'}")

    # ========================================================================
    # DISTRIBUTION SHIFT ANALYSIS
    # ========================================================================
    print("\n4. Analyzing distribution shift robustness...")

    shift_df = evaluate_under_distribution_shift(
        records_by_category=records_by_category,
        X_test_scaled=X_test_scaled,
        predictor=predictor,
        trainer=trainer,
        noise_levels=[0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0],
        ml_threshold=args.ml_threshold,
        abstain_threshold=args.abstain_threshold,
    )
    shift_df.to_csv(out_dir / "distribution_shift_analysis.csv", index=False)
    print(f"   Saved: {out_dir / 'distribution_shift_analysis.csv'}")

    plot_distribution_shift(shift_df, out_dir / "distribution_shift_plot.png")
    print(f"   Saved: {out_dir / 'distribution_shift_plot.png'}")

    print("\n   Distribution Shift Results:")
    print(f"   {'Noise σ':>10} {'Coverage':>10} {'Abstain':>10} {'Cov.Acc':>10}")
    print("   " + "-" * 45)
    for _, row in shift_df.iterrows():
        print(f"   {row['noise_sigma']:>10.2f} {row['weighted_coverage']:>10.3f} "
              f"{row['weighted_abstain_rate']:>10.3f} {row['weighted_covered_accuracy']:>10.3f}")

    # ========================================================================
    # CALIBRATION ANALYSIS
    # ========================================================================
    print("\n5. Computing calibration metrics (ECE)...")

    calibration_results = calibration_analysis(records_by_category)
    with open(out_dir / "calibration_analysis.json", 'w') as f:
        json.dump(calibration_results, f, indent=2)
    print(f"   Saved: {out_dir / 'calibration_analysis.json'}")

    plot_calibration(calibration_results, out_dir / "calibration_reliability_diagram.png")
    print(f"   Saved: {out_dir / 'calibration_reliability_diagram.png'}")

    print("\n   ECE by Category:")
    for cat, data in calibration_results.items():
        print(f"   {cat}: ECE = {data['ece']:.4f}, Acc = {data['accuracy']:.3f}, "
              f"Mean Conf = {data['mean_confidence']:.3f}")

    # ========================================================================
    # ABSTAINED CASE ANALYSIS
    # ========================================================================
    print("\n6. Analyzing abstained cases...")

    abstain_analysis = analyze_abstained_cases(
        records_by_category,
        ml_threshold=args.ml_threshold,
        abstain_threshold=args.abstain_threshold,
    )
    with open(out_dir / "abstained_case_analysis.json", 'w') as f:
        json.dump(abstain_analysis, f, indent=2)
    print(f"   Saved: {out_dir / 'abstained_case_analysis.json'}")

    if abstain_analysis.get("overall"):
        ov = abstain_analysis["overall"]
        print(f"\n   Overall Abstained Case Statistics:")
        print(f"   - N abstained: {ov.get('n_abstained', 0)}")
        print(f"   - N predicted: {ov.get('n_predicted', 0)}")
        print(f"   - Abstained mean confidence: {ov.get('abstained_mean_conf', 0):.3f}")
        print(f"   - Predicted mean confidence: {ov.get('predicted_mean_conf', 0):.3f}")
        print(f"   - If we had predicted abstained: {ov.get('abstained_would_be_correct', 0)*100:.1f}% correct")
        print(f"   - Confidence gap: {ov.get('confidence_gap', 0):.3f}")

    # ========================================================================
    # CONFIDENCE DISTRIBUTION
    # ========================================================================
    print("\n7. Plotting confidence distributions...")

    plot_confidence_distribution(records_by_category, out_dir / "confidence_distribution.png")
    print(f"   Saved: {out_dir / 'confidence_distribution.png'}")

    # ========================================================================
    # PER-CATEGORY ROUTING BREAKDOWN
    # ========================================================================
    print("\n8. Plotting per-category routing breakdown...")

    plot_per_category_routing(
        records_by_category,
        ml_threshold=args.ml_threshold,
        abstain_threshold=args.abstain_threshold,
        out_path=out_dir / "per_category_routing.png"
    )
    print(f"   Saved: {out_dir / 'per_category_routing.png'}")

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nAll outputs saved to: {out_dir}")
    print("\nKey findings for paper:")

    # Find our method's results
    our_result = results_df[results_df["method"] == "adaptive_routing"].iloc[0]
    best_baseline = results_df[results_df["method"] != "adaptive_routing"].sort_values(
        "covered_accuracy", ascending=False
    ).iloc[0]

    print(f"\n1. Our method (Adaptive Routing):")
    print(f"   - Coverage: {our_result['coverage']:.3f}")
    print(f"   - Covered Accuracy: {our_result['covered_accuracy']:.3f}")
    print(f"   - Risk at Coverage: {our_result['risk_at_coverage']:.4f}")

    print(f"\n2. Best baseline ({best_baseline['method_full']}):")
    print(f"   - Coverage: {best_baseline['coverage']:.3f}")
    print(f"   - Covered Accuracy: {best_baseline['covered_accuracy']:.3f}")
    print(f"   - Risk at Coverage: {best_baseline['risk_at_coverage']:.4f}")

    print(f"\n3. Distribution shift robustness:")
    print(f"   - At σ=0.5 noise: abstention rate increases to {shift_df[shift_df['noise_sigma']==0.5]['weighted_abstain_rate'].values[0]:.3f}")
    print(f"   - Covered accuracy maintained at {shift_df[shift_df['noise_sigma']==0.5]['weighted_covered_accuracy'].values[0]:.3f}")


if __name__ == "__main__":
    main()
