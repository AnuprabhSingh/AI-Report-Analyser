#!/usr/bin/env python3
"""
Generate IEEE-style explainability figures for trained interpretation models.

Outputs per category:
- SHAP global importance (bar)
- SHAP waterfall (single representative instance)
- SHAP dependence (top SHAP feature)
- Partial dependence + ICE (top model-importance feature)
- One-at-a-time sensitivity curves (top 3 features)
- Monte Carlo uncertainty distribution
- Global sensitivity (correlation-based)
- Pairwise interaction heatmap (top 2 features)
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.inspection import PartialDependenceDisplay

from train_interpretation_model import InterpretationModelTrainer

warnings.filterwarnings("ignore")

try:
    import shap

    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate IEEE explainability and sensitivity plots")
    parser.add_argument("--dataset", default="data/processed/combined_training_dataset.json")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--suffix", default="v2_expanded")
    parser.add_argument("--output-dir", default="outputs/paper_plots/ieee_explainability")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=250)
    parser.add_argument("--mc-simulations", type=int, default=1000)
    parser.add_argument("--oat-steps", type=int, default=25)
    return parser.parse_args()


def apply_ieee_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.8,
            "grid.linewidth": 0.4,
            "grid.alpha": 0.35,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    sns.set_theme(style="whitegrid", context="paper")


def load_artifacts(model_dir: Path, suffix: str) -> Tuple[Dict, object, Dict[str, object], List[str]]:
    suffix_part = f"_{suffix}" if suffix else ""
    metadata_path = model_dir / f"model_metadata{suffix_part}.json"
    scaler_path = model_dir / f"scaler{suffix_part}.pkl"
    feature_path = model_dir / f"feature_names{suffix_part}.json"

    metadata = json.loads(metadata_path.read_text())
    scaler = joblib.load(scaler_path)

    if hasattr(scaler, "feature_names_in_"):
        feature_names = list(scaler.feature_names_in_)
    elif feature_path.exists():
        feature_names = json.loads(feature_path.read_text())
    else:
        feature_names = metadata.get("feature_names", [])

    models: Dict[str, object] = {}
    for category in metadata["categories"]:
        model_path = model_dir / f"model_{category}{suffix_part}.pkl"
        if model_path.exists():
            models[category] = joblib.load(model_path)
    return metadata, scaler, models, feature_names


def save_figure(fig: plt.Figure, output_stem: Path) -> None:
    fig.savefig(str(output_stem) + ".png")
    plt.close(fig)


def prediction_score(model: object, X_scaled: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_scaled)
        if proba.ndim == 2:
            return np.max(proba, axis=1)
    pred = model.predict(X_scaled)
    if np.issubdtype(np.asarray(pred).dtype, np.number):
        return np.asarray(pred, dtype=float)
    return np.ones(len(X_scaled), dtype=float)


def select_top_features(model: object, feature_names: List[str], top_n: int = 3) -> List[str]:
    if hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_, dtype=float)
        if importances.size == len(feature_names) and np.sum(importances) > 0:
            idx = np.argsort(importances)[::-1][:top_n]
            return [feature_names[i] for i in idx]
    return feature_names[:top_n]


def normalize_shap_values(
    shap_values: object,
    model: object,
    X_scaled: np.ndarray,
) -> Tuple[np.ndarray, int]:
    class_idx = 0
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X_scaled)
            if proba.ndim == 2:
                class_idx = int(np.argmax(np.mean(proba, axis=0)))
        except Exception:
            class_idx = 0

    if isinstance(shap_values, list):
        class_idx = min(class_idx, len(shap_values) - 1)
        values = np.asarray(shap_values[class_idx])
        return values, class_idx

    values = np.asarray(shap_values)
    if values.ndim == 3:
        n_features = X_scaled.shape[1]
        if values.shape[1] == n_features:
            # (n_samples, n_features, n_classes)
            class_idx = min(class_idx, values.shape[2] - 1)
            return values[:, :, class_idx], class_idx
        if values.shape[2] == n_features:
            # (n_samples, n_classes, n_features)
            class_idx = min(class_idx, values.shape[1] - 1)
            return values[:, class_idx, :], class_idx
        else:
            class_idx = 0
            return values[:, 0, :], class_idx
    return values, class_idx


def plot_shap_global_bar(
    shap_2d: np.ndarray,
    feature_names: List[str],
    category: str,
    output_dir: Path,
    top_n: int = 12,
) -> List[str]:
    mean_abs = np.mean(np.abs(shap_2d), axis=0)
    order = np.argsort(mean_abs)[::-1][:top_n]
    feat = [feature_names[i] for i in order][::-1]
    vals = mean_abs[order][::-1]

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    ax.barh(feat, vals, color="#004488", alpha=0.9)
    ax.set_title(f"SHAP Global Importance: {category}")
    ax.set_xlabel("mean(|SHAP value|)")
    ax.set_ylabel("Feature")
    ax.grid(axis="x", linestyle="--")
    save_figure(fig, output_dir / f"shap_global_bar_{category}")
    return [feature_names[i] for i in order]


def plot_shap_waterfall(
    explainer: object,
    shap_2d: np.ndarray,
    X_scaled_df: pd.DataFrame,
    class_idx: int,
    category: str,
    output_dir: Path,
) -> None:
    if len(shap_2d) == 0:
        return

    idx = int(np.argmax(np.sum(np.abs(shap_2d), axis=1)))
    base_val = explainer.expected_value
    if isinstance(base_val, np.ndarray):
        flat = np.ravel(base_val)
        base_val = float(flat[min(class_idx, len(flat) - 1)])

    explanation = shap.Explanation(
        values=shap_2d[idx],
        base_values=base_val,
        data=X_scaled_df.iloc[idx].to_numpy(),
        feature_names=X_scaled_df.columns.tolist(),
    )

    plt.figure(figsize=(3.5, 2.8))
    shap.plots.waterfall(explanation, max_display=10, show=False)
    plt.title(f"SHAP Waterfall: {category}")
    plt.tight_layout()
    plt.savefig(output_dir / f"shap_waterfall_{category}.png")
    plt.close()


def plot_shap_dependence(
    shap_2d: np.ndarray,
    X_scaled_df: pd.DataFrame,
    feature: str,
    category: str,
    output_dir: Path,
) -> None:
    plt.figure(figsize=(3.5, 2.8))
    shap.dependence_plot(feature, shap_2d, X_scaled_df, show=False)
    plt.title(f"SHAP Dependence: {feature} ({category})")
    plt.tight_layout()
    plt.savefig(output_dir / f"shap_dependence_{category}_{feature}.png")
    plt.close()


def plot_pdp_ice(
    model: object,
    X_scaled_df: pd.DataFrame,
    feature: str,
    category: str,
    output_dir: Path,
) -> None:
    if feature not in X_scaled_df.columns:
        return

    feat_idx = X_scaled_df.columns.get_loc(feature)
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    pdp_kwargs = {
        "estimator": model,
        "X": X_scaled_df,
        "features": [feat_idx],
        "feature_names": X_scaled_df.columns.tolist(),
        "kind": "both",
        "ax": ax,
    }

    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X_scaled_df.values)
            if proba.ndim == 2 and proba.shape[1] > 2:
                cls_idx = int(np.argmax(np.mean(proba, axis=0)))
                if hasattr(model, "classes_") and len(model.classes_) > cls_idx:
                    pdp_kwargs["target"] = model.classes_[cls_idx]
                else:
                    pdp_kwargs["target"] = cls_idx
        except Exception:
            pass

    PartialDependenceDisplay.from_estimator(**pdp_kwargs)
    ax.set_title(f"PDP + ICE: {feature} ({category})")
    save_figure(fig, output_dir / f"pdp_ice_{category}_{feature}")


def plot_oat_sensitivity(
    model: object,
    base_instance: pd.DataFrame,
    top_features: List[str],
    category: str,
    output_dir: Path,
    n_steps: int,
) -> None:
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    variation = np.linspace(-20.0, 20.0, n_steps)

    for feature in top_features[:3]:
        if feature not in base_instance.columns:
            continue
        base_val = float(base_instance[feature].iloc[0])
        preds = []
        for pct in variation:
            sample = base_instance.copy()
            sample[feature] = base_val * (1.0 + pct / 100.0)
            pred = prediction_score(model, sample.values)[0]
            preds.append(pred)
        ax.plot(variation, preds, linewidth=1.2, label=feature)

    ax.axvline(0.0, linestyle="--", color="black", linewidth=0.8)
    ax.set_title(f"OAT Sensitivity: {category}")
    ax.set_xlabel("Feature variation (%)")
    ax.set_ylabel("Prediction score")
    ax.legend(loc="best", frameon=True)
    ax.grid(True, linestyle="--")
    save_figure(fig, output_dir / f"oat_sensitivity_{category}")


def plot_monte_carlo(
    model: object,
    base_instance: pd.DataFrame,
    features: List[str],
    category: str,
    output_dir: Path,
    n_simulations: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    predictions: List[float] = []

    for _ in range(n_simulations):
        sample = base_instance.copy()
        for feature in features:
            if feature not in sample.columns:
                continue
            sample[feature] = float(sample[feature].iloc[0]) * (1.0 + rng.normal(0.0, 0.05))
        pred = prediction_score(model, sample.values)[0]
        predictions.append(float(pred))

    pred_arr = np.asarray(predictions, dtype=float)
    mu = float(np.mean(pred_arr))
    sigma = float(np.std(pred_arr))
    ci_low = float(np.percentile(pred_arr, 2.5))
    ci_high = float(np.percentile(pred_arr, 97.5))

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    ax.hist(pred_arr, bins=35, density=True, color="#6699CC", alpha=0.75, edgecolor="white")
    x = np.linspace(np.min(pred_arr), np.max(pred_arr), 200)
    if sigma > 0:
        ax.plot(x, stats.norm.pdf(x, mu, sigma), color="#BB5566", linewidth=1.2)
    ax.axvline(mu, color="black", linestyle="--", linewidth=0.9, label=f"mean={mu:.3f}")
    ax.axvline(ci_low, color="#BB5566", linestyle=":", linewidth=0.9)
    ax.axvline(ci_high, color="#BB5566", linestyle=":", linewidth=0.9, label="95% CI")
    ax.set_title(f"Monte Carlo Uncertainty: {category}")
    ax.set_xlabel("Prediction score")
    ax.set_ylabel("Density")
    ax.legend(loc="best", frameon=True)
    ax.grid(True, linestyle="--")
    save_figure(fig, output_dir / f"monte_carlo_{category}")

    return pred_arr


def plot_global_sensitivity(
    model: object,
    X_scaled_df: pd.DataFrame,
    category: str,
    output_dir: Path,
) -> List[str]:
    preds = prediction_score(model, X_scaled_df.values)
    rows: List[Dict[str, float | str]] = []

    for feature in X_scaled_df.columns:
        x = X_scaled_df[feature].to_numpy()
        if np.std(x) <= 1e-12:
            continue
        try:
            pcc = abs(float(stats.pearsonr(x, preds).statistic))
        except Exception:
            pcc = 0.0
        try:
            scc = abs(float(stats.spearmanr(x, preds).statistic))
        except Exception:
            scc = 0.0
        rows.append(
            {
                "Feature": feature,
                "Pearson": pcc,
                "Spearman": scc,
                "MeanSensitivity": (pcc + scc) / 2.0,
            }
        )

    if not rows:
        return []

    df = pd.DataFrame(rows).sort_values("MeanSensitivity", ascending=False)
    top = df.head(10)

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    ax.barh(top["Feature"][::-1], top["MeanSensitivity"][::-1], color="#117733", alpha=0.9)
    ax.set_title(f"Global Sensitivity: {category}")
    ax.set_xlabel("Mean(|Pearson|, |Spearman|)")
    ax.set_ylabel("Feature")
    ax.grid(axis="x", linestyle="--")
    save_figure(fig, output_dir / f"global_sensitivity_{category}")

    return df["Feature"].tolist()


def plot_interaction_heatmap(
    model: object,
    base_instance: pd.DataFrame,
    feature1: str,
    feature2: str,
    category: str,
    output_dir: Path,
) -> None:
    if feature1 not in base_instance.columns or feature2 not in base_instance.columns:
        return

    base1 = float(base_instance[feature1].iloc[0])
    base2 = float(base_instance[feature2].iloc[0])
    vals1 = np.linspace(base1 * 0.8, base1 * 1.2, 25)
    vals2 = np.linspace(base2 * 0.8, base2 * 1.2, 25)

    grid = np.zeros((len(vals2), len(vals1)))
    for i, v2 in enumerate(vals2):
        for j, v1 in enumerate(vals1):
            sample = base_instance.copy()
            sample[feature1] = v1
            sample[feature2] = v2
            grid[i, j] = prediction_score(model, sample.values)[0]

    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    sns.heatmap(
        grid,
        ax=ax,
        cmap="cividis",
        cbar_kws={"label": "Prediction score"},
    )
    ax.set_title(f"Interaction: {feature1} x {feature2} ({category})")
    ax.set_xlabel(feature1)
    ax.set_ylabel(feature2)
    save_figure(fig, output_dir / f"interaction_heatmap_{category}_{feature1}_{feature2}")


def write_manifest(output_dir: Path, generated: List[Dict[str, str]]) -> None:
    manifest = pd.DataFrame(generated)
    manifest.to_csv(output_dir / "ieee_explainability_manifest.csv", index=False)


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    apply_ieee_style()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata, scaler, models, feature_names = load_artifacts(Path(args.model_dir), args.suffix)
    if not feature_names:
        raise ValueError("Feature names missing. Expected feature_names file or metadata feature_names.")

    trainer = InterpretationModelTrainer()
    df = trainer.load_dataset(args.dataset)
    X, y_dict = trainer.prepare_features_and_labels(df)
    X = X[feature_names].copy()

    X_scaled = pd.DataFrame(scaler.transform(X), columns=feature_names, index=X.index)
    generated: List[Dict[str, str]] = []

    for category in metadata["categories"]:
        model = models.get(category)
        if model is None:
            print(f"Skipping {category}: model artifact missing")
            continue

        y = y_dict[category]
        mask = y != "Unknown"
        if int(mask.sum()) < 20:
            print(f"Skipping {category}: insufficient labeled samples")
            continue

        X_cat = X_scaled.loc[mask].copy()
        if len(X_cat) > args.max_samples:
            X_cat = X_cat.sample(args.max_samples, random_state=args.seed)

        base_instance = X_cat.median(axis=0).to_frame().T

        # SHAP plots
        top_shap_features: List[str] = []
        if SHAP_AVAILABLE:
            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_cat)
                shap_2d, class_idx = normalize_shap_values(shap_values, model, X_cat.values)
                top_shap_features = plot_shap_global_bar(shap_2d, feature_names, category, out_dir)
                generated.append({"category": category, "plot": "shap_global_bar", "status": "ok"})

                plot_shap_waterfall(explainer, shap_2d, X_cat, class_idx, category, out_dir)
                generated.append({"category": category, "plot": "shap_waterfall", "status": "ok"})

                if top_shap_features:
                    plot_shap_dependence(shap_2d, X_cat, top_shap_features[0], category, out_dir)
                    generated.append({"category": category, "plot": "shap_dependence", "status": "ok"})
            except Exception as exc:
                generated.append({"category": category, "plot": "shap", "status": f"failed: {exc}"})
                print(f"SHAP failed for {category}: {exc}")
        else:
            generated.append({"category": category, "plot": "shap", "status": "skipped: shap unavailable"})

        # PDP + ICE
        top_importance_features = select_top_features(model, feature_names, top_n=3)
        if top_importance_features:
            try:
                plot_pdp_ice(model, X_cat, top_importance_features[0], category, out_dir)
                generated.append({"category": category, "plot": "pdp_ice", "status": "ok"})
            except Exception as exc:
                generated.append({"category": category, "plot": "pdp_ice", "status": f"failed: {exc}"})

        # OAT sensitivity
        try:
            plot_oat_sensitivity(model, base_instance, top_importance_features, category, out_dir, args.oat_steps)
            generated.append({"category": category, "plot": "oat_sensitivity", "status": "ok"})
        except Exception as exc:
            generated.append({"category": category, "plot": "oat_sensitivity", "status": f"failed: {exc}"})

        # Monte Carlo
        try:
            plot_monte_carlo(
                model,
                base_instance,
                top_importance_features,
                category,
                out_dir,
                args.mc_simulations,
                args.seed,
            )
            generated.append({"category": category, "plot": "monte_carlo", "status": "ok"})
        except Exception as exc:
            generated.append({"category": category, "plot": "monte_carlo", "status": f"failed: {exc}"})

        # Global sensitivity
        global_top: List[str] = []
        try:
            global_top = plot_global_sensitivity(model, X_cat, category, out_dir)
            generated.append({"category": category, "plot": "global_sensitivity", "status": "ok"})
        except Exception as exc:
            generated.append({"category": category, "plot": "global_sensitivity", "status": f"failed: {exc}"})

        # Interaction heatmap
        interaction_features = (global_top[:2] if len(global_top) >= 2 else top_importance_features[:2])
        if len(interaction_features) >= 2:
            try:
                plot_interaction_heatmap(
                    model,
                    base_instance,
                    interaction_features[0],
                    interaction_features[1],
                    category,
                    out_dir,
                )
                generated.append({"category": category, "plot": "interaction_heatmap", "status": "ok"})
            except Exception as exc:
                generated.append({"category": category, "plot": "interaction_heatmap", "status": f"failed: {exc}"})

    write_manifest(out_dir, generated)

    print("\nIEEE explainability generation complete.")
    print(f"Output directory: {out_dir}")
    print("Manifest:", out_dir / "ieee_explainability_manifest.csv")


if __name__ == "__main__":
    main()
