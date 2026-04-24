#!/usr/bin/env python3
"""
Generate evaluation artifacts for adaptive hybrid routing (ML/Rule/Abstain).

Outputs:
- hybrid_routing_per_category.csv
- hybrid_routing_overall.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from train_interpretation_model import InterpretationModelTrainer
from src.predictor import ClinicalPredictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate adaptive hybrid routing with abstention")
    parser.add_argument("--dataset", default="data/processed_new/training_dataset_new.json")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--output-dir", default="outputs/model_selection_report_combined")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ml-threshold", type=float, default=0.70)
    parser.add_argument("--abstain-threshold", type=float, default=0.45)
    parser.add_argument("--disagreement-margin", type=float, default=0.10)
    parser.add_argument("--sweep", action="store_true", help="Run threshold sweep and generate risk-coverage plot")
    parser.add_argument(
        "--ml-threshold-grid",
        default="0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90",
        help="Comma-separated ml-threshold values for sweep",
    )
    parser.add_argument(
        "--abstain-threshold-grid",
        default="0.35,0.40,0.45,0.50,0.55,0.60,0.65",
        help="Comma-separated abstain-threshold values for sweep",
    )
    parser.add_argument(
        "--disagreement-margin-grid",
        default="0.05,0.10,0.15,0.20",
        help="Comma-separated disagreement-margin values for sweep",
    )
    return parser.parse_args()


def ensure_dir(path: str) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def compute_model_confidence(model: Any, x_scaled_row: np.ndarray) -> float:
    if hasattr(model, "predict_proba"):
        try:
            p = model.predict_proba(x_scaled_row)[0]
            return float(np.max(p))
        except Exception:
            return 0.0
    return 0.0


def route_decision(
    ml_conf: float,
    disagree: bool,
    has_required_features: bool,
    ml_threshold: float,
    abstain_threshold: float,
    disagreement_margin: float,
) -> str:
    if not has_required_features:
        return "Rule"

    high_conf = ml_conf >= ml_threshold
    conflict_low_conf = disagree and ml_conf < (ml_threshold + disagreement_margin)

    if high_conf and not conflict_low_conf:
        return "ML"

    if disagree and ml_conf < abstain_threshold:
        return "Abstain"

    return "Rule"


def has_required_features(category: str, measurements: Dict[str, float]) -> bool:
    required = {
        "LV_FUNCTION": ["EF", "FS"],
        "LV_SIZE": ["LVID_D"],
        "LV_HYPERTROPHY": ["IVS_D"],
        "LA_SIZE": ["LA_DIMENSION"],
        "DIASTOLIC_FUNCTION": ["MV_E_A"],
    }
    feats = required.get(category, [])
    if not feats:
        return True
    return any((measurements.get(f, 0) or 0) > 0 for f in feats)


def parse_grid(values: str) -> List[float]:
    out: List[float] = []
    for tok in values.split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(float(tok))
    if not out:
        raise ValueError("Threshold grid cannot be empty")
    return out


def build_eval_records(
    trainer: InterpretationModelTrainer,
    predictor: ClinicalPredictor,
    X_test_scaled: np.ndarray,
    test_indices: List[int],
    raw_items: List[Dict[str, Any]],
    y_dict: Dict[str, pd.Series],
) -> Dict[str, List[Dict[str, Any]]]:
    records_by_category: Dict[str, List[Dict[str, Any]]] = {}

    for category in trainer.categories:
        model = predictor.models.get(category)
        if model is None:
            continue

        y_test = y_dict[category].iloc[test_indices]
        category_records: List[Dict[str, Any]] = []

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

            x_row_scaled = X_test_scaled[pos : pos + 1]
            ml_pred = str(model.predict(x_row_scaled)[0])
            ml_conf = compute_model_confidence(model, x_row_scaled)
            required_ok = has_required_features(category, measurements)

            category_records.append(
                {
                    "y_true": y_true,
                    "rule_pred": rule_pred,
                    "ml_pred": ml_pred,
                    "ml_conf": float(ml_conf),
                    "required_ok": bool(required_ok),
                }
            )

        if category_records:
            records_by_category[category] = category_records

    return records_by_category


def evaluate_records(
    records_by_category: Dict[str, List[Dict[str, Any]]],
    ml_threshold: float,
    abstain_threshold: float,
    disagreement_margin: float,
) -> tuple[pd.DataFrame, Dict[str, float]]:
    rows: List[Dict[str, Any]] = []

    for category, records in records_by_category.items():
        y_true_list: List[str] = []
        y_pred_list: List[str] = []
        source_list: List[str] = []

        for rec in records:
            disagree = rec["rule_pred"] != rec["ml_pred"]
            source = route_decision(
                ml_conf=float(rec["ml_conf"]),
                disagree=disagree,
                has_required_features=bool(rec["required_ok"]),
                ml_threshold=ml_threshold,
                abstain_threshold=abstain_threshold,
                disagreement_margin=disagreement_margin,
            )

            if source == "ML":
                y_pred = rec["ml_pred"]
            elif source == "Rule":
                y_pred = rec["rule_pred"]
            else:
                y_pred = "ABSTAIN"

            y_true_list.append(rec["y_true"])
            y_pred_list.append(y_pred)
            source_list.append(source)

        n_total = len(y_true_list)
        n_abstain = int(sum(1 for s in source_list if s == "Abstain"))
        n_covered = n_total - n_abstain

        covered_true = [t for t, p in zip(y_true_list, y_pred_list) if p != "ABSTAIN"]
        covered_pred = [p for p in y_pred_list if p != "ABSTAIN"]

        coverage = float(n_covered / n_total) if n_total else 0.0
        abstain_rate = float(n_abstain / n_total) if n_total else 0.0

        if n_covered > 0:
            covered_acc = float(accuracy_score(covered_true, covered_pred))
            covered_f1 = float(f1_score(covered_true, covered_pred, average="macro", zero_division=0))
            risk_at_coverage = float(1.0 - covered_acc)
        else:
            covered_acc = 0.0
            covered_f1 = 0.0
            risk_at_coverage = 1.0

        strict_acc = float(
            np.mean([1.0 if yt == yp else 0.0 for yt, yp in zip(y_true_list, y_pred_list)])
        ) if n_total else 0.0

        rows.append(
            {
                "category": category,
                "samples": n_total,
                "coverage": coverage,
                "abstain_rate": abstain_rate,
                "accuracy_covered": covered_acc,
                "f1_macro_covered": covered_f1,
                "risk_at_coverage": risk_at_coverage,
                "strict_accuracy_with_abstain_as_error": strict_acc,
                "ml_selected_rate": float(sum(1 for s in source_list if s == "ML") / n_total) if n_total else 0.0,
                "rule_selected_rate": float(sum(1 for s in source_list if s == "Rule") / n_total) if n_total else 0.0,
            }
        )

    if not rows:
        raise RuntimeError("No valid categories produced results")

    per_cat = pd.DataFrame(rows)
    weights = per_cat["samples"].to_numpy(dtype=float)
    weights = weights / weights.sum()

    overall = {
        "ml_threshold": float(ml_threshold),
        "abstain_threshold": float(abstain_threshold),
        "disagreement_margin": float(disagreement_margin),
        "weighted_coverage": float(np.sum(per_cat["coverage"].to_numpy() * weights)),
        "weighted_abstain_rate": float(np.sum(per_cat["abstain_rate"].to_numpy() * weights)),
        "weighted_accuracy_covered": float(np.sum(per_cat["accuracy_covered"].to_numpy() * weights)),
        "weighted_f1_macro_covered": float(np.sum(per_cat["f1_macro_covered"].to_numpy() * weights)),
        "weighted_risk_at_coverage": float(np.sum(per_cat["risk_at_coverage"].to_numpy() * weights)),
        "weighted_strict_accuracy": float(np.sum(per_cat["strict_accuracy_with_abstain_as_error"].to_numpy() * weights)),
    }

    return per_cat, overall


def select_best_threshold(sweep_df: pd.DataFrame) -> Dict[str, Any]:
    # Primary objective: weighted f1 on covered predictions.
    # Tie-breakers: lower risk, higher coverage.
    ranked = sweep_df.sort_values(
        ["weighted_f1_macro_covered", "weighted_risk_at_coverage", "weighted_coverage"],
        ascending=[False, True, False],
    ).reset_index(drop=True)
    best = ranked.iloc[0]
    return {
        "ml_threshold": float(best["ml_threshold"]),
        "abstain_threshold": float(best["abstain_threshold"]),
        "disagreement_margin": float(best["disagreement_margin"]),
        "weighted_coverage": float(best["weighted_coverage"]),
        "weighted_abstain_rate": float(best["weighted_abstain_rate"]),
        "weighted_accuracy_covered": float(best["weighted_accuracy_covered"]),
        "weighted_f1_macro_covered": float(best["weighted_f1_macro_covered"]),
        "weighted_risk_at_coverage": float(best["weighted_risk_at_coverage"]),
        "weighted_strict_accuracy": float(best["weighted_strict_accuracy"]),
    }


def plot_risk_coverage_curve(sweep_df: pd.DataFrame, best: Dict[str, Any], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 6))

    x = sweep_df["weighted_coverage"].to_numpy()
    y = sweep_df["weighted_risk_at_coverage"].to_numpy()
    c = sweep_df["weighted_f1_macro_covered"].to_numpy()

    sc = ax.scatter(x, y, c=c, cmap="viridis", alpha=0.85, edgecolor="black", linewidth=0.3)

    ax.scatter(
        [best["weighted_coverage"]],
        [best["weighted_risk_at_coverage"]],
        marker="*",
        s=260,
        color="red",
        edgecolor="black",
        linewidth=0.8,
        label="Best threshold set",
        zorder=5,
    )

    ax.set_xlabel("Coverage")
    ax.set_ylabel("Risk at coverage (1 - covered accuracy)")
    ax.set_title("Adaptive Hybrid Routing: Risk-Coverage Sweep")
    ax.grid(True, linestyle="--", alpha=0.35)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Weighted F1 (covered)")
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def evaluate(args: argparse.Namespace) -> None:
    trainer = InterpretationModelTrainer()
    df = trainer.load_dataset(args.dataset)
    X, y_dict = trainer.prepare_features_and_labels(df)

    with open(args.dataset, "r") as f:
        raw_items = json.load(f)

    predictor = ClinicalPredictor(
        model_dir=args.model_dir,
        use_ml=True,
        routing_mode="adaptive",
        ml_confidence_threshold=args.ml_threshold,
        abstain_confidence_threshold=args.abstain_threshold,
        disagreement_margin=args.disagreement_margin,
        enable_abstain=True,
    )

    if not predictor.ml_available:
        raise RuntimeError("ML models are not available; train models before running this report")

    _, X_test, _, test_indices = train_test_split(
        X, X.index, test_size=args.test_size, random_state=args.seed
    )

    X_test_scaled = predictor.scaler.transform(X_test)
    test_index_list = list(test_indices)

    records_by_category = build_eval_records(
        trainer=trainer,
        predictor=predictor,
        X_test_scaled=X_test_scaled,
        test_indices=test_index_list,
        raw_items=raw_items,
        y_dict=y_dict,
    )

    out_dir = ensure_dir(args.output_dir)
    per_cat, overall = evaluate_records(
        records_by_category=records_by_category,
        ml_threshold=args.ml_threshold,
        abstain_threshold=args.abstain_threshold,
        disagreement_margin=args.disagreement_margin,
    )

    per_cat.to_csv(out_dir / "hybrid_routing_per_category.csv", index=False)

    with open(out_dir / "hybrid_routing_overall.json", "w") as f:
        json.dump(overall, f, indent=2)

    if args.sweep:
        ml_grid = parse_grid(args.ml_threshold_grid)
        abstain_grid = parse_grid(args.abstain_threshold_grid)
        margin_grid = parse_grid(args.disagreement_margin_grid)

        sweep_rows: List[Dict[str, Any]] = []
        for ml_t in ml_grid:
            for abstain_t in abstain_grid:
                for margin_t in margin_grid:
                    _, sweep_overall = evaluate_records(
                        records_by_category=records_by_category,
                        ml_threshold=ml_t,
                        abstain_threshold=abstain_t,
                        disagreement_margin=margin_t,
                    )
                    sweep_rows.append(sweep_overall)

        sweep_df = pd.DataFrame(sweep_rows).sort_values(
            ["weighted_coverage", "weighted_risk_at_coverage"], ascending=[True, True]
        )
        sweep_csv = out_dir / "hybrid_routing_threshold_sweep.csv"
        sweep_df.to_csv(sweep_csv, index=False)

        best = select_best_threshold(sweep_df)
        best_json = out_dir / "hybrid_routing_best_thresholds.json"
        with open(best_json, "w") as f:
            json.dump(best, f, indent=2)

        plot_path = out_dir / "hybrid_routing_risk_coverage_curve.png"
        plot_risk_coverage_curve(sweep_df, best, plot_path)

    print("Generated adaptive hybrid routing report:")
    print(f"  - {out_dir / 'hybrid_routing_per_category.csv'}")
    print(f"  - {out_dir / 'hybrid_routing_overall.json'}")
    if args.sweep:
        print(f"  - {out_dir / 'hybrid_routing_threshold_sweep.csv'}")
        print(f"  - {out_dir / 'hybrid_routing_best_thresholds.json'}")
        print(f"  - {out_dir / 'hybrid_routing_risk_coverage_curve.png'}")


if __name__ == "__main__":
    evaluate(parse_args())
