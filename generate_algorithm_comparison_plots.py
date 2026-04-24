#!/usr/bin/env python3
"""
Generate IEEE-ready comparison plots for multiple ML algorithms.

Outputs:
- algorithm_comparison_detailed.csv
- algorithm_comparison_summary.csv
- figure_avg_metrics_grouped.png
- figure_accuracy_heatmap.png
- figure_accuracy_boxplot.png
- figure_efficiency_scatter.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from compare_algorithms import AlgorithmComparator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate publication-quality ML algorithm comparison plots"
    )
    parser.add_argument(
        "--dataset",
        default="data/processed_new/training_dataset_new.json",
        help="Path to training dataset JSON",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/ieee_algorithm_comparison",
        help="Directory to store generated figures and CSV tables",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=["LV_HYPERTROPHY", "LA_SIZE", "DIASTOLIC_FUNCTION"],
        help="Target categories to compare",
    )
    return parser.parse_args()


def ensure_dir(path: str) -> Path:
    output_path = Path(path)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def build_results_dataframe(results: Dict[str, Dict[str, Dict[str, float]]]) -> pd.DataFrame:
    rows: List[Dict[str, float | str]] = []

    for category, algo_results in results.items():
        for algorithm, metrics in algo_results.items():
            if metrics is None:
                continue
            rows.append(
                {
                    "category": category,
                    "algorithm": algorithm,
                    "accuracy": float(metrics["accuracy"]),
                    "precision": float(metrics["precision"]),
                    "recall": float(metrics["recall"]),
                    "f1_score": float(metrics["f1_score"]),
                    "cv_mean": float(metrics["cv_mean"]),
                    "cv_std": float(metrics["cv_std"]),
                    "train_time_s": float(metrics["train_time"]),
                    "predict_time_s": float(metrics["predict_time"]),
                }
            )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    return df.sort_values(["algorithm", "category"]).reset_index(drop=True)


def build_summary_dataframe(detailed_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        detailed_df.groupby("algorithm", as_index=False)
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            f1_mean=("f1_score", "mean"),
            f1_std=("f1_score", "std"),
            precision_mean=("precision", "mean"),
            recall_mean=("recall", "mean"),
            cv_mean=("cv_mean", "mean"),
            cv_std_mean=("cv_std", "mean"),
            train_time_mean_s=("train_time_s", "mean"),
            predict_time_mean_s=("predict_time_s", "mean"),
        )
        .sort_values("accuracy_mean", ascending=False)
        .reset_index(drop=True)
    )

    for col in ["accuracy_std", "f1_std"]:
        summary[col] = summary[col].fillna(0.0)

    return summary


def configure_ieee_plot_style() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "Times"],
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )


def plot_avg_metrics_grouped(summary_df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 4.6))

    x = np.arange(len(summary_df))
    width = 0.16

    ax.bar(x - 1.5 * width, summary_df["accuracy_mean"], width, label="Accuracy")
    ax.bar(x - 0.5 * width, summary_df["precision_mean"], width, label="Precision")
    ax.bar(x + 0.5 * width, summary_df["recall_mean"], width, label="Recall")
    ax.bar(x + 1.5 * width, summary_df["f1_mean"], width, label="F1-score")

    ax.errorbar(
        x - 1.5 * width,
        summary_df["accuracy_mean"],
        yerr=summary_df["accuracy_std"],
        fmt="none",
        ecolor="black",
        elinewidth=0.8,
        capsize=2,
        alpha=0.8,
    )

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_xlabel("Algorithm")
    ax.set_title("Average Predictive Performance Across Categories")
    ax.set_xticks(x)
    ax.set_xticklabels(summary_df["algorithm"], rotation=25, ha="right")
    ax.legend(ncols=4, frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "figure_avg_metrics_grouped.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy_heatmap(detailed_df: pd.DataFrame, output_dir: Path) -> None:
    pivot = detailed_df.pivot(index="algorithm", columns="category", values="accuracy")
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        vmin=0.0,
        vmax=1.0,
        cbar_kws={"label": "Accuracy"},
        ax=ax,
    )
    ax.set_title("Category-wise Accuracy by Algorithm")
    ax.set_xlabel("Category")
    ax.set_ylabel("Algorithm")

    fig.tight_layout()
    fig.savefig(output_dir / "figure_accuracy_heatmap.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy_boxplot(detailed_df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.2, 4.6))
    sns.boxplot(data=detailed_df, x="algorithm", y="accuracy", ax=ax, color="#d5e3f0")
    sns.stripplot(
        data=detailed_df,
        x="algorithm",
        y="accuracy",
        hue="category",
        size=4,
        alpha=0.8,
        dodge=False,
        ax=ax,
    )

    ax.set_ylim(0, 1.05)
    ax.set_title("Accuracy Distribution Across Categories")
    ax.set_xlabel("Algorithm")
    ax.set_ylabel("Accuracy")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles[: len(detailed_df["category"].unique())], labels[: len(detailed_df["category"].unique())], title="Category", loc="lower right")

    fig.tight_layout()
    fig.savefig(output_dir / "figure_accuracy_boxplot.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_efficiency_scatter(summary_df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 5.0))

    sns.scatterplot(
        data=summary_df,
        x="train_time_mean_s",
        y="accuracy_mean",
        size="f1_mean",
        sizes=(60, 260),
        legend=False,
        ax=ax,
    )

    for _, row in summary_df.iterrows():
        ax.annotate(
            row["algorithm"],
            (row["train_time_mean_s"], row["accuracy_mean"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

    ax.set_title("Efficiency Trade-off: Accuracy vs Training Time")
    ax.set_xlabel("Average Training Time (s)")
    ax.set_ylabel("Average Accuracy")
    ax.set_ylim(0, 1.05)
    ax.grid(linestyle="--", alpha=0.35)

    fig.tight_layout()
    fig.savefig(output_dir / "figure_efficiency_scatter.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    configure_ieee_plot_style()

    comparator = AlgorithmComparator()
    results = comparator.compare_algorithms(args.dataset, args.categories)

    detailed_df = build_results_dataframe(results)
    if detailed_df.empty:
        print("No valid results were produced. Check dataset path and label coverage.")
        return

    summary_df = build_summary_dataframe(detailed_df)

    detailed_df.to_csv(output_dir / "algorithm_comparison_detailed.csv", index=False)
    summary_df.to_csv(output_dir / "algorithm_comparison_summary.csv", index=False)

    plot_avg_metrics_grouped(summary_df, output_dir)
    plot_accuracy_heatmap(detailed_df, output_dir)
    plot_accuracy_boxplot(detailed_df, output_dir)
    plot_efficiency_scatter(summary_df, output_dir)

    print("\nSaved IEEE-ready algorithm comparison outputs:")
    for output_file in sorted(output_dir.glob("*")):
        if output_file.is_file():
            print(f"  - {output_file}")


if __name__ == "__main__":
    main()
