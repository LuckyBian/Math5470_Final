#!/usr/bin/env python3
"""Generate ablation visualizations for the Math5470 final project."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib import ticker

BASELINE_LABEL = "Full Model"
BASELINE_AUC = 0.791


def build_ablation_frame(csv_path: Path) -> pd.DataFrame:
    """Read ablation results, inject the baseline row, and compute drop %."""
    df = pd.read_csv(csv_path)
    rename_map = {
        "No_Ext_Source": "No Ext Source",
        "No_Aggregations": "No Aggregations",
    }
    df["Experiment"] = df["Experiment"].replace(rename_map)

    baseline = pd.DataFrame({"Experiment": [BASELINE_LABEL], "AUC": [BASELINE_AUC]})
    df = pd.concat([baseline, df], ignore_index=True)
    df["PerformanceDropPct"] = (df["AUC"] - BASELINE_AUC) / BASELINE_AUC * 100
    return df


def plot_auc_comparison(df: pd.DataFrame, output_path: Path) -> None:
    """Create the AUC bar chart comparing ablation variants."""
    data = df.sort_values("AUC", ascending=False)
    colors = [
        "#d84b36" if exp == BASELINE_LABEL else "#c6ccd8" for exp in data["Experiment"]
    ]

    fig, ax = plt.subplots(figsize=(9, 5.2))
    sns.barplot(
        data=data,
        x="Experiment",
        y="AUC",
        palette=colors,
        edgecolor="none",
        ax=ax,
    )
    ax.set_ylim(0.75, 0.80)
    ax.set_xlabel("Experiment")
    ax.set_ylabel("AUC")
    ax.set_title("AUC Comparison Across Ablations", pad=18, weight="bold")
    sns.despine(ax=ax)
    # Ensure horizontal labels
    ax.tick_params(axis="x", rotation=0)

    for patch, (experiment, auc) in zip(ax.patches, data[["Experiment", "AUC"]].itertuples(index=False, name=None)):
        ax.annotate(
            f"{auc:.3f}",
            (patch.get_x() + patch.get_width() / 2, auc + 0.0005),
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold" if experiment == BASELINE_LABEL else "normal",
        )

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_drop_analysis(df: pd.DataFrame, output_path: Path) -> None:
    """Plot the percent performance drop when features are removed."""
    drop_df = df[df["Experiment"] != BASELINE_LABEL].copy()
    drop_df = drop_df.sort_values("PerformanceDropPct")
    palette = sns.color_palette("Blues_r", drop_df.shape[0])

    fig, ax = plt.subplots(figsize=(8.6, 4.5))
    bars = ax.barh(
        drop_df["Experiment"],
        drop_df["PerformanceDropPct"],
        color=palette,
        edgecolor="none",
        height=0.6,
    )
    ax.set_xlabel("Performance change (%) vs Full Model")
    ax.set_ylabel("Removed feature block")
    # Use suptitle or just ensure title is centered on the plot area, which is standard.
    # Increasing left margin via xlim helps separate y-labels from bars.
    ax.set_title("Performance Drop when Removing Features", pad=18, weight="bold")
    ax.axvline(0, color="#2f3542", linestyle="--", linewidth=0.9, alpha=0.6)
    
    # Expand x-limit to prevent overlap between bar labels and y-axis text
    min_drop = drop_df["PerformanceDropPct"].min()
    # Make the left limit roughly 30% wider than the smallest value to give space
    ax.set_xlim(min_drop * 1.35, 0.2)
    
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda val, _: f"{val:.1f}%")
    )
    ax.grid(axis="x", linestyle=":", linewidth=0.8, alpha=0.7)
    sns.despine(ax=ax, left=True, bottom=False)

    for bar in bars:
        width = bar.get_width()
        y_center = bar.get_y() + bar.get_height() / 2
        ha = "right" if width < 0 else "left"
        offset = -0.05 if width < 0 else 0.05
        ax.annotate(
            f"{width:.2f}%",
            (width + offset, y_center),
            ha=ha,
            va="center",
            fontsize=11,
            color="#1f2a44",
            fontweight="bold",
        )

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_prediction_distribution(csv_path: Path, output_path: Path) -> None:
    """Plot the submission TARGET prediction distribution."""
    predictions = pd.read_csv(csv_path, usecols=["TARGET"])

    fig, ax = plt.subplots(figsize=(8.8, 5.5))
    sns.histplot(
        predictions["TARGET"],
        bins=60,
        color="#5dade2",
        edgecolor="white",
        linewidth=0.3,
        alpha=0.9,
        ax=ax,
    )
    sns.kdeplot(
        predictions["TARGET"],
        color="#0b5394",
        linewidth=2.0,
        ax=ax,
    )
    ax.set_xlabel("Predicted Default Probability")
    ax.set_ylabel("Frequency (log scale)")
    ax.set_title("Full Model Prediction Distribution", pad=18, weight="bold")
    ax.set_xlim(0, 0.5)
    ax.set_yscale("log")
    ax.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.7)
    sns.despine(ax=ax)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    sns.set_theme(
        style="whitegrid",
        context="talk",
        rc={
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Liberation Serif", "Times New Roman", "serif"],
            "axes.titlesize": 15,
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
        },
    )

    base_dir = Path(__file__).resolve().parent
    ablation_csv = base_dir / "ablation_results.csv"
    submission_csv = base_dir / "submission_advanced.csv"
    
    plots_dir = base_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    ablation_df = build_ablation_frame(ablation_csv)
    plot_auc_comparison(ablation_df, plots_dir / "1_auc_comparison.png")
    plot_drop_analysis(ablation_df, plots_dir / "2_drop_analysis.png")
    plot_prediction_distribution(submission_csv, plots_dir / "3_risk_distribution.png")


if __name__ == "__main__":
    main()

