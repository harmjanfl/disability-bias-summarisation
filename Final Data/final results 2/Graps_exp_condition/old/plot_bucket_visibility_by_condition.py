#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Plot bucket-level visibility curves (DIS vs GEN) for each model,
with experiment_condition on the x-axis (expressed as % of input),
with 95% Wilson confidence intervals. Fixed 2x3 panel layout:
Llama/cluster on top, GPT on bottom.

Reads:  Final Data/final results 2/combined/combined_iteration_topic.xlsx
Writes: Final Data/final results 2/combined/plot_visibility_baseline_by_condition.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Config
# -----------------------------
INPUT_FILE = "Final Data/final results 2/combined/combined_iteration_topic.xlsx"
OUTPUT_FILE = "Final Data/final results 2/combined/graphs by condition/plot_visibility_baseline_by_condition.png"
INPUT_TOTAL = 50

# Fixed 2x3 panel layout: Llama/cluster on top, GPT on bottom.
MODEL_GRID = [
    [
        ("baseline_cluster", "Baseline Llama 3.1 8B"),
        ("grr_cluster", "GRR Llama 3.1 8B"),
        ("grtx_cluster", "GRTx Llama 3.1 8B"),
    ],
    [
        ("baseline_gpt", "Baseline GPT 4.1"),
        ("grr_gpt", "GRR GPT 4.1"),
        ("grtx_gpt", "GRTx GPT 4.1"),
    ],
]


def wilson_halfwidth(p: float, n: int, z: float = 1.96) -> float:
    """Half-width of the Wilson 95% confidence interval for a proportion."""
    if n == 0 or pd.isna(p):
        return np.nan
    denom = 1 + z**2 / n
    return (z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)) / denom


def aggregate_by_condition(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (model, condition, bucket): mean V_t and observation count."""
    sub = df[df["present_in_input"] == 1].copy()
    grouped = (
        sub.groupby(["model", "experiment_condition", "bucket"])
        .agg(V_hat=("V_t", "mean"), n=("V_t", "size"))
        .reset_index()
    )
    return grouped


def pivot_wide(agg: pd.DataFrame) -> pd.DataFrame:
    """Pivot so each row is (model, experiment_condition) with DIS/GEN columns."""
    wide = agg.pivot_table(
        index=["model", "experiment_condition"],
        columns="bucket",
        values=["V_hat", "n"],
    ).reset_index()
    wide.columns = [
        f"{a}_{b}" if b else a
        for a, b in wide.columns.to_flat_index()
    ]
    return wide


def main():
    df = pd.read_excel(INPUT_FILE)
    agg = aggregate_by_condition(df)
    wide = pivot_wide(agg)

    wide["ci_DIS"] = wide.apply(
        lambda r: wilson_halfwidth(r.get("V_hat_DIS"), r.get("n_DIS", 0)), axis=1)
    wide["ci_GEN"] = wide.apply(
        lambda r: wilson_halfwidth(r.get("V_hat_GEN"), r.get("n_GEN", 0)), axis=1)

    conditions = sorted(wide["experiment_condition"].dropna().unique())
    if not conditions:
        print("No experiment conditions found.")
        return

    fig, axes = plt.subplots(2, 3, figsize=(11, 7), sharey=True)
    legend_handles = None
    legend_labels = None

    # use evenly-spaced index positions on the x-axis (conditions
    # are not evenly spaced in absolute terms, so plotting them on a
    # linear axis crams the low values together)
    x_positions = list(range(len(conditions)))
    tick_labels = [f"{round(c / INPUT_TOTAL * 100):.0f}%" for c in conditions]

    for row_idx, row in enumerate(MODEL_GRID):
        for col_idx, (model, label) in enumerate(row):
            ax = axes[row_idx, col_idx]
            m = wide[wide["model"] == model].set_index("experiment_condition").reindex(conditions)

            if m["V_hat_DIS"].isna().all() and m["V_hat_GEN"].isna().all():
                ax.set_title(f"{label}\n(no data yet)", fontsize=10)
                ax.set_xticks(x_positions)
                ax.set_xticklabels(tick_labels)
                ax.set_ylim(0, 1)
                ax.grid(True, alpha=0.3)
                continue

            ax.errorbar(x_positions, m["V_hat_DIS"], yerr=m["ci_DIS"],
                        marker="o", capsize=3, label="DIS",
                        color="#c0392b", linewidth=1.5)
            ax.errorbar(x_positions, m["V_hat_GEN"], yerr=m["ci_GEN"],
                        marker="s", capsize=3, label="GEN",
                        color="#2c3e50", linewidth=1.5)
            if legend_handles is None:
                legend_handles, legend_labels = ax.get_legend_handles_labels()

            ax.set_title(label, fontsize=10)
            ax.set_xticks(x_positions)
            ax.set_xticklabels(tick_labels)
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)

    for ax in axes[1, :]:
        ax.set_xlabel("Percentage of input that is DIS")

    axes[0, 0].set_ylabel(r"$\hat{V}$  (visibility rate)")
    axes[1, 0].set_ylabel(r"$\hat{V}$  (visibility rate)")
    if legend_handles is not None:
        fig.legend(legend_handles, legend_labels, loc="upper right", frameon=True)
    fig.suptitle(
        "Topic visibility by percentage of DIS input: DIS vs GEN buckets, 95% Wilson CI",
        fontsize=11, y=1.02,
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    plt.savefig(OUTPUT_FILE, dpi=200, bbox_inches="tight")
    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
