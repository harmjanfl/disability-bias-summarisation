#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Plot DIS bucket visibility for Llama 3.1 8B (cluster) models across baseline,
GRR, and GRTx, with experiment_condition on the x-axis (expressed as %
of input), and 95% Wilson confidence intervals. Pairwise z-tests are
saved as CSV.

Reads:  Final Data/final results 2/combined/combined_iteration_topic.xlsx
Writes: Final Data/final results 2/combined/plot_visibility_dis_cluster_bars_by_condition.png
        Final Data/final results 2/combined/plot_visibility_dis_cluster_stats_by_condition.csv
"""

import os
import math
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Config
# -----------------------------
INPUT_FILE = "Final Data/final results 2/combined/combined_iteration_topic.xlsx"
OUTPUT_FILE = "Final Data/final results 2/combined/graphs by condition/plot_visibility_dis_cluster_bars_by_condition.png"
OUTPUT_STATS = "Final Data/final results 2/combined/graphs by condition/plot_visibility_dis_cluster_stats_by_condition.csv"
OUTPUT_TABLE = "Final Data/final results 2/combined/graphs by condition/plot_visibility_dis_cluster_table.xlsx"
INPUT_TOTAL = 50

MODEL_ORDER = [
    ("baseline_cluster", "Baseline"),
    ("grr_cluster",      "GRR"),
    ("grtx_cluster",     "GRTx"),
]
BUCKET = "DIS"


def wilson_halfwidth(p: float, n: int, z: float = 1.96) -> float:
    if n == 0 or pd.isna(p):
        return np.nan
    denom = 1 + z**2 / n
    return (z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)) / denom


def two_prop_z_test(p1, n1, p2, n2):
    if n1 == 0 or n2 == 0 or pd.isna(p1) or pd.isna(p2):
        return math.nan, math.nan
    x1 = int(round(p1 * n1))
    x2 = int(round(p2 * n2))
    p_pool = (x1 + x2) / (n1 + n2)
    denom = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if denom == 0:
        return math.nan, math.nan
    z = (p1 - p2) / denom
    pval = math.erfc(abs(z) / math.sqrt(2))
    return z, pval


def aggregate_by_condition(df: pd.DataFrame, bucket: str) -> pd.DataFrame:
    """
    Aggregate visibility at the iteration (summary) level.

    For each summary iteration:
        visibility_iteration =
            (# visible topics in bucket) /
            (# eligible topics in bucket)

    This produces one observation per generated summary,
    resulting in more stable and interpretable sample sizes.
    """
    sub = df[
        (df["present_in_input"] == 1) &
        (df["bucket"] == bucket)
    ].copy()

    per_iteration = (
        sub.groupby(["model", "experiment_condition", "iteration"])
        .agg(
            visible_topics=("V_t", "sum"),
            total_topics=("V_t", "size")
        )
        .reset_index()
    )

    per_iteration["visibility_iteration"] = (
        per_iteration["visible_topics"] /
        per_iteration["total_topics"]
    )

    grouped = (
        per_iteration.groupby(["model", "experiment_condition"])
        .agg(
            V_hat=("visibility_iteration", "mean"),
            n=("visibility_iteration", "size")
        )
        .reset_index()
    )

    return grouped


def main():
    df = pd.read_excel(INPUT_FILE)
    agg = aggregate_by_condition(df, BUCKET)
    agg["ci"] = agg.apply(lambda r: wilson_halfwidth(r["V_hat"], r["n"]), axis=1)
    agg["ci_lower"] = (agg["V_hat"] - agg["ci"]).clip(lower=0)
    agg["ci_upper"] = (agg["V_hat"] + agg["ci"]).clip(upper=1)

    sub = agg[agg["model"].isin([m for m, _ in MODEL_ORDER])].copy()
    models_present = [(m, lbl) for m, lbl in MODEL_ORDER if m in sub["model"].unique()]
    if not models_present:
        print("No matching models found.")
        return

    conditions = sorted(sub["experiment_condition"].dropna().unique())
    if not conditions:
        print("No experiment conditions found.")
        return

    data_by_model = {m: sub[sub["model"] == m].set_index("experiment_condition")
                     for m, _ in models_present}

    results = []
    model_names = [m for m, _ in models_present]
    for c in conditions:
        for a, b in combinations(model_names, 2):
            ma, mb = data_by_model[a], data_by_model[b]
            if c not in ma.index or c not in mb.index:
                continue
            p1 = ma.at[c, "V_hat"]
            n1 = int(ma.at[c, "n"]) if not pd.isna(ma.at[c, "n"]) else 0
            p2 = mb.at[c, "V_hat"]
            n2 = int(mb.at[c, "n"]) if not pd.isna(mb.at[c, "n"]) else 0
            z, pval = two_prop_z_test(p1, n1, p2, n2)
            sig = False if pd.isna(pval) else (pval < 0.05)
            results.append({"experiment_condition": c, "model_a": a, "model_b": b,
                            "p_value": pval, "z": z, "significant": sig,
                            "p_hat_a": p1, "n_a": n1, "p_hat_b": p2, "n_b": n2})

    if results:
        res_df = pd.DataFrame(results)
        os.makedirs(os.path.dirname(OUTPUT_STATS), exist_ok=True)
        res_df.to_csv(OUTPUT_STATS, index=False)
        print(f"Saved stats: {OUTPUT_STATS}")
        for c in conditions:
            subr = res_df[res_df["experiment_condition"] == c]
            if subr.empty:
                continue
            print(f"condition={c}:")
            for _, r in subr.iterrows():
                tag = "YES" if r["significant"] else "no"
                print(f"  {r['model_a']} vs {r['model_b']}: p={r['p_value']:.3g}, sig={tag}")

    
    # Export table for thesis reporting
    table_df = sub.copy()
    table_df["condition_percent"] = (table_df["experiment_condition"] / INPUT_TOTAL * 100).round().astype(int)
    table_df["V_hat"] = table_df["V_hat"].round(3)
    table_df["ci_lower"] = table_df["ci_lower"].round(3)
    table_df["ci_upper"] = table_df["ci_upper"].round(3)

    export_table = table_df[[
        "model",
        "condition_percent",
        "V_hat",
        "ci_lower",
        "ci_upper",
        "n"
    ]].rename(columns={
        "model": "Model",
        "condition_percent": "Condition (%)",
        "V_hat": "Visibility Rate",
        "ci_lower": "95% CI Lower",
        "ci_upper": "95% CI Upper",
        "n": "Sample Size"
    })

    export_table["Model"] = export_table["Model"].replace({
        "baseline_cluster": "Baseline",
        "grr_cluster": "GRR",
        "grtx_cluster": "GRTx"
    })

    export_table = export_table.sort_values(["Condition (%)", "Model"])

    os.makedirs(os.path.dirname(OUTPUT_TABLE), exist_ok=True)
    with pd.ExcelWriter(OUTPUT_TABLE, engine="openpyxl") as writer:
        export_table.to_excel(writer, sheet_name="Visibility Table", index=False)

        if results:
            res_df.to_excel(writer, sheet_name="Pairwise Tests", index=False)

    print(f"Saved table output: {OUTPUT_TABLE}")

    fig, ax = plt.subplots(figsize=(max(8, len(conditions) * 1.2), 5))

    x = np.arange(len(conditions), dtype=float)
    group_width = 0.78
    bar_width = group_width / len(models_present)
    offsets = np.linspace(
        -(group_width - bar_width) / 2,
        (group_width - bar_width) / 2,
        len(models_present),
    ) if len(models_present) > 1 else [0.0]

    colors = {
        "baseline_cluster": "#1f77b4",
        "grr_cluster": "#2ca02c",
        "grtx_cluster": "#d62728",
    }

    for offset, (model, label) in zip(offsets, models_present):
        m = sub[sub["model"] == model].set_index("experiment_condition").reindex(conditions)
        ax.bar(x + offset, m["V_hat"].to_numpy(), width=bar_width,
               yerr=m["ci"].to_numpy(), capsize=4, label=label,
               color=colors.get(model), edgecolor="black", linewidth=0.6, alpha=0.9)

    ax.set_xlabel("Percentage of input that is DIS")
    ax.set_ylabel(r"$\hat{V}_{DIS}$  (visibility rate)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{round(c / INPUT_TOTAL * 100):.0f}%" for c in conditions])
    ax.set_ylim(0, 1)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper left", frameon=True)
    ax.set_title("Disability visibility for Llama 3.1 8B models across baseline, GRR, and GRTx",
                 fontsize=11)

    fig.tight_layout()
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    plt.savefig(OUTPUT_FILE, dpi=200, bbox_inches="tight")
    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
