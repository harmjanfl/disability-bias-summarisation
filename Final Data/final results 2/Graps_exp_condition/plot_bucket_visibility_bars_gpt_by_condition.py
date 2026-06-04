#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Plot DIS bucket visibility for GPT models across baseline, GRR, and GRTx,
with experiment_condition on the x-axis (expressed as % of input),
and 95% confidence intervals (mean +- 1.96*SEM, summary level). Pairwise
Welch two-sample t-tests on per-summary visibility are saved as CSV.

Reads:  Final Data/final results 2/combined/combined_iteration_topic.xlsx
Writes: Final Data/final results 2/combined/plot_visibility_dis_gpt_bars_by_condition.png
        Final Data/final results 2/combined/plot_visibility_dis_gpt_stats_by_condition.csv
"""

import os
import math
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import sem, ttest_ind

# -----------------------------
# Config
# -----------------------------
INPUT_FILE = "Final Data/final results 2/combined/combined_iteration_topic.xlsx"
OUTPUT_FILE = "Final Data/final results 2/combined/graphs by condition/plot_visibility_dis_gpt_bars_by_condition.png"
OUTPUT_STATS = "Final Data/final results 2/combined/graphs by condition/plot_visibility_dis_gpt_stats_by_condition.csv"
OUTPUT_STATS_XL = "Final Data/final results 2/combined/graphs by condition/plot_visibility_dis_gpt_stats_by_condition.xlsx"
OUTPUT_TABLE = "Final Data/final results 2/combined/graphs by condition/plot_visibility_dis_gpt_table.xlsx"
INPUT_TOTAL = 50

MODEL_ORDER = [
    ("baseline_gpt", "Baseline"),
    ("grr_gpt",      "GRR"),
    ("grtx_gpt",     "GRTx"),
]
BUCKET = "DIS"


def mean_ci_halfwidth(values, confidence: float = 0.95):
    """
    Compute 95% confidence interval half-width around the mean
    using the standard error of the mean.
    """
    values = pd.Series(values).dropna()

    if len(values) <= 1:
        return np.nan

    return 1.96 * sem(values)


def summary_level_proportions(df: pd.DataFrame, model: str, condition, bucket: str):
    """Return the per-summary visibility proportions for a (model, condition, bucket) cell.

    Each generated summary (iteration) contributes one proportion:
        visible topics in bucket / eligible topics in bucket.
    The summary is the unit of analysis, matching the methodology and the
    summary-level confidence intervals shown on the bars.
    """
    if "present_in_input" not in df.columns:
        df = df.copy()
        df["present_in_input"] = 1

    sub = df[(df["present_in_input"] == 1)
             & (df["model"] == model)
             & (df["experiment_condition"] == condition)
             & (df["bucket"] == bucket)].copy()
    if sub.empty:
        return np.array([])
    per_iter = (
        sub.groupby(["iteration"])
        .agg(visible_topics=("V_t", "sum"), total_topics=("V_t", "size"))
        .reset_index()
    )
    return (per_iter["visible_topics"] / per_iter["total_topics"]).to_numpy()


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
    if "present_in_input" not in df.columns:
        print("Warning: 'present_in_input' column not found - assuming all rows are present.")
        df = df.copy()
        df["present_in_input"] = 1

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
            sd=("visibility_iteration", "std"),
            n=("visibility_iteration", "size")
        )
        .reset_index()
    )

    return grouped


def main():
    df = pd.read_excel(INPUT_FILE, sheet_name="Sheet1")
    agg = aggregate_by_condition(df, BUCKET)
    agg["ci"] = agg["sd"] / np.sqrt(agg["n"]) * 1.96
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
    # Summary-level test: each generated summary is one observation, so the
    # two architectures are compared with a Welch two-sample t-test on their
    # per-summary visibility proportions (n = number of summaries per cell).
    for c in conditions:
        for a, b in combinations(model_names, 2):
            ma, mb = data_by_model[a], data_by_model[b]
            if c not in ma.index or c not in mb.index:
                continue
            # summary-level mean estimates (for reference in the output)
            p1 = ma.at[c, "V_hat"]
            n1_summary = int(ma.at[c, "n"]) if not pd.isna(ma.at[c, "n"]) else 0
            p2 = mb.at[c, "V_hat"]
            n2_summary = int(mb.at[c, "n"]) if not pd.isna(mb.at[c, "n"]) else 0

            # per-summary proportions (the unit of analysis for the test)
            va = summary_level_proportions(df, a, c, BUCKET)
            vb = summary_level_proportions(df, b, c, BUCKET)

            if len(va) < 2 or len(vb) < 2:
                t_stat, pval = math.nan, math.nan
                sig = False
            else:
                t_stat, pval = ttest_ind(va, vb, equal_var=False)
                sig = False if pd.isna(pval) else (pval < 0.05)

            results.append({
                "experiment_condition": c, "model_a": a, "model_b": b,
                "p_value": pval, "t": t_stat, "significant": sig,
                "p_hat_a": p1, "n_a_summary": n1_summary,
                "p_hat_b": p2, "n_b_summary": n2_summary,
            })

    if results:
        res_df = pd.DataFrame(results)
        os.makedirs(os.path.dirname(OUTPUT_STATS), exist_ok=True)
        try:
            res_df.to_csv(OUTPUT_STATS, index=False)
            # also save Excel for convenience
            res_df.to_excel(OUTPUT_STATS_XL, index=False)
            print(f"Saved stats: {OUTPUT_STATS} and {OUTPUT_STATS_XL}")
        except PermissionError:
            alt_csv = OUTPUT_STATS.replace('.csv', '_new.csv')
            alt_xl = OUTPUT_STATS_XL.replace('.xlsx', '_new.xlsx')
            try:
                res_df.to_csv(alt_csv, index=False)
                res_df.to_excel(alt_xl, index=False)
                print(f"Could not overwrite locked files; saved as: {alt_csv} and {alt_xl}")
            except Exception as e:
                print(f"Failed to save stats CSV/XLSX: {e}")
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
    table_df["ci"] = table_df["ci"].round(3)

    export_table = table_df[[
        "model",
        "condition_percent",
        "V_hat",
        "ci",
        "n"
    ]].rename(columns={
        "model": "Model",
        "condition_percent": "Condition (%)",
        "V_hat": "Visibility Rate",
        "ci": "95% CI (±)",
        "n": "Sample Size"
    })

    export_table["Model"] = export_table["Model"].replace({
        "baseline_gpt": "Baseline",
        "grr_gpt": "GRR",
        "grtx_gpt": "GRTx"
    })

    export_table = export_table.sort_values(["Condition (%)", "Model"])

    os.makedirs(os.path.dirname(OUTPUT_TABLE), exist_ok=True)
    try:
        with pd.ExcelWriter(OUTPUT_TABLE, engine="openpyxl") as writer:
            export_table.to_excel(writer, sheet_name="Visibility Table", index=False)

            if results:
                res_df.to_excel(writer, sheet_name="Pairwise Tests", index=False)

        print(f"Saved table output: {OUTPUT_TABLE}")
    except PermissionError:
        alt_table = OUTPUT_TABLE.replace('.xlsx', '_new.xlsx')
        try:
            with pd.ExcelWriter(alt_table, engine="openpyxl") as writer:
                export_table.to_excel(writer, sheet_name="Visibility Table", index=False)
                if results:
                    res_df.to_excel(writer, sheet_name="Pairwise Tests", index=False)
            print(f"Could not overwrite table; saved as: {alt_table}")
        except Exception as e:
            print(f"Failed to save table Excel: {e}")
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
        "baseline_gpt": "#1f77b4",
        "grr_gpt": "#2ca02c",
        "grtx_gpt": "#d62728",
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
    ax.set_title("Disability visibility for GPT 4.1 models across baseline, GRR, and GRTx",
                 fontsize=11)

    fig.tight_layout()
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    plt.savefig(OUTPUT_FILE, dpi=200, bbox_inches="tight")
    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
