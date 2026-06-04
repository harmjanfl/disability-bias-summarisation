"""
Fine-grained subcategory visibility analysis (DIS topics only).

For each model family (GPT-4.1, Llama 3.1 8B), this script:
  1. Aggregates per-subcategory DIS visibility across all input conditions and
     summaries, separately per architecture, producing a mean visibility rate
     and SEM-based 95% CI for each (model, architecture, subcategory) cell.
  2. Produces a wide-format summary table (subcategory x architecture) for each
     model family, ordered by mean visibility, suitable for inclusion in the
     results section.
  3. Writes an Excel file with one sheet per model family plus a 'Combined'
     sheet sorted by cross-model mean for the headline observation.
  4. Produces a horizontal bar plot per model family for visual inspection.

Reads:  Final Data/final results 2/combined/combined_iteration_topic.xlsx
Writes: Final Data/final results 2/combined/graphs by condition/
            subcategory_visibility_table.xlsx
            subcategory_visibility_cluster.png
            subcategory_visibility_gpt.png

Notes:
  - Visibility is measured at the topic-instance level: for each (iteration,
    subcategory) pair where the topic was present in the input, V_t is 1 if
    the topic was visible in the generated summary and 0 otherwise. Mean
    visibility is taken across all such observations per cell.
  - CIs are SEM-based across topic-instance observations, matching the
    aggregation level. Cells differ in n: subcategories that occur in fewer
    conditions have fewer observations.
  - The analysis is exploratory and pools across input conditions; this is
    intentional, because per-subcategory per-condition cells would be too
    small for reliable estimation.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

INPUT_FILE = "Final Data/final results 2/combined/combined_iteration_topic.xlsx"
OUTPUT_DIR = "Final Data/final results 2/combined/graphs by condition"
OUTPUT_TABLE = os.path.join(OUTPUT_DIR, "subcategory_visibility_table.xlsx")
OUTPUT_PLOT_GPT = os.path.join(OUTPUT_DIR, "subcategory_visibility_gpt.png")
OUTPUT_PLOT_CLU = os.path.join(OUTPUT_DIR, "subcategory_visibility_cluster.png")

FAMILY_LABELS = {"gpt": "GPT-4.1", "cluster": "Llama 3.1 8B"}
ARCH_ORDER = ["baseline", "grr", "grtx"]
ARCH_LABELS = {"baseline": "Baseline", "grr": "GRR", "grtx": "GRTx"}
ARCH_COLORS = {"baseline": "#1f77b4", "grr": "#2ca02c", "grtx": "#d62728"}


def parse_model(m):
    parts = m.split("_")
    return parts[0], parts[1]  # arch, family


def build_subcategory_table(df, family):
    """Per-subcategory mean visibility + 95% CI, by architecture, for one family."""
    sub = df[(df["bucket"] == "DIS") & (df["present_in_input"] == 1)].copy()
    sub["arch"], sub["family"] = zip(*sub["model"].map(parse_model))
    sub = sub[sub["family"] == family]

    rows = []
    for group_name in sorted(sub["group"].unique()):
        for arch in ARCH_ORDER:
            cell = sub[(sub["group"] == group_name) & (sub["arch"] == arch)]
            if len(cell) == 0:
                continue
            mean_v = cell["V_t"].mean()
            sd_v = cell["V_t"].std(ddof=1)
            n_v = len(cell)
            ci = 1.96 * sd_v / np.sqrt(n_v) if n_v > 1 else np.nan
            rows.append({
                "subcategory": group_name.replace("DIS ", ""),
                "architecture": ARCH_LABELS[arch],
                "mean_visibility": round(float(mean_v), 3),
                "ci_halfwidth": round(float(ci), 3) if pd.notna(ci) else np.nan,
                "n_observations": int(n_v),
            })
    long_df = pd.DataFrame(rows)

    # Wide: subcategory x architecture, with cross-architecture mean for ordering
    wide = long_df.pivot(index="subcategory", columns="architecture",
                         values="mean_visibility")
    arch_cols = [ARCH_LABELS[a] for a in ARCH_ORDER if ARCH_LABELS[a] in wide.columns]
    wide = wide[arch_cols]
    wide["mean (all arch)"] = wide.mean(axis=1).round(3)
    wide = wide.sort_values("mean (all arch)", ascending=False)
    wide = wide.reset_index()

    return long_df, wide


def plot_subcategory(long_df, family_label, output_path):
    """Horizontal grouped bars: subcategory on y, architectures within each."""
    # Order subcategories by overall mean (across architectures)
    order = (long_df.groupby("subcategory")["mean_visibility"].mean()
             .sort_values(ascending=True).index.tolist())  # ascending for horizontal: lowest at top of plot
    # But conventionally we want highest at top, so:
    order = order[::-1]

    fig, ax = plt.subplots(figsize=(8, max(4, 0.5 * len(order))))
    y = np.arange(len(order))
    bar_h = 0.78 / len(ARCH_ORDER)
    # Reverse offsets so Baseline (first in ARCH_ORDER) sits at the top of each
    # subcategory group, matching the legend order top-to-bottom.
    offsets = np.linspace(bar_h * (len(ARCH_ORDER) - 1) / 2,
                          -(bar_h * (len(ARCH_ORDER) - 1) / 2),
                          len(ARCH_ORDER))

    for arch, offset in zip(ARCH_ORDER, offsets):
        arch_label = ARCH_LABELS[arch]
        sub = long_df[long_df["architecture"] == arch_label].set_index("subcategory")
        vals = [sub.loc[g, "mean_visibility"] if g in sub.index else 0 for g in order]
        errs = [sub.loc[g, "ci_halfwidth"] if g in sub.index else 0 for g in order]
        ax.barh(y + offset, vals, height=bar_h, xerr=errs,
                color=ARCH_COLORS[arch], edgecolor="black", linewidth=0.6,
                label=arch_label, alpha=0.9, capsize=3,
                error_kw={"linewidth": 0.8})

    ax.set_yticks(y)
    ax.set_yticklabels(order)
    ax.set_xlabel(r"Mean visibility rate $\hat{V}$")
    ax.set_xlim(0, 1)
    ax.set_title(f"DIS subcategory visibility — {family_label}\n(aggregated across input conditions)",
                 fontsize=11)
    ax.grid(True, axis="x", alpha=0.3)
    ax.legend(loc="upper right", title="Architecture", frameon=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {output_path}")


def main():
    df = pd.read_excel(INPUT_FILE, sheet_name="Sheet1")

    long_gpt, wide_gpt = build_subcategory_table(df, "gpt")
    long_clu, wide_clu = build_subcategory_table(df, "cluster")

    # Combined ordering (across both model families) for the headline table
    combined = (pd.concat([long_gpt.assign(family="GPT-4.1"),
                           long_clu.assign(family="Llama 3.1 8B")])
                .groupby(["subcategory", "family"])["mean_visibility"].mean()
                .unstack("family").round(3))
    combined["mean (both families)"] = combined.mean(axis=1).round(3)
    combined = combined.sort_values("mean (both families)", ascending=False).reset_index()

    # Print everything
    print("\n=== GPT-4.1 ===")
    print(wide_gpt.to_string(index=False))
    print("\n=== Llama 3.1 8B ===")
    print(wide_clu.to_string(index=False))
    print("\n=== Combined (cross-model mean visibility per subcategory) ===")
    print(combined.to_string(index=False))

    # Write to Excel
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    try:
        with pd.ExcelWriter(OUTPUT_TABLE, engine="openpyxl") as writer:
            wide_gpt.to_excel(writer, sheet_name="GPT-4.1", index=False)
            wide_clu.to_excel(writer, sheet_name="Llama 3.1 8B", index=False)
            combined.to_excel(writer, sheet_name="Combined", index=False)
            long_gpt.to_excel(writer, sheet_name="GPT-4.1 long", index=False)
            long_clu.to_excel(writer, sheet_name="Llama 3.1 8B long", index=False)
        print(f"\nWrote table: {OUTPUT_TABLE}")
    except PermissionError:
        alt = OUTPUT_TABLE.replace(".xlsx", "_new.xlsx")
        with pd.ExcelWriter(alt, engine="openpyxl") as writer:
            wide_gpt.to_excel(writer, sheet_name="GPT-4.1", index=False)
            wide_clu.to_excel(writer, sheet_name="Llama 3.1 8B", index=False)
            combined.to_excel(writer, sheet_name="Combined", index=False)
            long_gpt.to_excel(writer, sheet_name="GPT-4.1 long", index=False)
            long_clu.to_excel(writer, sheet_name="Llama 3.1 8B long", index=False)
        print(f"\nCould not overwrite; wrote to: {alt}")

    plot_subcategory(long_gpt, "GPT-4.1", OUTPUT_PLOT_GPT)
    plot_subcategory(long_clu, "Llama 3.1 8B", OUTPUT_PLOT_CLU)


if __name__ == "__main__":
    main()
