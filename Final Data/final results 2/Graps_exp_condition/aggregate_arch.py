"""
Aggregate architecture comparison across input conditions.

For each model family (GPT-4.1, Llama 3.1 8B), this script:
  1. Builds per-summary DIS visibility proportions
     (summary = unit of analysis, matching the thesis methodology).
  2. Fits a two-way ANOVA: visibility ~ architecture + condition + architecture*condition
     to test (a) whether architectures differ on average across conditions
     and (b) whether the architecture effect varies with condition.
  3. Runs follow-up pairwise Welch t-tests on visibility aggregated across
     all conditions, with the caveat that this discards condition information.

The two-way ANOVA is the principled aggregate test; the pooled Welch is
included only as a sanity check.
"""

import pandas as pd
import numpy as np
from itertools import combinations
from scipy.stats import ttest_ind
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
import os

INPUT_FILE = "Final Data/final results 2/combined/combined_iteration_topic.xlsx"
BUCKET = "DIS"
OUTPUT_FILE = "Final Data/final results 2/combined/graphs by condition/aggregate_arch_results.xlsx"

df = pd.read_excel(INPUT_FILE, sheet_name="Sheet1")

# per-summary DIS visibility
sub = df[(df["present_in_input"] == 1) & (df["bucket"] == BUCKET)].copy()
per_iter = (sub.groupby(["model","experiment_condition","iteration"])
              .agg(visible=("V_t","sum"), total=("V_t","size")).reset_index())
per_iter["vis"] = per_iter["visible"] / per_iter["total"]
per_iter = per_iter[per_iter["experiment_condition"] > 0]  # drop 0% (no DIS topics)

# split model family from architecture
def parse(m):
    parts = m.split("_")
    return parts[0], parts[1]
per_iter["arch"], per_iter["family"] = zip(*per_iter["model"].map(parse))

anova_tables = []
pairwise_tables = []
summary_tables = []

for family, fname in [("gpt","GPT-4.1"),("cluster","Llama 3.1 8B")]:
   print("=" * 72)
   print(f"  Model family: {fname}")
   print("=" * 72)
   d = per_iter[per_iter["family"] == family].copy()
   d["arch"] = pd.Categorical(d["arch"], categories=["baseline", "grr", "grtx"])
   d["condition"] = pd.Categorical(d["experiment_condition"])

   # --- Two-way ANOVA ---
   model = smf.ols("vis ~ C(arch) + C(condition) + C(arch):C(condition)", data=d).fit()
   aov = anova_lm(model, typ=2)
   print("\nTwo-way ANOVA (architecture x condition):")
   print(aov.round(4))
   anova_out = aov.reset_index().rename(columns={"index": "term"})
   anova_out.insert(0, "family", fname)
   anova_tables.append(anova_out)

   # --- Pairwise architecture comparisons, aggregated across conditions ---
   print("\nPairwise Welch t-tests on visibility aggregated across all conditions:")
   print("(NB: ignores condition variation; ANOVA above is the principled test)")
   for a, b in combinations(["baseline", "grr", "grtx"], 2):
      va = d[d["arch"] == a]["vis"].values
      vb = d[d["arch"] == b]["vis"].values
      t_stat, p = ttest_ind(va, vb, equal_var=False)
      print(
         f"  {a:>8s} vs {b:<8s} : n=({len(va)},{len(vb)}), "
         f"means=({va.mean():.3f},{vb.mean():.3f}), t={t_stat:+.2f}, p={p:.4f}"
      )
      pairwise_tables.append({
         "family": fname,
         "comparison": f"{a} vs {b}",
         "n_a": len(va),
         "n_b": len(vb),
         "mean_a": float(va.mean()),
         "mean_b": float(vb.mean()),
         "t_stat": float(t_stat),
         "p_value": float(p),
      })

   # --- Marginal means per architecture (averaging over conditions) ---
   print("\nMarginal mean DIS visibility per architecture (averaged over conditions):")
   mm = d.groupby("arch", observed=True)["vis"].agg(["mean", "std", "count"]).round(3)
   print(mm)
   print()
   mm_out = mm.reset_index().rename(columns={"arch": "architecture"})
   mm_out.insert(0, "family", fname)
   summary_tables.append(mm_out)

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
   if anova_tables:
      pd.concat(anova_tables, ignore_index=True).to_excel(writer, sheet_name="ANOVA", index=False)
   if pairwise_tables:
      pd.DataFrame(pairwise_tables).to_excel(writer, sheet_name="Pairwise Tests", index=False)
   if summary_tables:
      pd.concat(summary_tables, ignore_index=True).to_excel(writer, sheet_name="Marginal Means", index=False)

print(f"Saved Excel output: {OUTPUT_FILE}")
