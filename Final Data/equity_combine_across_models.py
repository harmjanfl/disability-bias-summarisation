#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Combine per-model equity outputs into cross-model files for tables and plots.

Reads `*_iteration_topic.xlsx` files from every subfolder of INPUT_ROOT,
overrides the `model` column with the subfolder name (with the
`_results_simple` suffix stripped, e.g.
`baseline_cluster_results_simple` -> `baseline_cluster`),
concatenates everything, and re-runs the aggregation steps so that every
output carries a `model` column.

Folder layout assumed:

    final results 2/
        baseline_cluster_results_simple/*_iteration_topic.xlsx
        baseline_gpt_results_simple/*_iteration_topic.xlsx
        grr_cluster_results_simple/*_iteration_topic.xlsx
        grr_gpt_results_simple/*_iteration_topic.xlsx
        grtx_gpt_results_simple/*_iteration_topic.xlsx
        combined/                       <- created by this script

Outputs (under OUT_DIR):

- combined_iteration_topic.xlsx
- combined_visibility_by_k.xlsx          sheets: per_topic_k, by_condition_DIS, by_bucket_k
- combined_k_star.xlsx                   sheets: per_topic, per_bucket
- combined_matched_pair_gaps.xlsx        sheets: gap_curves, k_star_gaps
- combined_bucket_summary.xlsx
- combined_bucket_gap_curves.xlsx

Wide pivots for cross-model comparison tables:

- comparison_k_star_wide.xlsx
- comparison_k_star_bucket_wide.xlsx
- comparison_k_star_gaps_wide.xlsx
- comparison_bucket_summary_wide.xlsx
"""

import glob
import os
from typing import List

import pandas as pd

from equity_calc_prioritized_visibility import (
    ALPHAS,
    MIN_ITER_PER_BIN_FOR_K_STAR,
    compute_visibility_by_k_per_topic,
    compute_visibility_by_k_condition_dis,
    compute_visibility_by_k_bucket,
    compute_k_star,
    compute_k_star_bucket,
    compute_matched_pair_gap_curves,
    compute_matched_pair_k_star_gaps,
    compute_bucket_summary,
    compute_bucket_gap_curves,
)

# -----------------------------
# Config
# -----------------------------

INPUT_ROOT = "Final Data/final results 2"
FILE_PATTERN = "*_iteration_topic.xlsx"
# only consider subfolders ending in this suffix as model-result folders
RESULTS_SUFFIX = "_results_simple"

OUT_DIR = os.path.join(INPUT_ROOT, "combined")

OUT_COMBINED_ITERATION = os.path.join(OUT_DIR, "combined_iteration_topic.xlsx")
OUT_COMBINED_VIS_BY_K = os.path.join(OUT_DIR, "combined_visibility_by_k.xlsx")
OUT_COMBINED_K_STAR = os.path.join(OUT_DIR, "combined_k_star.xlsx")
OUT_COMBINED_PAIR_GAPS = os.path.join(OUT_DIR, "combined_matched_pair_gaps.xlsx")
OUT_COMBINED_BUCKET_SUMMARY = os.path.join(OUT_DIR, "combined_bucket_summary.xlsx")
OUT_COMBINED_BUCKET_GAPS = os.path.join(OUT_DIR, "combined_bucket_gap_curves.xlsx")

OUT_COMPARISON_K_STAR = os.path.join(OUT_DIR, "comparison_k_star_wide.xlsx")
OUT_COMPARISON_K_STAR_BUCKET = os.path.join(OUT_DIR, "comparison_k_star_bucket_wide.xlsx")
OUT_COMPARISON_PAIR_GAPS = os.path.join(OUT_DIR, "comparison_k_star_gaps_wide.xlsx")
OUT_COMPARISON_BUCKET_SUMMARY = os.path.join(OUT_DIR, "comparison_bucket_summary_wide.xlsx")


def model_name_from_folder(folder_name: str) -> str:
    """Strip the _results_simple suffix if present."""
    if folder_name.endswith(RESULTS_SUFFIX):
        return folder_name[: -len(RESULTS_SUFFIX)]
    return folder_name


def discover_input_files(input_root: str) -> List[tuple]:
    """
    Return a list of (model_name, path) for every iteration_topic file found
    in a *_results_simple subfolder of input_root.
    """
    pairs = []
    if not os.path.isdir(input_root):
        print(f"INPUT_ROOT does not exist: {input_root}")
        return pairs
    for entry in sorted(os.listdir(input_root)):
        sub = os.path.join(input_root, entry)
        if not os.path.isdir(sub):
            continue
        if not entry.endswith(RESULTS_SUFFIX):
            continue
        model = model_name_from_folder(entry)
        for path in sorted(glob.glob(os.path.join(sub, FILE_PATTERN))):
            pairs.append((model, path))
    return pairs


def load_iteration_topic_files(pairs: List[tuple]) -> pd.DataFrame:
    """Load iteration-topic files and overwrite the model column with the
    folder-derived model name. If multiple files are present for a single
    model (e.g. split across runs), they are concatenated.
    """
    frames = []
    for model, path in pairs:
        df = pd.read_excel(path)
        df["model"] = model
        # equity_calc_prioritized_visibility.py expects `input_count`,
        # the simple per-model script writes `n_t`. Bridge the schema.
        if "input_count" not in df.columns and "n_t" in df.columns:
            df = df.rename(columns={"n_t": "input_count"})
        if "bucket" not in df.columns:
            df["bucket"] = df["group"].apply(
                lambda g: "DIS" if str(g).startswith("DIS")
                else ("GEN" if str(g).startswith("GEN") else "OTHER")
            )
            print(f"  note: derived bucket column from group in {os.path.basename(path)}")
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def pivot_k_star_wide(k_star_df: pd.DataFrame) -> pd.DataFrame:
    if k_star_df.empty:
        return pd.DataFrame()
    wide = k_star_df.pivot_table(
        index=["group", "k_definition", "alpha"],
        columns="model",
        values="k_star",
        aggfunc="first",
    ).reset_index()
    wide.columns.name = None
    return wide


def pivot_k_star_bucket_wide(k_star_bucket_df: pd.DataFrame) -> pd.DataFrame:
    if k_star_bucket_df.empty:
        return pd.DataFrame()
    wide = k_star_bucket_df.pivot_table(
        index=["bucket", "k_definition", "alpha"],
        columns="model",
        values="k_star",
        aggfunc="first",
    ).reset_index()
    wide.columns.name = None
    return wide


def pivot_pair_gaps_wide(pair_k_star_gaps: pd.DataFrame) -> pd.DataFrame:
    if pair_k_star_gaps.empty:
        return pd.DataFrame()
    wide = pair_k_star_gaps.pivot_table(
        index=["pair_label", "t_A", "t_B", "k_definition", "alpha"],
        columns="model",
        values="k_star_gap",
        aggfunc="first",
    ).reset_index()
    wide.columns.name = None
    return wide


def pivot_bucket_summary_wide(bucket_summary_df: pd.DataFrame) -> pd.DataFrame:
    if bucket_summary_df.empty:
        return pd.DataFrame()
    wide = bucket_summary_df.pivot_table(
        index=["bucket", "weighting"],
        columns="model",
        values="V_hat",
        aggfunc="first",
    ).reset_index()
    wide.columns.name = None
    return wide


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    pairs = discover_input_files(INPUT_ROOT)
    if not pairs:
        print(f"No iteration_topic files found under {INPUT_ROOT}")
        return

    print(f"Loading {len(pairs)} per-model files:")
    for model, path in pairs:
        print(f"  - [{model}] {path}")

    iter_topic_df = load_iteration_topic_files(pairs)
    if iter_topic_df.empty:
        print("No data loaded.")
        return

    models_found = sorted(iter_topic_df["model"].dropna().unique().tolist())
    print(f"Models in combined data: {models_found}")

    # re-run aggregations on the union
    vis_per_topic = compute_visibility_by_k_per_topic(iter_topic_df)
    vis_condition = compute_visibility_by_k_condition_dis(iter_topic_df)
    vis_by_k_all = pd.concat([vis_per_topic, vis_condition], ignore_index=True)
    vis_bucket = compute_visibility_by_k_bucket(iter_topic_df)

    k_star_df = compute_k_star(vis_by_k_all, ALPHAS, MIN_ITER_PER_BIN_FOR_K_STAR)
    k_star_bucket_df = compute_k_star_bucket(vis_bucket, ALPHAS, MIN_ITER_PER_BIN_FOR_K_STAR)

    pair_gap_curves = compute_matched_pair_gap_curves(vis_by_k_all)
    pair_k_star_gaps = compute_matched_pair_k_star_gaps(k_star_df)

    bucket_summary_df = compute_bucket_summary(iter_topic_df)
    bucket_gaps_df = compute_bucket_gap_curves(vis_bucket, MIN_ITER_PER_BIN_FOR_K_STAR)

    # wide pivots
    k_star_wide = pivot_k_star_wide(k_star_df)
    k_star_bucket_wide = pivot_k_star_bucket_wide(k_star_bucket_df)
    pair_gaps_wide = pivot_pair_gaps_wide(pair_k_star_gaps)
    bucket_summary_wide = pivot_bucket_summary_wide(bucket_summary_df)

    # write outputs
    iter_topic_df.to_excel(OUT_COMBINED_ITERATION, index=False)

    with pd.ExcelWriter(OUT_COMBINED_VIS_BY_K) as writer:
        if not vis_per_topic.empty:
            vis_per_topic.to_excel(writer, sheet_name="per_topic_k", index=False)
        if not vis_condition.empty:
            vis_condition.to_excel(writer, sheet_name="by_condition_DIS", index=False)
        if not vis_bucket.empty:
            vis_bucket.to_excel(writer, sheet_name="by_bucket_k", index=False)

    with pd.ExcelWriter(OUT_COMBINED_K_STAR) as writer:
        if not k_star_df.empty:
            k_star_df.to_excel(writer, sheet_name="per_topic", index=False)
        if not k_star_bucket_df.empty:
            k_star_bucket_df.to_excel(writer, sheet_name="per_bucket", index=False)

    with pd.ExcelWriter(OUT_COMBINED_PAIR_GAPS) as writer:
        if not pair_gap_curves.empty:
            pair_gap_curves.to_excel(writer, sheet_name="gap_curves", index=False)
        if not pair_k_star_gaps.empty:
            pair_k_star_gaps.to_excel(writer, sheet_name="k_star_gaps", index=False)

    if not bucket_summary_df.empty:
        bucket_summary_df.to_excel(OUT_COMBINED_BUCKET_SUMMARY, index=False)
    if not bucket_gaps_df.empty:
        bucket_gaps_df.to_excel(OUT_COMBINED_BUCKET_GAPS, index=False)

    if not k_star_wide.empty:
        k_star_wide.to_excel(OUT_COMPARISON_K_STAR, index=False)
    if not k_star_bucket_wide.empty:
        k_star_bucket_wide.to_excel(OUT_COMPARISON_K_STAR_BUCKET, index=False)
    if not pair_gaps_wide.empty:
        pair_gaps_wide.to_excel(OUT_COMPARISON_PAIR_GAPS, index=False)
    if not bucket_summary_wide.empty:
        bucket_summary_wide.to_excel(OUT_COMPARISON_BUCKET_SUMMARY, index=False)

    print("Saved:")
    print(f"- {OUT_COMBINED_ITERATION}")
    print(f"- {OUT_COMBINED_VIS_BY_K}")
    print(f"- {OUT_COMBINED_K_STAR}")
    print(f"- {OUT_COMBINED_PAIR_GAPS}")
    print(f"- {OUT_COMBINED_BUCKET_SUMMARY}")
    print(f"- {OUT_COMBINED_BUCKET_GAPS}")
    if not k_star_wide.empty:
        print(f"- {OUT_COMPARISON_K_STAR}")
    if not k_star_bucket_wide.empty:
        print(f"- {OUT_COMPARISON_K_STAR_BUCKET}")
    if not pair_gaps_wide.empty:
        print(f"- {OUT_COMPARISON_PAIR_GAPS}")
    if not bucket_summary_wide.empty:
        print(f"- {OUT_COMPARISON_BUCKET_SUMMARY}")
    print(f"Parameters: alphas = {ALPHAS}, min_iter_per_bin = {MIN_ITER_PER_BIN_FOR_K_STAR}")


if __name__ == "__main__":
    main()
