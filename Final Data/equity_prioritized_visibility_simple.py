#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Equity-oriented analysis: prioritised visibility (thesis Section 4.1.2).

Strict implementation of the equations in Section 4.1.2, plus a bucket-level
aggregation that pools all DIS topics and all GEN topics.

    Eq. 4.1  sim(x_i, s_j) = cosine(e(x_i), e(s_j))
    Eq. 4.2  A_tj = (1 / n_t) * sum_{i in I_t} sim(x_i, s_j)
    Eq. 4.3  Assign s_j to t*_j = argmax_t A_tj iff
                 A_{t*_j, j} >= theta   AND   A_{t*_j, j} - max_{t != t*_j} A_tj >= delta
             Otherwise s_j is unassigned.
             m_t = sum_j 1[hat{t}_j = t]
    V_t      V_t = 1 iff m_t >= 1
    Eq. 4.4  V_hat_t(k) = (1/R) sum_r V_t^(r), aggregated over iterations
             where topic t appears with input frequency n_t = k.
    Eq. 4.5  k*_t(alpha) = min { k : V_hat_t(k) >= alpha }
    Eq. 4.6  Equity (matched pair t_A vs t_B):  V_hat_{t_A}(k) > V_hat_{t_B}(k)
    Eq. 4.7  k*-gap:  k*_{t_B}(alpha) - k*_{t_A}(alpha)  > 0 means DIS is prioritised

Bucket extension (not yet in the thesis; document it before reporting):
    V_hat_{B}(k) is computed by pooling every (iteration, topic) observation
    where topic is in bucket B and that topic appears k times in the input:
        V_hat_{B}(k) = (sum of V_t over such observations) / (count of such observations)
    k*_{B}(alpha) is then defined exactly as in Eq. 4.5 but applied to V_hat_{B}(k).

Input pickle payload format (unchanged from the original script):
    payload = {
        "iteration":            <int>,
        "experiment_condition": <int>,
        "summary_sentences":    [<str>, ...],
        "cosine_similarities":  cos_matrix,   # cos_matrix[j][i] = sim(s_j, x_i)
        "sent_df":              <DataFrame with TOPIC_COLUMN labelling each input row>,
    }

Outputs (under OUT_DIR):
    - iteration_topic.xlsx      one row per (iteration, topic): m_t, V_t, n_t
    - visibility_by_k.xlsx      sheets: per_topic, per_bucket  (V_hat curves)
    - k_star.xlsx               sheets: per_topic, per_bucket  (k* per alpha)
    - matched_pair_gaps.xlsx    sheets: gap_curves, k_star_gaps
"""

import glob
import os
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# -----------------------------
# Config
# -----------------------------

MODEL_NAME = "grtx_cluster"

SIM_PICKLE_DIR = "Final Data/grtx/grtx_raw_cosines/grtx_equity_cluster_metadata_cosines"
PICKLE_PATTERN = "cos_sims_iteration_*.pkl"
TOPIC_COLUMN = "group"

OUT_DIR = "Final Data/final results 2/grtx_cluster_results_simple"
OUT_ITERATION_TOPIC = os.path.join(OUT_DIR, f"{MODEL_NAME}_iteration_topic.xlsx")
OUT_VISIBILITY_BY_K = os.path.join(OUT_DIR, f"{MODEL_NAME}_visibility_by_k.xlsx")
OUT_K_STAR = os.path.join(OUT_DIR, f"{MODEL_NAME}_k_star.xlsx")
OUT_MATCHED_PAIR_GAPS = os.path.join(OUT_DIR, f"{MODEL_NAME}_matched_pair_gaps.xlsx")

# Gating parameters (Eq. 4.3). Calibration is reported in Section 5.1 of the thesis.
THETA = 0.40   # minimum evidence threshold
DELTA = 0.02   # ambiguity margin

# Target visibility rates for k* (Eq. 4.5).
ALPHAS = [0.5, 0.6, 0.7, 0.8]

# Full topic taxonomy T (Section 5.1).
ALL_CATEGORIES = [
    "DIS accessibility", "DIS communication", "DIS eligibility", "DIS information",
    "DIS organization", "DIS prioritization", "DIS services", "DIS specialized_support",
    "DIS supplies",
    "GEN accessibility", "GEN communication", "GEN crowding", "GEN eligibility",
    "GEN information", "GEN organization", "GEN prioritization", "GEN registration",
    "GEN services", "GEN supplies",
]

# Matched DIS-GEN pairs for equity analysis (Eqs. 4.6 and 4.7).
MATCHED_PAIRS = [
    ("DIS accessibility",  "GEN accessibility"),
    ("DIS communication",  "GEN communication"),
    ("DIS eligibility",    "GEN eligibility"),
    ("DIS information",    "GEN information"),
    ("DIS organization",   "GEN organization"),
    ("DIS prioritization", "GEN prioritization"),
    ("DIS services",       "GEN services"),
    ("DIS supplies",       "GEN supplies"),
]


def topic_bucket(group_name: str) -> str:
    if group_name.startswith("DIS"):
        return "DIS"
    if group_name.startswith("GEN"):
        return "GEN"
    return "OTHER"


# =============================================================================
# Per-iteration analysis: Eqs. 4.1 - 4.3 and the per-summary count m_t
# =============================================================================


def topic_affinity(
    cos_matrix,
    summary_idx: int,
    input_indices: List[int],
) -> float:
    """Eq. 4.2: A_tj = mean over i in I_t of sim(x_i, s_j).

    Returns -inf if the topic does not appear in the input (n_t = 0), so it
    cannot win the argmax and the sentence will fall through the gate.
    """
    if not input_indices:
        return float("-inf")
    sims = [cos_matrix[summary_idx][i] for i in input_indices]
    return sum(sims) / len(sims)


def assign_summary_sentence(
    affinities: Dict[str, float],
    theta: float,
    delta: float,
) -> Optional[str]:
    """Eq. 4.3: gated hard assignment.

    Returns the assigned topic, or None if either gating condition fails.
    """
    valid = [(t, a) for t, a in affinities.items() if a != float("-inf")]
    if not valid:
        return None
    valid.sort(key=lambda ta: ta[1], reverse=True)
    best_topic, best_score = valid[0]
    second_score = valid[1][1] if len(valid) > 1 else float("-inf")

    if best_score < theta:
        return None
    if best_score - second_score < delta:
        return None
    return best_topic


def analyse_iteration(payload, theta: float, delta: float) -> Optional[pd.DataFrame]:
    """One row per (iteration, topic). Columns:
        - n_t                   = input_count
        - m_t                   = number of summary sentences assigned to t
        - V_t                   = 1 if m_t >= 1 else 0
        - present_in_input      = 1 if n_t >= 1 else 0
    """
    iteration = payload.get("iteration")
    condition = payload.get("experiment_condition")
    summary_sentences = payload["summary_sentences"]
    cos_matrix = payload["cosine_similarities"]
    sent_df = payload["sent_df"].copy()

    if sent_df.empty or len(summary_sentences) == 0:
        return None

    # Restrict to the predefined taxonomy.
    sent_df = sent_df[sent_df[TOPIC_COLUMN].isin(ALL_CATEGORIES)].copy()
    if sent_df.empty:
        return None

    # Build I_t: input-row indices belonging to each topic.
    topic_to_indices: Dict[str, List[int]] = {t: [] for t in ALL_CATEGORIES}
    for idx, topic in zip(sent_df.index.tolist(), sent_df[TOPIC_COLUMN].tolist()):
        topic_to_indices[topic].append(idx)

    # m_t accumulator.
    m_t = {t: 0 for t in ALL_CATEGORIES}

    for j in range(len(summary_sentences)):
        # Eq. 4.2: compute A_tj for every topic.
        affinities = {
            t: topic_affinity(cos_matrix, j, topic_to_indices[t])
            for t in ALL_CATEGORIES
        }
        # Eq. 4.3: gated assignment.
        assigned = assign_summary_sentence(affinities, theta, delta)
        if assigned is not None:
            m_t[assigned] += 1

    rows = []
    for t in ALL_CATEGORIES:
        n_t = len(topic_to_indices[t])
        rows.append({
            "model": MODEL_NAME,
            "iteration": iteration,
            "experiment_condition": condition,
            "group": t,
            "bucket": topic_bucket(t),
            "n_t": n_t,
            "present_in_input": int(n_t > 0),
            "m_t": m_t[t],
            "V_t": int(m_t[t] >= 1),
        })
    return pd.DataFrame(rows)


# =============================================================================
# Aggregation: V_hat curves (Eq. 4.4) - per topic and per bucket
# =============================================================================


def compute_visibility_by_k_per_topic(iter_topic_df: pd.DataFrame) -> pd.DataFrame:
    """Eq. 4.4: V_hat_t(k) where k = n_t (the actual input count of topic t
    in that iteration). Restricted to iterations where the topic is present
    (k >= 1).
    """
    present = iter_topic_df[iter_topic_df["present_in_input"] == 1]
    if present.empty:
        return pd.DataFrame()

    grouped = (
        present.groupby(["model", "group", "n_t"])
        .agg(n_iterations=("V_t", "size"), n_visible=("V_t", "sum"))
        .reset_index()
        .rename(columns={"n_t": "k"})
    )
    grouped["V_hat"] = grouped["n_visible"] / grouped["n_iterations"]
    return grouped


def compute_visibility_by_k_per_bucket(iter_topic_df: pd.DataFrame) -> pd.DataFrame:
    """Bucket extension: pool all topics in a bucket at each k.

    For each bucket B in {DIS, GEN} and each k >= 1:
        V_hat_B(k) = sum of V_t over (iteration, topic) with topic in B and n_t = k
                     ---------------------------------------------------------------
                     number of (iteration, topic) observations with topic in B and n_t = k
    """
    present = iter_topic_df[iter_topic_df["present_in_input"] == 1]
    if present.empty:
        return pd.DataFrame()

    grouped = (
        present.groupby(["model", "bucket", "n_t"])
        .agg(n_iterations=("V_t", "size"), n_visible=("V_t", "sum"))
        .reset_index()
        .rename(columns={"n_t": "k"})
    )
    grouped["V_hat"] = grouped["n_visible"] / grouped["n_iterations"]
    return grouped


# =============================================================================
# Aggregation: k_star (Eq. 4.5)
# =============================================================================


def _smallest_k_above_alpha(sub_sorted: pd.DataFrame, alpha: float) -> Tuple[float, bool]:
    """Eq. 4.5: min { k : V_hat(k) >= alpha }. Returns (k_star, reached_flag)."""
    qualifying = sub_sorted[sub_sorted["V_hat"] >= alpha]
    if not qualifying.empty:
        return float(qualifying.iloc[0]["k"]), True
    return float("nan"), False


def compute_k_star(
    vis_by_k: pd.DataFrame,
    alphas: List[float],
    entity_col: str,
) -> pd.DataFrame:
    """Generic k_star scan over (model, <entity>, alpha). entity_col is
    'group' for per-topic curves or 'bucket' for per-bucket curves.
    """
    if vis_by_k.empty:
        return pd.DataFrame()

    rows = []
    for (model, entity), sub in vis_by_k.groupby(["model", entity_col]):
        sub_sorted = sub.sort_values("k")
        for alpha in alphas:
            k_star, reached = _smallest_k_above_alpha(sub_sorted, alpha)
            rows.append({
                "model": model,
                entity_col: entity,
                "alpha": alpha,
                "k_star": k_star,
                "reached_alpha": reached,
                "max_V_hat": float(sub_sorted["V_hat"].max()),
                "max_k_observed": int(sub_sorted["k"].max()),
            })
    return pd.DataFrame(rows)


# =============================================================================
# Matched-pair equity: Eqs. 4.6 and 4.7
# =============================================================================


def compute_matched_pair_gap_curves(vis_per_topic: pd.DataFrame) -> pd.DataFrame:
    """Eq. 4.6: gap(k) = V_hat_{t_A}(k) - V_hat_{t_B}(k) at each shared k.
    Positive => DIS topic is prioritised at that k.
    """
    if vis_per_topic.empty:
        return pd.DataFrame()

    rows = []
    for model, sub_model in vis_per_topic.groupby("model"):
        for t_A, t_B in MATCHED_PAIRS:
            a = sub_model[sub_model["group"] == t_A].set_index("k")
            b = sub_model[sub_model["group"] == t_B].set_index("k")
            shared_k = sorted(set(a.index) & set(b.index))
            for k in shared_k:
                rows.append({
                    "model": model,
                    "pair_label": f"{t_A} vs {t_B}",
                    "t_A": t_A,
                    "t_B": t_B,
                    "k": int(k),
                    "V_hat_A": float(a.loc[k, "V_hat"]),
                    "V_hat_B": float(b.loc[k, "V_hat"]),
                    "gap_V_hat": float(a.loc[k, "V_hat"] - b.loc[k, "V_hat"]),
                    "n_iter_A": int(a.loc[k, "n_iterations"]),
                    "n_iter_B": int(b.loc[k, "n_iterations"]),
                })
    return pd.DataFrame(rows)


def compute_matched_pair_k_star_gaps(k_star_per_topic: pd.DataFrame) -> pd.DataFrame:
    """Eq. 4.7: k*_gap(alpha) = k*_{t_B}(alpha) - k*_{t_A}(alpha).
    Positive => DIS topic reaches the target visibility at a lower k.
    """
    if k_star_per_topic.empty:
        return pd.DataFrame()

    rows = []
    for (model, alpha), sub in k_star_per_topic.groupby(["model", "alpha"]):
        indexed = sub.set_index("group")
        for t_A, t_B in MATCHED_PAIRS:
            if t_A not in indexed.index or t_B not in indexed.index:
                continue
            k_star_A = indexed.loc[t_A, "k_star"]
            k_star_B = indexed.loc[t_B, "k_star"]
            reached_A = bool(indexed.loc[t_A, "reached_alpha"])
            reached_B = bool(indexed.loc[t_B, "reached_alpha"])

            gap = (k_star_B - k_star_A) if (reached_A and reached_B) else np.nan
            rows.append({
                "model": model,
                "alpha": alpha,
                "pair_label": f"{t_A} vs {t_B}",
                "t_A": t_A,
                "t_B": t_B,
                "k_star_A": k_star_A,
                "k_star_B": k_star_B,
                "reached_A": reached_A,
                "reached_B": reached_B,
                "k_star_gap": gap,
            })
    return pd.DataFrame(rows)


# =============================================================================
# Main
# =============================================================================


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    paths = sorted(glob.glob(os.path.join(SIM_PICKLE_DIR, PICKLE_PATTERN)))
    if not paths:
        print(f"No pickles matched {os.path.join(SIM_PICKLE_DIR, PICKLE_PATTERN)}")
        return

    # Step 1: per-iteration analysis (Eqs. 4.1 - 4.3 and V_t).
    iter_frames = []
    for path in paths:
        with open(path, "rb") as f:
            payload = pickle.load(f)
        iter_df = analyse_iteration(payload, theta=THETA, delta=DELTA)
        if iter_df is not None:
            iter_frames.append(iter_df)

    if not iter_frames:
        print("No results to aggregate.")
        return

    iter_topic_df = pd.concat(iter_frames, ignore_index=True)

    # Step 2: V_hat curves (Eq. 4.4), per topic and per bucket.
    vis_per_topic = compute_visibility_by_k_per_topic(iter_topic_df)
    vis_per_bucket = compute_visibility_by_k_per_bucket(iter_topic_df)

    # Step 3: k_star (Eq. 4.5), per topic and per bucket.
    k_star_per_topic = compute_k_star(vis_per_topic, ALPHAS, entity_col="group")
    k_star_per_bucket = compute_k_star(vis_per_bucket, ALPHAS, entity_col="bucket")

    # Step 4: matched-pair equity (Eqs. 4.6 and 4.7).
    pair_gap_curves = compute_matched_pair_gap_curves(vis_per_topic)
    pair_k_star_gaps = compute_matched_pair_k_star_gaps(k_star_per_topic)

    # Write outputs.
    iter_topic_df.to_excel(OUT_ITERATION_TOPIC, index=False)

    with pd.ExcelWriter(OUT_VISIBILITY_BY_K) as writer:
        if not vis_per_topic.empty:
            vis_per_topic.to_excel(writer, sheet_name="per_topic", index=False)
        if not vis_per_bucket.empty:
            vis_per_bucket.to_excel(writer, sheet_name="per_bucket", index=False)

    with pd.ExcelWriter(OUT_K_STAR) as writer:
        if not k_star_per_topic.empty:
            k_star_per_topic.to_excel(writer, sheet_name="per_topic", index=False)
        if not k_star_per_bucket.empty:
            k_star_per_bucket.to_excel(writer, sheet_name="per_bucket", index=False)

    with pd.ExcelWriter(OUT_MATCHED_PAIR_GAPS) as writer:
        if not pair_gap_curves.empty:
            pair_gap_curves.to_excel(writer, sheet_name="gap_curves", index=False)
        if not pair_k_star_gaps.empty:
            pair_k_star_gaps.to_excel(writer, sheet_name="k_star_gaps", index=False)

    print("Saved:")
    print(f"- {OUT_ITERATION_TOPIC}")
    print(f"- {OUT_VISIBILITY_BY_K}")
    print(f"- {OUT_K_STAR}")
    print(f"- {OUT_MATCHED_PAIR_GAPS}")
    print(f"Parameters: theta={THETA}, delta={DELTA}, alphas={ALPHAS}, model={MODEL_NAME}")


if __name__ == "__main__":
    main()
