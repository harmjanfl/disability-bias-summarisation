"""
Compare DeepEval scores across models.

Auto-discovers result files (.csv or .xlsx) produced by evaluate_summaries.py
in a folder, builds a summary table, and plots side-by-side box plots for
summarization_score and hallucination_score.

The model name for each file is taken from the filename stem
(e.g. 'cluster_grr_responses.csv' -> 'cluster_grr_responses').

Usage:
    python compare_models.py
    python compare_models.py --input-dir results --output-dir comparison
    python compare_models.py --input-dir results --output-dir comparison --pattern "*.xlsx"
"""

import argparse
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd
from scipy.stats import ttest_ind


SCORE_COLS = ["summarization_score", "hallucination_score"]
ARCHITECTURE_ORDER = ["Baseline", "GRR", "GRTx"]
ARCHITECTURE_COLORS = {
    "Baseline": "#1f77b4",
    "GRR": "#2ca02c",
    "GRTx": "#d62728",
}
FAMILY_ORDER = ["GPT-4.1", "Llama 3.1 8B"]

# -----------------------------
# Editable run config
# -----------------------------
# Update these values at the top of the file to run directly in VS Code.
DEFAULT_INPUT_DIR = "Final Data/deepeval/results"
DEFAULT_OUTPUT_DIR = "Final Data/deepeval/comparison"
DEFAULT_PATTERN = "*.csv"


def infer_architecture(stem: str) -> str:
    name = stem.lower()
    if "grtx" in name:
        return "GRTx"
    if "grr" in name:
        return "GRR"
    if "baseline" in name:
        return "Baseline"
    return stem


def infer_family(stem: str) -> str:
    name = stem.lower()
    if "llama" in name or "cluster" in name:
        return "Llama 3.1 8B"
    if "gpt" in name:
        return "GPT-4.1"
    return "Other"


def make_display_label(stem: str) -> str:
    architecture = infer_architecture(stem)
    family = infer_family(stem)
    if family == "Other":
        return architecture
    return f"{architecture} ({family})"


def family_sort_key(family: str) -> int:
    try:
        return FAMILY_ORDER.index(family)
    except ValueError:
        return len(FAMILY_ORDER)


def architecture_sort_key(architecture: str) -> int:
    try:
        return ARCHITECTURE_ORDER.index(architecture)
    except ValueError:
        return len(ARCHITECTURE_ORDER)


def load_result_file(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    elif path.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")
    df["source_file"] = path.name
    df["file_stem"] = path.stem
    df["architecture"] = infer_architecture(path.stem)
    df["family"] = infer_family(path.stem)
    df["display_label"] = make_display_label(path.stem)
    return df


def discover_files(folder: Path, pattern: str) -> list[Path]:
    files = sorted(folder.glob(pattern))
    # de-dup if both csv and xlsx of same stem exist (prefer csv)
    by_stem: dict[str, Path] = {}
    for f in files:
        if f.stem in by_stem and f.suffix.lower() != ".csv":
            continue
        by_stem[f.stem] = f
    return sorted(by_stem.values())


def build_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per model, columns = mean/std/min/max/n for each score."""
    rows = []
    for label, sub in df.groupby("display_label"):
        row = {
            "model": label,
            "family": sub["family"].iloc[0] if "family" in sub.columns else "Other",
            "architecture": sub["architecture"].iloc[0] if "architecture" in sub.columns else label,
            "n_rows": len(sub),
        }
        for col in SCORE_COLS:
            if col in sub.columns:
                s = pd.to_numeric(sub[col], errors="coerce").dropna()
                row[f"{col}_mean"]  = s.mean() if len(s) else None
                row[f"{col}_std"]   = s.std()  if len(s) else None
                row[f"{col}_min"]   = s.min()  if len(s) else None
                row[f"{col}_max"]   = s.max()  if len(s) else None
                row[f"{col}_n"]     = len(s)
        rows.append(row)
    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary["family_order"] = summary["family"].map(family_sort_key)
        summary["architecture_order"] = summary["architecture"].map(architecture_sort_key)
        summary = summary.sort_values(["family_order", "architecture_order", "model"]).drop(
            columns=["family_order", "architecture_order"], errors="ignore"
        )
    return summary.reset_index(drop=True)


def build_pairwise_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Pairwise Welch two-sample t-tests between architectures, per model family,
    for each DeepEval score column. One row per (metric, family, comparison)."""
    rows = []
    families_present = [f for f in FAMILY_ORDER if f in df["family"].unique()]
    for col in SCORE_COLS:
        if col not in df.columns:
            continue
        for family in families_present:
            fam_df = df[df["family"] == family]
            archs_present = [a for a in ARCHITECTURE_ORDER if a in fam_df["architecture"].unique()]
            for a, b in combinations(archs_present, 2):
                va = pd.to_numeric(fam_df.loc[fam_df["architecture"] == a, col], errors="coerce").dropna().values
                vb = pd.to_numeric(fam_df.loc[fam_df["architecture"] == b, col], errors="coerce").dropna().values
                if len(va) < 2 or len(vb) < 2:
                    t_stat, p_value = float("nan"), float("nan")
                else:
                    t_stat, p_value = ttest_ind(va, vb, equal_var=False)
                rows.append({
                    "metric": col,
                    "family": family,
                    "architecture_a": a,
                    "architecture_b": b,
                    "n_a": len(va),
                    "n_b": len(vb),
                    "mean_a": float(va.mean()) if len(va) else None,
                    "mean_b": float(vb.mean()) if len(vb) else None,
                    "t": float(t_stat),
                    "p_value": float(p_value),
                    "significant": (p_value < 0.05) if p_value == p_value else False,  # NaN-safe
                })
    return pd.DataFrame(rows)


def plot_boxplots(df: pd.DataFrame, output_path: Path):
    """Side-by-side box plots: one panel per score, ordered by model family then architecture."""
    score_cols_present = [c for c in SCORE_COLS if c in df.columns]
    if not score_cols_present:
        print("No score columns found; skipping plot.")
        return

    meta = (
        df[["display_label", "family", "architecture"]]
        .drop_duplicates()
        .assign(
            family_order=lambda d: d["family"].map(family_sort_key),
            architecture_order=lambda d: d["architecture"].map(architecture_sort_key),
        )
        .sort_values(["family_order", "architecture_order", "display_label"])
    )
    if meta.empty:
        print("No model labels found; skipping plot.")
        return

    ordered_labels = meta["display_label"].tolist()
    ordered_families = meta["family"].tolist()
    ordered_architectures = meta["architecture"].tolist()

    positions = []
    family_centers = []
    family_breaks = []
    current_position = 0
    previous_family = None
    family_start = None
    for label, family in zip(ordered_labels, ordered_families):
        if previous_family is not None and family != previous_family:
            family_centers.append((previous_family, (family_start + positions[-1]) / 2))
            family_breaks.append(current_position)
            current_position += 1
            family_start = current_position
        elif family_start is None:
            family_start = current_position
        positions.append(current_position)
        current_position += 1
        previous_family = family
    if ordered_families:
        family_centers.append((ordered_families[-1], (family_start + positions[-1]) / 2))

    n_panels = len(score_cols_present)

    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5), squeeze=False)
    axes = axes[0]

    for ax, col in zip(axes, score_cols_present):
        panel_values = []
        for label in ordered_labels:
            series = pd.to_numeric(df.loc[df["display_label"] == label, col], errors="coerce").dropna()
            panel_values.append(series.values)

        bp = ax.boxplot(
            panel_values,
            positions=positions,
            tick_labels=ordered_labels,
            showmeans=True,
            patch_artist=True,
        )

        for box, architecture in zip(bp["boxes"], ordered_architectures):
            box.set_facecolor(ARCHITECTURE_COLORS.get(architecture, "#999999"))
            box.set_alpha(0.75)
            box.set_edgecolor("black")

        for median in bp["medians"]:
            median.set_color("black")
            median.set_linewidth(1.2)

        for whisker in bp["whiskers"]:
            whisker.set_color("#444444")
        for cap in bp["caps"]:
            cap.set_color("#444444")
        for mean in bp["means"]:
            mean.set_markerfacecolor("white")
            mean.set_markeredgecolor("black")

        ax.set_title(col.replace("_", " ").title())
        ax.set_ylabel("Score")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_xticklabels([])
        ax.tick_params(axis="x", length=0)

        for family, center in family_centers:
            ax.text(
                center,
                -0.16,
                family,
                ha="center",
                va="top",
                transform=ax.get_xaxis_transform(),
                fontsize=9,
            )

        for break_pos in family_breaks:
            ax.axvline(break_pos, color="#bbbbbb", linestyle="--", linewidth=0.8, zorder=0)

    legend_handles = [
        Patch(facecolor=ARCHITECTURE_COLORS[a], edgecolor="black", alpha=0.75, label=a)
        for a in ARCHITECTURE_ORDER
    ]
    fig.legend(handles=legend_handles, title="Architecture", loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=True)

    fig.suptitle("DeepEval score distributions by model family and architecture", fontsize=13)
    fig.subplots_adjust(bottom=0.14, right=0.83)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Wrote {output_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default=None, help="Folder containing result files")
    ap.add_argument("--output-dir", default=None, help="Where to write table + plot")
    ap.add_argument("--pattern", default=None,
                    help="Glob pattern (default '*.csv'; try '*.xlsx' if needed)")
    args = ap.parse_args()

    in_dir = Path(args.input_dir or DEFAULT_INPUT_DIR)
    out_dir = Path(args.output_dir or DEFAULT_OUTPUT_DIR)
    pattern = args.pattern or DEFAULT_PATTERN
    out_dir.mkdir(parents=True, exist_ok=True)

    files = discover_files(in_dir, pattern)
    if not files:
        print(f"No files matching {pattern} in {in_dir}")
        return

    print(f"Found {len(files)} result file(s):")
    for f in files:
        print(f"  - {f.name}")

    # Combine
    df = pd.concat([load_result_file(f) for f in files], ignore_index=True)

    # Summary table
    summary = build_summary_table(df)
    print("\n=== Summary table ===")
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(summary.round(3).to_string(index=False))

    summary.to_csv(out_dir / "summary_table.csv", index=False)
    summary.to_excel(out_dir / "summary_table.xlsx", index=False)
    print(f"\nWrote {out_dir / 'summary_table.csv'}  and  {out_dir / 'summary_table.xlsx'}")

    # Pairwise Welch t-tests between architectures, per family, per score
    pairwise = build_pairwise_tests(df)
    if not pairwise.empty:
        print("\n=== Pairwise Welch t-tests (per family, per metric) ===")
        with pd.option_context("display.width", 200, "display.max_columns", None):
            print(pairwise.round(4).to_string(index=False))
        pairwise.to_csv(out_dir / "pairwise_tests.csv", index=False)
        # also append as a second sheet of summary_table.xlsx
        with pd.ExcelWriter(out_dir / "summary_table.xlsx", engine="openpyxl",
                            mode="a", if_sheet_exists="replace") as writer:
            pairwise.to_excel(writer, sheet_name="Pairwise Tests", index=False)
        print(f"Wrote {out_dir / 'pairwise_tests.csv'} (and 'Pairwise Tests' sheet in summary_table.xlsx)")

    # Box plots
    plot_boxplots(df, out_dir / "score_boxplots.png")


if __name__ == "__main__":
    main()
