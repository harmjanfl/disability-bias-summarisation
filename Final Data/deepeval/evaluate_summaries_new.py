"""
Evaluate model responses with DeepEval's SummarizationMetric and
HallucinationMetric, using your Azure OpenAI deployment as the judge.

Supports two pickle schemas:

  1) Baseline pickle (e.g. gpt_baseline_responses.pkl):
       row['system_message'] -> source feedback (after 'Here is the collected feedback:')
       row['user_message']   -> user prompt
       row['response']       -> model output

  2) Graph output pickle (produced by main.py):
       row['input_messages'] -> list[{role, content}]; role='system' carries the source
       row['prompt']         -> user prompt
       row['final_response'] -> model output  (or 'generated_response' if --use-generated)

Env vars required (same as your main.py):
    AZURE_OPENAI_API_KEY
    AZURE_OPENAI_ENDPOINT
    AZURE_OPENAI_DEPLOYMENT     # deployment name in Azure
    AZURE_OPENAI_API_VERSION
Optional:
    AZURE_OPENAI_MODEL          # underlying model name, e.g. 'gpt-4.1' (defaults to deployment name)

Usage:
    python evaluate_summaries.py --input-dir "Final Data/deepeval/response pickles"
    python evaluate_summaries.py --input out.pkl --output-dir "Final Data/deepeval"
    python evaluate_summaries.py --input out.pkl --use-generated   # score pre-review response
    python evaluate_summaries.py --input out.pkl --limit 5         # smoke test
"""

import argparse
import json
import os
import pickle
import re
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

# deepeval (external) imports are optional here so we can report missing deps
try:
    from deepeval.metrics import SummarizationMetric, HallucinationMetric
    from deepeval.models import AzureOpenAIModel
    from deepeval.test_case import LLMTestCase
    _DEEPEVAL_AVAILABLE = True
except Exception as e:  # pragma: no cover - runtime environment dependent
    SummarizationMetric = HallucinationMetric = AzureOpenAIModel = LLMTestCase = None
    _DEEPEVAL_AVAILABLE = False
    _DEEPEVAL_IMPORT_ERROR = e

load_dotenv()

# --------------------------- User-configurable defaults ------------------ #
# Edit these at the top of the file when running interactively in VS Code.
DEFAULTS = {
    "INPUT_DIR": "Final Data/deepeval/response pickles",  # folder of .pkl files to evaluate
    "INPUT": None,                                          # set to a .pkl file to run just one
    "OUTPUT_DIR": "Final Data/deepeval",                   # where .xlsx/.csv/.json files are written
    "OUTPUT_PREFIX": None,                                  # optional override for one-file mode
    "LIMIT": None,                                            # integer or None
    "USE_GENERATED": False,
    "RUN_SUMMARIZATION": True,
    "RUN_HALLUCINATION": True,
}


def check_dependencies():
    """Check and print missing Python packages used by this script."""
    import importlib

    reqs = {
        "pandas": "pandas",
        "dotenv": "python-dotenv",
        "tqdm": "tqdm",
        "deepeval": "deepeval",
    }
    missing = []
    for mod, pkg in reqs.items():
        try:
            importlib.import_module(mod)
        except Exception:
            missing.append(pkg)
    if missing:
        print("Missing packages:\n  ", "\n  ".join(missing))
        print("Install via: python -m pip install ", " ".join(sorted(set(missing))))
    else:
        print("All required packages appear installed.")

# --------------------------- field extraction --------------------------- #

FEEDBACK_LINE_RE = re.compile(r"^ID:\s*[^,]+,.*Feedback:\s*(.+)$", re.MULTILINE)
FEEDBACK_MARKER  = "Here is the collected feedback:"


def _strip_to_feedback(system_text: str) -> str:
    """Drop the instruction preamble; keep only the feedback corpus."""
    if FEEDBACK_MARKER in system_text:
        return system_text.split(FEEDBACK_MARKER, 1)[1].strip()
    return system_text.strip()


def get_source_text(row: dict) -> str:
    # schema 1: baseline pickle
    if row.get("system_message"):
        return _strip_to_feedback(row["system_message"])
    # schema 2: graph output pickle
    msgs = row.get("input_messages") or []
    for m in msgs:
        if isinstance(m, dict) and m.get("role") == "system":
            return _strip_to_feedback(str(m.get("content", "")))
    return ""


def get_user_prompt(row: dict) -> str:
    if row.get("user_message"):
        return str(row["user_message"])
    if row.get("prompt"):
        return str(row["prompt"])
    msgs = row.get("input_messages") or []
    for m in msgs:
        if isinstance(m, dict) and m.get("role") == "user":
            return str(m.get("content", ""))
    return ""


def get_output(row: dict, use_generated: bool) -> str:
    # graph output pickle has both generated_response (pre-review) and final_response (post-review)
    if use_generated and row.get("generated_response"):
        return str(row["generated_response"])
    if row.get("final_response"):
        return str(row["final_response"])
    if row.get("response"):  # baseline pickle
        return str(row["response"])
    return ""


def get_contexts(source_text: str) -> list[str]:
    """Each 'Feedback:' line becomes one context item for HallucinationMetric."""
    items = [m.group(1).strip() for m in FEEDBACK_LINE_RE.finditer(source_text)]
    return items if items else [source_text]


def normalize_records(obj) -> list[dict]:
    """Accept pickled list[dict] OR pandas DataFrame; return list[dict]."""
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, list):
        return obj
    raise TypeError(
        f"Expected list[dict] or pandas.DataFrame in pickle, got {type(obj).__name__}"
    )


# --------------------------- judge setup -------------------------------- #

def build_judge() -> object:
    required = ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT",
                "AZURE_OPENAI_DEPLOYMENT", "AZURE_OPENAI_API_VERSION"]
    missing = [v for v in required if not os.getenv(v)]
    if missing:
        raise RuntimeError(f"Missing Azure env vars: {', '.join(missing)}")

    deployment = os.environ["AZURE_OPENAI_DEPLOYMENT"]
    return AzureOpenAIModel(
        model=os.getenv("AZURE_OPENAI_MODEL", deployment),
        deployment_name=deployment,
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        base_url=os.environ["AZURE_OPENAI_ENDPOINT"],
        temperature=0,
    )


# --------------------------- main eval loop ----------------------------- #

def evaluate(records, judge, run_summarization, run_hallucination, use_generated):
    summ = (SummarizationMetric(threshold=0.5, model=judge, async_mode=False)
            if run_summarization else None)
    hall = (HallucinationMetric(threshold=0.5, model=judge, async_mode=False)
            if run_hallucination else None)

    rows = []
    for i, rec in enumerate(tqdm(records, desc="Evaluating")):
        if not isinstance(rec, dict):
            continue

        source_text = get_source_text(rec)
        user_input  = get_user_prompt(rec)
        output      = get_output(rec, use_generated=use_generated)
        contexts    = get_contexts(source_text)

        row = {
            "index"          : i,
            "iteration"      : rec.get("iteration"),
            "prompt_index"   : rec.get("prompt_index"),
            "synthetic_count": rec.get("synthetic_count"),
            "n_feedback"     : len(contexts),
            "output_len"     : len(output),
        }

        # skip rows that are unusable rather than crashing the run
        if not source_text or not output:
            row["skip_reason"] = "missing source or output"
            rows.append(row)
            continue

        if summ is not None:
            try:
                tc = LLMTestCase(input=source_text, actual_output=output)
                summ.measure(tc)
                row["summarization_score"]  = summ.score
                row["summarization_reason"] = summ.reason
            except Exception as e:
                row["summarization_score"]  = None
                row["summarization_reason"] = f"ERROR: {e}"

        if hall is not None:
            try:
                tc = LLMTestCase(input=user_input or "(no prompt)",
                                 actual_output=output, context=contexts)
                hall.measure(tc)
                row["hallucination_score"]  = hall.score
                row["hallucination_reason"] = hall.reason
            except Exception as e:
                row["hallucination_score"]  = None
                row["hallucination_reason"] = f"ERROR: {e}"

        rows.append(row)

    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default=None, help="Folder containing .pkl files to evaluate")
    ap.add_argument("--input", required=False, default=None, help="Path to a single .pkl file")
    ap.add_argument("--output-dir", default=None, help="Folder where outputs are written")
    ap.add_argument("--output-prefix", default=None, help="Optional output prefix for single-file mode")
    ap.add_argument("--limit",  type=int, default=None, help="Evaluate only first N rows")
    ap.add_argument("--use-generated", action="store_true",
                    help="For graph-output pickles, score 'generated_response' instead of 'final_response'")
    ap.add_argument("--no-summarization", action="store_true")
    ap.add_argument("--no-hallucination", action="store_true")
    args = ap.parse_args()

    check_dependencies()

    if not _DEEPEVAL_AVAILABLE:
        print("deepeval package could not be imported:", _DEEPEVAL_IMPORT_ERROR)
        print("Install deepeval (and dependencies) before running full evaluation.")
        judge = None
    else:
        judge = build_judge()
        print(f"Judge: Azure deployment '{os.getenv('AZURE_OPENAI_DEPLOYMENT')}' "
              f"(model={os.getenv('AZURE_OPENAI_MODEL', os.getenv('AZURE_OPENAI_DEPLOYMENT'))})")
    # apply defaults for easier VS Code runs
    input_path = args.input or DEFAULTS["INPUT"]
    input_dir = args.input_dir or DEFAULTS["INPUT_DIR"]
    output_dir = args.output_dir or DEFAULTS["OUTPUT_DIR"]
    output_prefix = args.output_prefix or DEFAULTS["OUTPUT_PREFIX"]
    limit = args.limit if args.limit is not None else DEFAULTS["LIMIT"]
    use_generated = args.use_generated or DEFAULTS["USE_GENERATED"]
    run_summarization = DEFAULTS["RUN_SUMMARIZATION"]
    if args.no_summarization:
        run_summarization = False
    run_hallucination = DEFAULTS["RUN_HALLUCINATION"]
    if args.no_hallucination:
        run_hallucination = False

    if input_path:
        input_files = [Path(input_path)]
    else:
        folder = Path(input_dir)
        if not folder.exists():
            print(f"Input folder not found: {folder}")
            return
        input_files = sorted(folder.glob("*.pkl"))
        if not input_files:
            print(f"No .pkl files found in {folder}")
            return

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    for input_file in input_files:
        with open(input_file, "rb") as f:
            records = pickle.load(f)
        records = normalize_records(records)
        if limit:
            records = records[: limit]
        print(f"Loaded {len(records)} records from {input_file}")

        rows = evaluate(
            records,
            judge=judge,
            run_summarization=run_summarization,
            run_hallucination=run_hallucination,
            use_generated=use_generated,
        )

        if output_prefix and len(input_files) == 1:
            out = Path(output_prefix)
            if not out.is_absolute() and out.parent == Path("."):
                out = output_root / out.name
        else:
            out = output_root / input_file.stem

        df = pd.DataFrame(rows)
        xlsx_path = out.with_suffix(".xlsx")
        csv_path = out.with_suffix(".csv")
        json_path = out.with_suffix(".json")

        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="Results", index=False)

        df.to_csv(csv_path, index=False)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False, default=str)

        print("\n=== Summary ===")
        for col in ("summarization_score", "hallucination_score"):
            if col in df.columns:
                s = pd.to_numeric(df[col], errors="coerce").dropna()
                if len(s):
                    print(f"{col:>22}: mean={s.mean():.3f}  min={s.min():.3f}  "
                          f"max={s.max():.3f}  n={len(s)}")
        skipped = df["skip_reason"].notna().sum() if "skip_reason" in df.columns else 0
        if skipped:
            print(f"Skipped (missing fields): {skipped}")
        print(f"\nWrote {xlsx_path}  {csv_path}  {json_path}")


if __name__ == "__main__":
    main()
