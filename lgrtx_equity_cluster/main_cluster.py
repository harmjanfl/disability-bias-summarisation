from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from graph_cluster import graph

load_dotenv()


def validate_env() -> None:
    model_name = os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3.1-8B-Instruct")
    print(f"Using model: {model_name}")
    if not os.getenv("HF_TOKEN"):
        print("Warning: HF_TOKEN is not set. Gated Hugging Face models may fail to load.")


def normalize_prompt_objects(prompt_objects: Any) -> list[dict[str, Any]]:
    try:
        import pandas as pd

        if isinstance(prompt_objects, pd.DataFrame):
            return prompt_objects.to_dict(orient="records")
    except ImportError:
        pass

    if not isinstance(prompt_objects, list):
        raise ValueError("Prompt pickle must contain a list of dicts or a pandas DataFrame.")

    return prompt_objects


def extract_prompt(row: dict[str, Any]) -> str:
    messages = row.get("messages")
    if isinstance(messages, list):
        for msg in messages:
            if msg.get("role") == "user":
                return str(msg.get("content", ""))

    for key in ("prompt", "user_message"):
        if row.get(key):
            return str(row[key])
    return ""


def extract_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    messages = row.get("messages")
    if isinstance(messages, list):
        normalized = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            content = msg.get("content")
            if role and content is not None:
                normalized.append({"role": str(role), "content": str(content)})
        if normalized:
            return normalized

    fallback = []
    if row.get("system_message"):
        fallback.append({"role": "system", "content": str(row["system_message"])})
    prompt_text = extract_prompt(row)
    if prompt_text:
        fallback.append({"role": "user", "content": prompt_text})
    return fallback


def build_graph_input(
    prompt_text: str,
    input_messages: list[dict[str, str]],
    max_iterations: int,
    item_index: int,
) -> dict[str, Any]:
    return {
        "input_messages": input_messages,
        "max_iterations": max_iterations,
        "metadata": {
            "source": "cluster_batch_runner",
            "item_index": item_index,
            "user_prompt": prompt_text,
        },
    }


def build_response_record(
    row: dict[str, Any],
    prompt_text: str,
    input_messages: list[dict[str, str]],
    result: dict[str, Any],
) -> dict[str, Any]:
    record = dict(row)
    record.update(
        {
            "prompt": prompt_text,
            "input_messages": input_messages,
            "feedback_items": result.get("feedback_items", []),
            "labeled_feedback": result.get("labeled_feedback", []),
            "generated_response": result.get("generated_response", ""),
            "final_response": result.get("final_response", ""),
            "equity_metrics": result.get("equity_metrics", {}),
            "review_feedback": result.get("review_feedback", ""),
            "fairness_feedback": result.get("fairness_feedback", ""),
            "rewrite_instructions": result.get("rewrite_instructions", ""),
            "iteration": result.get("iteration", 0),
            "done": result.get("done", False),
            "raw_topic_labeler_output": result.get("raw_topic_labeler_output", ""),
            "raw_fairness_output": result.get("raw_fairness_output", ""),
            "errors": result.get("errors", []),
            "metadata": result.get("metadata", {}),
        }
    )
    return record


def run_graph_from_pickle(
    prompt_pickle_path: str,
    output_pickle_path: str,
    *,
    shard_rank: int = 0,
    shard_count: int = 1,
    max_iterations: int = 3,
):
    validate_env()

    if shard_rank < 0 or shard_count < 1 or shard_rank >= shard_count:
        raise ValueError("Invalid shard settings: require 0 <= shard_rank < shard_count and shard_count >= 1.")

    prompt_pickle_path = Path(prompt_pickle_path)
    output_pickle_path = Path(output_pickle_path)

    print(f"Loading prompts from pickle: {prompt_pickle_path}")
    with prompt_pickle_path.open("rb") as f:
        prompt_objects = normalize_prompt_objects(pickle.load(f))

    total_items = len(prompt_objects)
    shard_items = prompt_objects[shard_rank::shard_count]

    print(f"Worker {shard_rank}/{shard_count} processing {len(shard_items)} of {total_items} items")

    response_records = []

    for local_idx, row in enumerate(shard_items, start=1):
        if not isinstance(row, dict):
            print(f"Skipping non-dict item at shard index {local_idx}")
            continue

        global_idx = shard_rank + (local_idx - 1) * shard_count
        prompt_text = extract_prompt(row)
        input_messages = extract_messages(row)

        if not input_messages:
            print(f"Skipping item {global_idx}: no messages or prompt found.")
            continue

        if not prompt_text:
            prompt_text = "[No user message found]"

        print(f"Processing shard item {local_idx}/{len(shard_items)} (global index {global_idx})")

        try:
            result = graph.invoke(
                build_graph_input(
                    prompt_text=prompt_text,
                    input_messages=input_messages,
                    max_iterations=max_iterations,
                    item_index=global_idx,
                )
            )
            response_records.append(build_response_record(row, prompt_text, input_messages, result))
        except Exception as exc:
            response_records.append(
                {
                    **dict(row),
                    "prompt": prompt_text,
                    "input_messages": input_messages,
                    "generated_response": None,
                    "final_response": None,
                    "feedback_items": [],
                    "labeled_feedback": [],
                    "equity_metrics": {},
                    "review_feedback": None,
                    "fairness_feedback": None,
                    "rewrite_instructions": None,
                    "iteration": 0,
                    "done": False,
                    "raw_topic_labeler_output": None,
                    "raw_fairness_output": None,
                    "errors": [str(exc)],
                    "metadata": {},
                }
            )

    output_pickle_path.parent.mkdir(parents=True, exist_ok=True)
    with output_pickle_path.open("wb") as f:
        pickle.dump(response_records, f)

    print(f"Saved {len(response_records)} results to: {output_pickle_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the equity LangGraph pipeline on a cluster.")
    parser.add_argument("--prompt_pickle", required=True, help="Input pickle with prompts/messages")
    parser.add_argument("--out_pickle", required=True, help="Output pickle path")
    parser.add_argument("--shard_rank", type=int, default=0, help="This worker's shard index")
    parser.add_argument("--shard_count", type=int, default=1, help="Total number of shards")
    parser.add_argument("--max_iterations", type=int, default=3, help="Maximum review-rewrite loops")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_graph_from_pickle(
        prompt_pickle_path=args.prompt_pickle,
        output_pickle_path=args.out_pickle,
        shard_rank=args.shard_rank,
        shard_count=args.shard_count,
        max_iterations=args.max_iterations,
    )
