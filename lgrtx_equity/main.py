from __future__ import annotations

import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from app.graph import graph

load_dotenv()


REQUIRED_ENV_VARS = [
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_DEPLOYMENT",
    "AZURE_OPENAI_API_VERSION",
]


def validate_env() -> None:
    missing = [var for var in REQUIRED_ENV_VARS if not os.getenv(var)]
    if missing:
        raise RuntimeError(
            f"Missing environment variables: {', '.join(missing)}. Add them to your environment or .env file."
        )


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

    prompt_text = extract_prompt(row)
    if prompt_text:
        return [{"role": "user", "content": prompt_text}]

    return []


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
            "source": "batch_runner",
            "item_index": item_index,
            "user_prompt": prompt_text,
        },
    }


def build_langsmith_config(item_index: int, prompt_text: str) -> dict[str, Any]:
    truncated_prompt = (prompt_text[:120] + "…") if len(prompt_text) > 120 else prompt_text
    return {
        "run_name": f"lgtrx-bias-review-{item_index}",
        "tags": ["lgtrx", "bias-mitigation", "batch-run"],
        "metadata": {
            "item_index": item_index,
            "prompt_preview": truncated_prompt,
        },
        "configurable": {
            "thread_id": f"bias-review-{item_index}",
        },
    }


def build_response_record(
    row: dict[str, Any],
    prompt_text: str,
    input_messages: list[dict[str, str]],
    result: dict[str, Any],
) -> dict[str, Any]:
    return {
        "prompt": prompt_text,
        "selected_rows": row.get("selected_rows"),
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


def run_graph_from_pickle(
    prompt_pickle_path: str,
    output_pickle_path: str,
    max_iterations: int = 5,
    start: int | None = 150,
    stop: int | None = 151,
) -> Path:
    validate_env()

    prompt_pickle_path = Path(prompt_pickle_path)
    output_pickle_path = Path(output_pickle_path)

    print(f"Loading prompts from pickle: {prompt_pickle_path}")
    with prompt_pickle_path.open("rb") as f:
        prompt_objects = normalize_prompt_objects(pickle.load(f))

    if start is not None or stop is not None:
        prompt_objects = prompt_objects[start:stop]

    response_records = []

    for idx, row in enumerate(prompt_objects, start=1):
        if not isinstance(row, dict):
            print(f"Skipping non-dict item at index {idx}")
            continue

        prompt_text = extract_prompt(row)
        input_messages = extract_messages(row)
        if not input_messages:
            print(f"Skipping item {idx}: no messages or prompt found.")
            continue

        if not prompt_text:
            prompt_text = "[No user message found]"

        print(f"Processing item {idx}/{len(prompt_objects)}")
        try:
            result = graph.invoke(
                build_graph_input(
                    prompt_text=prompt_text,
                    input_messages=input_messages,
                    max_iterations=max_iterations,
                    item_index=idx,
                ),
                config=build_langsmith_config(idx, prompt_text),
            )
            response_records.append(build_response_record(row, prompt_text, input_messages, result))
        except Exception as exc:
            response_records.append(
                {
                    "prompt": prompt_text,
                    "selected_rows": row.get("selected_rows"),
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

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_pickle_path = output_pickle_path.with_stem(f"{output_pickle_path.stem}_{timestamp}")
    output_pickle_path.parent.mkdir(parents=True, exist_ok=True)
    with output_pickle_path.open("wb") as f:
        pickle.dump(response_records, f)

    print(f"Saved results to: {output_pickle_path}")
    return output_pickle_path


if __name__ == "__main__":
    prompt_file = Path("app") / "Full_synthetic_GPT_prompts_50_combined_0-1-2-5-10-20.pkl"
    output_file = Path("app") / "test_experiment" / "testing_output.pkl"

    run_graph_from_pickle(
        prompt_pickle_path=str(prompt_file),
        output_pickle_path=str(output_file),
        max_iterations=5,
        start=None,
        stop=None,
    )
