import argparse
import os
import pickle
from pathlib import Path

from dotenv import load_dotenv

from graph_cluster import graph

load_dotenv()


def validate_env() -> None:
    model_name = os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3.1-8B-Instruct")
    print(f"Using model: {model_name}")
    if not os.getenv("HF_TOKEN"):
        print("Warning: HF_TOKEN is not set. Gated Hugging Face models may fail to load.")



def normalize_prompt_objects(prompt_objects):
    try:
        import pandas as pd

        if isinstance(prompt_objects, pd.DataFrame):
            return prompt_objects.to_dict(orient="records")
    except ImportError:
        pass

    if not isinstance(prompt_objects, list):
        raise ValueError("Prompt pickle must contain a list of dicts or a pandas DataFrame.")

    return prompt_objects



def extract_prompt(row: dict) -> str:
    messages = row.get("messages")
    if isinstance(messages, list):
        for msg in messages:
            if msg.get("role") == "user":
                return str(msg.get("content", ""))

    if "prompt" in row and row["prompt"]:
        return str(row["prompt"])
    if "user_message" in row and row["user_message"]:
        return str(row["user_message"])
    return ""



def extract_messages(row: dict) -> list[dict[str, str]]:
    messages = row.get("messages")
    if isinstance(messages, list):
        normalized_messages = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            content = msg.get("content")
            if role and content is not None:
                normalized_messages.append({"role": str(role), "content": str(content)})
        if normalized_messages:
            return normalized_messages

    system_text = row.get("system_message")
    prompt_text = extract_prompt(row)
    fallback_messages = []
    if system_text:
        fallback_messages.append({"role": "system", "content": str(system_text)})
    if prompt_text:
        fallback_messages.append({"role": "user", "content": prompt_text})
    return fallback_messages



def run_graph_from_pickle(
    prompt_pickle_path: str,
    output_pickle_path: str,
    *,
    shard_rank: int = 0,
    shard_count: int = 1,
):
    validate_env()

    if shard_rank < 0 or shard_count < 1 or shard_rank >= shard_count:
        raise ValueError("Invalid shard settings: require 0 <= shard_rank < shard_count and shard_count >= 1.")

    prompt_pickle_path = Path(prompt_pickle_path)
    output_pickle_path = Path(output_pickle_path)

    print(f"Loading prompts from pickle: {prompt_pickle_path}")
    with open(prompt_pickle_path, "rb") as f:
        prompt_objects = pickle.load(f)

    prompt_objects = normalize_prompt_objects(prompt_objects)
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
            print(f"Skipping item {global_idx}: no messages/prompt found.")
            continue

        if not prompt_text:
            prompt_text = "[No user message found]"

        print(f"Processing shard item {local_idx}/{len(shard_items)} (global index {global_idx})")

        try:
            result = graph.invoke(
                {
                    "prompt": prompt_text,
                    "input_messages": input_messages,
                    "generated_response": "",
                    "review_feedback": "",
                    "final_response": "",
                },
                config={"configurable": {"thread_id": f"bias-review-r{shard_rank}-i{global_idx}"}},
            )

            record = dict(row)
            record.update(
                {
                    "prompt": prompt_text,
                    "input_messages": input_messages,
                    "generated_response": result["generated_response"],
                    "review_feedback": result["review_feedback"],
                    "final_response": result["final_response"],
                }
            )
            response_records.append(record)
        except Exception as e:
            record = dict(row)
            record.update(
                {
                    "prompt": prompt_text,
                    "input_messages": input_messages,
                    "generated_response": None,
                    "review_feedback": None,
                    "final_response": None,
                    "error": str(e),
                }
            )
            response_records.append(record)

    output_pickle_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_pickle_path, "wb") as f:
        pickle.dump(response_records, f)

    print(f"Saved {len(response_records)} results to: {output_pickle_path}")



def parse_args():
    parser = argparse.ArgumentParser(description="Run LangGraph bias review pipeline on a cluster.")
    parser.add_argument("--prompt_pickle", required=True, help="Input pickle with prompts/messages")
    parser.add_argument("--out_pickle", required=True, help="Output pickle path")
    parser.add_argument("--shard_rank", type=int, default=0, help="This worker's shard index")
    parser.add_argument("--shard_count", type=int, default=1, help="Total number of shards")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_graph_from_pickle(
        prompt_pickle_path=args.prompt_pickle,
        output_pickle_path=args.out_pickle,
        shard_rank=args.shard_rank,
        shard_count=args.shard_count,
    )
