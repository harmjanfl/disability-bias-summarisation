import os
import pickle
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from app.graph import graph

load_dotenv()


def validate_env():
    required_vars = [
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_DEPLOYMENT",
        "AZURE_OPENAI_API_VERSION",
    ]

    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        raise RuntimeError(
            f"Missing environment variables: {', '.join(missing)}. "
            "Add them to your environment or .env file."
        )


def normalize_prompt_objects(prompt_objects):
    try:
        import pandas as pd
        if isinstance(prompt_objects, pd.DataFrame):
            return prompt_objects.to_dict(orient="records")
    except ImportError:
        pass

    if not isinstance(prompt_objects, list):
        raise ValueError(
            "Prompt pickle must contain a list of dicts or a pandas DataFrame."
        )

    return prompt_objects


def extract_prompt(row: dict) -> str:
    messages = row.get("messages")
    if isinstance(messages, list):
        for msg in messages:
            if msg.get("role") == "user":
                return str(msg.get("content", ""))

    # Backward-compatible fallback for older pickles without a messages column.
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

    # Backward-compatible fallback for rows that only have prompt/user_message.
    prompt_text = extract_prompt(row)
    if prompt_text:
        return [{"role": "user", "content": prompt_text}]

    return []


def run_graph_from_pickle(prompt_pickle_path: str, output_pickle_path: str):
    validate_env()

    prompt_pickle_path = Path(prompt_pickle_path)
    output_pickle_path = Path(output_pickle_path)

    print(f"Loading prompts from pickle: {prompt_pickle_path}")

    with open(prompt_pickle_path, "rb") as f:
        prompt_objects = pickle.load(f)

    prompt_objects = normalize_prompt_objects(prompt_objects)

    #prompt_objects = prompt_objects[150:154]  # For testing, limit to first 4 prompts. Remove for full run.

    response_records = []

    for idx, row in enumerate(prompt_objects, start=1):
        if not isinstance(row, dict):
            print(f"Skipping non-dict item at index {idx}")
            continue

        prompt_text = extract_prompt(row)
        input_messages = extract_messages(row)

        if not input_messages:
            print(f"Skipping item {idx}: no messages/prompt found.")
            continue

        if not prompt_text:
            prompt_text = "[No user message found]"

        selected_rows = row.get("selected_rows")

        print(f"Processing item {idx}/{len(prompt_objects)}")

        try:
            result = graph.invoke(
                {
                    "prompt": prompt_text,
                    "input_messages": input_messages,
                    "generated_response": "",
                    "review_feedback": "",
                    "final_response": "",
                },
                config={"configurable": {"thread_id": f"bias-review-{idx}"}},
            )

            response_records.append(
                {
                    "prompt": prompt_text,
                    "selected_rows": selected_rows,
                    "input_messages": input_messages,
                    "generated_response": result["generated_response"],
                    "review_feedback": result["review_feedback"],
                    "final_response": result["final_response"],
                }
            )

        except Exception as e:
            response_records.append(
                {
                    "prompt": prompt_text,
                    "selected_rows": selected_rows,
                    "input_messages": input_messages,
                    "generated_response": None,
                    "review_feedback": None,
                    "final_response": None,
                    "error": str(e),
                }
            )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_pickle_path = output_pickle_path.with_stem(
        f"{output_pickle_path.stem}_{timestamp}"
    )

    with open(output_pickle_path, "wb") as f:
        pickle.dump(response_records, f)

    print(f"Saved results to: {output_pickle_path}")


if __name__ == "__main__":
    prompt_file = (
        Path("app")
        / "Full_synthetic_GPT_prompts_50_combined_0-1-2-5-10-20.pkl"
    )
    output_file = Path("app") / "grr_equity_outputs" / "output.pkl"

    run_graph_from_pickle(
        prompt_pickle_path=str(prompt_file),
        output_pickle_path=str(output_file),
    )