import os
from functools import lru_cache
from typing import Any

import torch
from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from transformers import AutoModelForCausalLM, AutoTokenizer

from state import AgentState

load_dotenv()

DEFAULT_MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3.1-8B-Instruct")
DEFAULT_MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "512"))
DEFAULT_TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))
DEFAULT_TOP_P = float(os.getenv("TOP_P", "0.9"))


@lru_cache(maxsize=1)
def setup_model() -> tuple[Any, Any]:
    model_name = os.getenv("MODEL_NAME", DEFAULT_MODEL_NAME)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        token=os.getenv("HF_TOKEN") or True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto",
        token=os.getenv("HF_TOKEN") or True,
    ).eval()

    return tokenizer, model


def invoke_local_llm(
    messages: list[dict[str, str]],
    *,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
) -> str:
    tokenizer, model = setup_model()

    normalized_messages: list[dict[str, str]] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role", "")).strip()
        content = str(msg.get("content", "")).strip()
        if role and content:
            normalized_messages.append({"role": role, "content": content})

    if not normalized_messages:
        raise ValueError("No valid messages were provided to the model.")

    if tokenizer.chat_template is None:
        prompt_text = "\n".join(
            f"{m['role'].capitalize()}: {m['content']}" for m in normalized_messages
        ) + "\nAssistant:"
    else:
        prompt_text = tokenizer.apply_chat_template(
            normalized_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    inputs = tokenizer(prompt_text, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]

    do_sample = temperature > 0.0
    generate_kwargs = {
        **inputs,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generate_kwargs["temperature"] = temperature
        generate_kwargs["top_p"] = top_p

    with torch.no_grad():
        output_ids = model.generate(**generate_kwargs)

    gen_ids = output_ids[0, input_len:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def generator_node(state: AgentState):
    prompt = state["prompt"]
    input_messages = state.get("input_messages", [])

    if input_messages:
        messages = input_messages
    else:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful summarization assistant. Write a clear, accurate summary grounded only in the user's input. "
                    "Prioritize equitable representation rather than equal mention frequency. "
                    "If the input mentions vulnerable, marginalized, at-risk, underserved, or otherwise disproportionately affected groups, "
                    "increase the likelihood that they appear in the summary when they are relevant to the main content. "
                    "Do not invent groups, harms, or claims that are not supported by the input. "
                    "When such groups are mentioned, explicitly preserve their needs, risks, barriers, or perspectives in the summary instead of collapsing them into generic language. "
                    "Use salience-weighting: details about vulnerable groups should receive extra attention relative to baseline summarization, even if they are mentioned briefly, provided they are supported by the input. "
                    "If multiple groups are mentioned, give additional space to the most vulnerable or disproportionately impacted groups. "
                    "Be specific, concise, and natural."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

    response = invoke_local_llm(messages)
    return {"generated_response": response}



def reviewer_node(state: AgentState):
    prompt = state["prompt"]
    generated_response = state["generated_response"]

    messages = [
        {
            "role": "system",
            "content": (
                "You are a bias, fairness, and equity reviewer evaluating an LLM-generated summary. "
                "Your job is to review the summary against the original input and then write revision instructions for an LLM rewriter. "
                "Optimize for measurable equity gain over a baseline summary. "
                "Check whether vulnerable, marginalized, at-risk, underserved, or disproportionately affected groups mentioned in the original input are missing, minimized, generalized away, or framed less prominently than warranted. "
                "Prefer supported amplification over equal distribution: when the input mentions such groups, recommend increasing their visibility in the output if that would better reflect equity concerns in the source. "
                "Do not ask the rewriter to add any group, harm, or perspective that is not grounded in the original input. "
                "Flag overcorrection too: if the summary adds unsupported equity claims or overstates a group's role, say so. "
                "Use a structured review with concise bullet points under these headings: Missing Equity Signals, Harmful or Flattening Language, Overcorrections/Unsupported Additions, and Rewrite Instructions for the LLM Rewriter. "
                "In the rewrite instructions, name the specific sentence or omission to fix and explicitly tell the LLM rewriter how to change prominence, wording, or detail."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Original prompt:\n{prompt}\n\n"
                f"Generated summary:\n{generated_response}\n\n"
                "Review this summary for bias, fairness, and equity concerns."
            ),
        },
    ]

    response = invoke_local_llm(messages)
    return {"review_feedback": response}



def rewriter_node(state: AgentState):
    prompt = state["prompt"]
    generated_response = state["generated_response"]
    review_feedback = state["review_feedback"]

    messages = [
        {
            "role": "system",
            "content": (
                "You are a rewriting assistant. Rewrite an LLM-generated summary using review feedback written for an LLM rewriter. "
                "Keep the answer accurate, natural, and fully grounded in the original input. "
                "Your goal is to increase equity in the summary, not merely maintain equal representation. "
                "If the original input mentions vulnerable, marginalized, at-risk, underserved, or disproportionately affected groups, make sure they are explicitly represented in the rewritten summary when relevant. "
                "Apply supported emphasis: increase the prominence, specificity, and retention of those groups' needs, risks, barriers, or perspectives when the source supports it. "
                "Do not invent groups, impacts, or causal claims, and do not use generic boilerplate equity language unsupported by the input. "
                "Follow the review instructions precisely, resolve omissions first, then remove harmful or flattening phrasing, then tighten for clarity. "
                "Return only the rewritten response and do not mention the review or rewriting process."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Original prompt:\n{prompt}\n\n"
                f"Original generated response:\n{generated_response}\n\n"
                f"Reviewer feedback:\n{review_feedback}\n\n"
                "Please rewrite the response accordingly."
            ),
        },
    ]

    response = invoke_local_llm(messages)
    return {"final_response": response}


builder = StateGraph(AgentState)
builder.add_node("generator", generator_node)
builder.add_node("reviewer", reviewer_node)
builder.add_node("rewriter", rewriter_node)

builder.add_edge(START, "generator")
builder.add_edge("generator", "reviewer")
builder.add_edge("reviewer", "rewriter")
builder.add_edge("rewriter", END)

graph = builder.compile()
