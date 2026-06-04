from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Any

from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI

from .state import AgentState, FeedbackItem
from .tools import TOPIC_TAXONOMY, analyze_equity

load_dotenv()


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing environment variable: {name}")
    return value


model = AzureChatOpenAI(
    azure_endpoint=_require_env("AZURE_OPENAI_ENDPOINT"),
    azure_deployment=_require_env("AZURE_OPENAI_DEPLOYMENT"),
    api_key=_require_env("AZURE_OPENAI_API_KEY"),
    api_version=_require_env("AZURE_OPENAI_API_VERSION"),
    temperature=0,
)


DEBUG_MODE = os.getenv("PIPELINE_DEBUG", "0").lower() in {"1", "true", "yes", "on"}


def _extract_json_block(text: str) -> Any:
    text = (text or "").strip()
    if not text:
        raise ValueError("Empty model response.")

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL)
    if fenced:
        return json.loads(fenced.group(1).strip())

    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start : end + 1])

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start : end + 1])

    raise ValueError("Could not parse JSON from model response.")


def _all_message_text(messages: list[dict[str, str]]) -> str:
    return "\n\n".join(str(msg.get("content", "")) for msg in messages if msg.get("content"))


def _first_user_message(messages: list[dict[str, str]]) -> str:
    for msg in messages:
        if msg.get("role") == "user":
            return str(msg.get("content", ""))
    return ""


def _canonical_topic(topic: str) -> str:
    allowed_topics = {name.lower(): name for name in TOPIC_TAXONOMY}
    return allowed_topics.get(topic.strip().lower(), "other")


def _stable_hash(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:12]


def _preview(text: str, limit: int = 240) -> str:
    compact = re.sub(r"\s+", " ", text or "").strip()
    return compact[:limit]


def _ensure_debug_metadata(state: AgentState) -> dict[str, Any]:
    metadata = dict(state.get("metadata", {}))
    debug = dict(metadata.get("debug", {}))
    debug.setdefault("enabled", DEBUG_MODE)
    debug.setdefault("summarizer_calls", [])
    debug.setdefault("fairness_calls", [])
    debug.setdefault("draft_chain_ok", True)
    metadata["debug"] = debug
    return metadata


def _append_debug_event(state: AgentState, bucket: str, event: dict[str, Any]) -> dict[str, Any]:
    metadata = _ensure_debug_metadata(state)
    debug = metadata["debug"]
    debug.setdefault(bucket, [])
    debug[bucket] = [*debug[bucket], event]
    metadata["debug"] = debug
    return metadata


def initialize_node(state: AgentState) -> AgentState:
    input_messages = state.get("input_messages", [])
    prompt = state.get("prompt") or _first_user_message(input_messages)
    source_text = _all_message_text(input_messages)
    max_iterations = int(state.get("max_iterations", 3) or 3)

    metadata = _ensure_debug_metadata(state)

    return {
        "prompt": prompt,
        "source_text": source_text,
        "max_iterations": max_iterations,
        "iteration": int(state.get("iteration", 0) or 0),
        "done": False,
        "generated_response": state.get("generated_response", ""),
        "final_response": state.get("final_response", ""),
        "feedback_items": state.get("feedback_items", []),
        "labeled_feedback": state.get("labeled_feedback", []),
        "equity_metrics": state.get("equity_metrics", {}),
        "fairness_feedback": state.get("fairness_feedback", ""),
        "review_feedback": state.get("review_feedback", ""),
        "rewrite_instructions": state.get("rewrite_instructions", ""),
        "raw_topic_labeler_output": state.get("raw_topic_labeler_output", ""),
        "raw_fairness_output": state.get("raw_fairness_output", ""),
        "errors": list(state.get("errors", [])),
        "metadata": metadata,
    }


FEEDBACK_PATTERN = re.compile(
    r"ID:\s*(?P<id>[^,\n]+)"
    r"(?:,\s*Date:\s*(?P<date>[^,\n]+))?"
    r"(?:,\s*Region:\s*(?P<region>[^,\n]+))?"
    r"(?:,\s*Sex:\s*(?P<sex>[^,\n]+))?"
    r"(?:,\s*Age:\s*(?P<age>[^,\n]+))?"
    r"(?:,\s*Other factors:\s*(?P<other_factors>[^\n]*?))?"
    r",\s*Feedback:\s*(?P<feedback>.*?)(?=\nID:|\Z)",
    flags=re.DOTALL | re.IGNORECASE,
)


def parse_feedback_node(state: AgentState) -> AgentState:
    source_text = state.get("source_text", "") or _all_message_text(state.get("input_messages", []))
    feedback_items: list[FeedbackItem] = []

    for match in FEEDBACK_PATTERN.finditer(source_text):
        item: FeedbackItem = {
            k: v.strip()
            for k, v in match.groupdict().items()
            if isinstance(v, str) and v.strip()
        }
        if item.get("feedback"):
            feedback_items.append(item)

    errors = list(state.get("errors", []))
    if not feedback_items:
        errors.append("No feedback items could be parsed from the provided messages.")

    return {
        "feedback_items": feedback_items,
        "errors": errors,
    }


def topic_labeler_node(state: AgentState) -> AgentState:
    feedback_items = state.get("feedback_items", [])
    if not feedback_items:
        return {"labeled_feedback": []}

    compact_items = [
        {
            "id": item.get("id", ""),
            "region": item.get("region", ""),
            "feedback": item.get("feedback", ""),
        }
        for item in feedback_items
    ]

    messages = [
        {
            "role": "system",
            "content": (
                "You are a humanitarian feedback topic labeler. "
                "Assign exactly one topic to each feedback item. "
                "Use only the allowed topics below and return valid JSON only.\n\n"
                f"Allowed topics: {', '.join(TOPIC_TAXONOMY)}\n\n"
                "Return a JSON array where each item has keys: id, topic."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(compact_items, ensure_ascii=False),
        },
    ]

    response = model.invoke(messages)
    raw_output = response.content

    label_map: dict[str, str] = {}
    try:
        parsed = _extract_json_block(raw_output)
        if isinstance(parsed, list):
            for item in parsed:
                if not isinstance(item, dict):
                    continue
                item_id = str(item.get("id", "")).strip()
                topic = _canonical_topic(str(item.get("topic", "")))
                if item_id:
                    label_map[item_id] = topic
    except Exception:
        pass

    labeled_feedback: list[FeedbackItem] = []
    for item in feedback_items:
        topic = label_map.get(item.get("id", ""), "other")
        enriched = dict(item)
        enriched["topic"] = topic
        labeled_feedback.append(enriched)

    return {
        "labeled_feedback": labeled_feedback,
        "raw_topic_labeler_output": raw_output,
    }


def summarizer_node(state: AgentState) -> AgentState:
    labeled_feedback = state.get("labeled_feedback", [])
    user_prompt = state.get("prompt", "") or _first_user_message(state.get("input_messages", []))
    rewrite_instructions = state.get("rewrite_instructions", "")
    current_summary = state.get("generated_response", "")
    iteration = int(state.get("iteration", 0) or 0)
    task_mode = "revise" if rewrite_instructions and current_summary else "new_summary"

    feedback_for_prompt = [
        {
            "id": item.get("id", ""),
            "region": item.get("region", ""),
            "topic": item.get("topic", "other"),
            "feedback": item.get("feedback", ""),
        }
        for item in labeled_feedback
    ]

    incoming_hash = _stable_hash(current_summary)
    previous_output_hash = state.get("metadata", {}).get("debug", {}).get("last_summary_output_hash")
    draft_chain_ok = True
    if task_mode == "revise" and previous_output_hash and incoming_hash != previous_output_hash:
        draft_chain_ok = False

    messages = [
        {
            "role": "system",
            "content": (
                "You are a humanitarian analysis assistant.\n\n"
                "Your task depends on whether revision instructions are provided.\n\n"
                "--- MODE 1: NEW SUMMARY ---\n"
                "If task_mode is new_summary:\n"
                "- Write a concise, evidence-based summary that answers the user request.\n"
                "- Ground every claim in the provided feedback items.\n"
                "- Do not invent groups, risks, or evidence that do not appear in the data.\n\n"
                "--- MODE 2: REVISION ---\n"
                "If task_mode is revise:\n"
                "- You are revising an existing draft.\n"
                "- Treat current_summary as the only draft to edit.\n"
                "- Apply ONLY the requested changes.\n"
                "- Do NOT rewrite the entire summary.\n"
                "- Do NOT change content unrelated to the instructions.\n"
                "- Make the smallest possible edits needed.\n\n"
                "Revision rules:\n"
                "- Follow the revision instructions exactly.\n"
                "- Do not independently rebalance topics.\n"
                "- Do not introduce new topics or evidence.\n"
                "- Do not expand or reduce content beyond what is requested.\n"
                "- Limit edits to the specific topics mentioned.\n"
                "- Preserve tone, structure, and all other content.\n"
                "- When applying ADD AFTER / ADD BEFORE / ADD instructions, insert the new content as a STAND-ALONE sentence. Never merge it into an adjacent sentence or paragraph.\n\n"
                "Sentence rules (apply in BOTH modes):\n"
                "- Each sentence must address exactly ONE topic.\n"
                "- Do not join multiple topics in one sentence using 'and', 'while', 'additionally', semicolons, or em-dashes.\n"
                "- Keep each sentence at most 35 words.\n"
                "- Do not start sentences with bare enumerators like '1.', '2.', '3.' Use plain prose.\n\n"
                "General rules:\n"
                "- Ground every claim in the provided feedback items.\n"
                "- Do not hallucinate or generalize beyond the data.\n\n"
                "Output format:\n"
                "- Return valid JSON only with keys: mode, source_draft_hash, revised_summary.\n"
                "- revised_summary MUST be a JSON array of strings, where each element is exactly ONE sentence.\n"
                "- Do not include newline characters, bullet markers, or numeric prefixes inside any element."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Task mode: {task_mode}\n"
                f"Iteration: {iteration}\n"
                f"Source draft hash: {incoming_hash}\n\n"
                f"User request:\n{user_prompt}\n\n"
                + (f"Current summary:\n{current_summary}\n\n" if current_summary else "")
                + (f"Revision instructions:\n{rewrite_instructions}\n\n" if rewrite_instructions else "")
                + "Feedback items with topics:\n"
                + json.dumps(feedback_for_prompt, ensure_ascii=False)
            ),
        },
    ]

    response = model.invoke(messages)
    raw_output = response.content
    generated_response = raw_output
    echoed_mode = task_mode
    echoed_hash = incoming_hash
    errors = list(state.get("errors", []))

    try:
        parsed = _extract_json_block(raw_output)
        if isinstance(parsed, dict):
            revised = parsed.get("revised_summary", generated_response)
            if isinstance(revised, list):
                revised_sentences = [str(s).strip() for s in revised if str(s).strip()]
                generated_response = " ".join(revised_sentences)
                state["summary_sentences_structured"] = revised_sentences
            else:
                generated_response = str(revised).strip()
                state["summary_sentences_structured"] = None
            echoed_mode = str(parsed.get("mode", echoed_mode)).strip() or task_mode
            echoed_hash = str(parsed.get("source_draft_hash", echoed_hash)).strip() or incoming_hash
    except Exception:
        if DEBUG_MODE:
            errors.append("Summarizer response was not valid JSON; using raw text as generated_response.")
    
    output_hash = _stable_hash(generated_response)
    mode_mismatch = echoed_mode != task_mode
    hash_mismatch = task_mode == "revise" and echoed_hash != incoming_hash
    if mode_mismatch:
        errors.append(
            f"Summarizer mode mismatch: expected {task_mode}, got {echoed_mode}."
        )
    if hash_mismatch:
        errors.append(
            f"Summarizer source_draft_hash mismatch: expected {incoming_hash}, got {echoed_hash}."
        )

    debug_event = {
        "iteration": iteration,
        "task_mode": task_mode,
        "input_summary_hash": incoming_hash,
        "expected_previous_output_hash": previous_output_hash,
        "draft_chain_ok": draft_chain_ok,
        "echoed_mode": echoed_mode,
        "echoed_hash": echoed_hash,
        "output_summary_hash": output_hash,
        "rewrite_instructions_preview": _preview(rewrite_instructions),
        "input_summary_preview": _preview(current_summary),
        "output_summary_preview": _preview(generated_response),
    }
    metadata = _append_debug_event(state, "summarizer_calls", debug_event)
    metadata["debug"]["last_summary_output_hash"] = output_hash
    metadata["debug"]["draft_chain_ok"] = metadata["debug"].get("draft_chain_ok", True) and draft_chain_ok

    return {
        "generated_response": generated_response,
        "final_response": generated_response,
        "summary_sentences_structured": revised_sentences if isinstance(revised, list) else None,
        "errors": errors,
        "metadata": metadata,
    }


def representation_analyzer_node(state: AgentState) -> AgentState:
    labeled_feedback = state.get("labeled_feedback", [])
    summary = state.get("generated_response", "")
    structured = state.get("summary_sentences_structured") or []
    # Minority-aware acceptance thresholds. These defaults make the loop
    # require full coverage of DIS-prefixed topics while accepting partial
    # coverage of GEN-prefixed topics. Override from Studio's Input panel.
    min_minority = float(state.get("min_minority_visibility_rate", 1.0))
    min_majority = float(state.get("min_majority_visibility_rate", 0.50))
    payload = {
        "labeled_feedback": labeled_feedback,
        "summary": summary,
        "min_evidence_threshold": 0.40,
        "ambiguity_margin": 0.02,
        "min_minority_visibility_rate": min_minority,
        "min_majority_visibility_rate": min_majority,
    }
    if structured:
        payload["pre_split_sentences"] = structured
    metrics = analyze_equity.invoke(payload)
    return {"equity_metrics": metrics}


def fairness_reviewer_node(state: AgentState) -> AgentState:
    metrics = state.get("equity_metrics", {})
    generated_response = state.get("generated_response", "")
    iteration = int(state.get("iteration", 0) or 0)
    max_iterations = int(state.get("max_iterations", 3) or 3)
    summary_hash = _stable_hash(generated_response)

    messages = [
        {
            "role": "system",
            "content": (
                "You are the equity reviewer in a humanitarian summarization pipeline. "
                "Your job is to interpret topic-level equity metrics and produce revision guidance for another LLM summarizer. "
                "Do not answer the user request directly. Do not introduce new evidence, new topics, or unsupported claims.\n\n"
                "Equity standard (gated hard mention):\n"
                "- The core equity goal is VISIBILITY of the minority class: every topic in the input that represents "
                "the minority group must receive at least one clearly assigned sentence in the summary.\n"
                "- The minority class consists of topics whose identifier begins with 'dis' (disability-related). "
                "All other topics (typically 'gen'-prefixed) are the majority class.\n"
                "- A summary sentence is hard-assigned to a topic only when its best topic affinity score meets the "
                "minimum evidence threshold (theta) AND exceeds the second-best affinity score by the ambiguity margin (delta).\n"
                "- invisible_topics are topics present in the input but receiving zero hard-assigned summary sentences.\n"
                "- visible_topics are topics that have at least one assigned sentence and are considered equitably covered.\n"
                "- minority_visibility_rate is the share of minority topics in the input that are visible. "
                "majority_visibility_rate is the equivalent for majority topics.\n"
                "- within_range is True when minority_visibility_rate >= min_minority_visibility_rate AND "
                "majority_visibility_rate >= min_majority_visibility_rate. This is the formal pass condition.\n"
                "- unassigned_summary_count tells you how many summary sentences were too ambiguous to assign to any topic.\n"
                "- Use warnings about fallback embeddings as uncertainty signals and mention them when relevant.\n\n"
                "Minority topic priority:\n"
                "- Topics whose identifier begins with 'dis' represent a minority group (people with disabilities) and "
                "are the primary focus of this equity pipeline.\n"
                "- When ANY minority topic appears in invisible_topics, address it BEFORE any majority topic this iteration.\n"
                "- If multiple minority topics are invisible, address all of them in the same iteration even if it means "
                "producing more than the usual number of rewrite_instructions.\n"
                "- Never set accept=true while any minority topic remains invisible, unless the same minority topic has "
                "stayed invisible across two or more consecutive iterations (loop is provably stuck).\n\n"
                "Revision strategy:\n"
                "- Your rewrite guidance will be read by another LLM, so it must be concrete, minimal, and easy to follow.\n"
                "- Prefer the smallest possible edit that makes an invisible topic visible.\n"
                "- Do not ask for a full rewrite.\n"
                "- After all invisible minority topics are addressed, you may request edits for up to 2 invisible majority topics per iteration.\n"
                "- Preserve all other content unless it must be minimally shortened to make room.\n"
                "- Avoid introducing new ambiguity: new sentences should clearly and specifically address the target topic.\n"
                "- A topic only needs one clear, unambiguous sentence to become visible.\n\n"
                "Acceptance criteria (decide the value of `accept`):\n"
                "- Set accept=true when within_range is true (both class thresholds satisfied).\n"
                "- Set accept=true when minority_visibility_rate >= min_minority_visibility_rate AND iteration >= max_iterations - 1 "
                "(graceful exit before the cap, but never sacrifice minority coverage).\n"
                "- Set accept=true when the previous iteration already attempted to address the same invisible topics and "
                "visibility_rate_when_present did not improve (loop is stuck; do not keep trying).\n"
                "- Set accept=true when adding more sentences would noticeably hurt readability or duplicate existing content, "
                "AND minority_visibility_rate is already at or above its threshold.\n"
                "- Otherwise set accept=false and emit focused rewrite_instructions for the highest-leverage invisible topics, "
                "minority topics first.\n"
                "- Important: `accept` and `within_range` are related but distinct. within_range is the strict mathematical pass "
                "condition; accept is your editorial judgement about whether further iteration is worthwhile. accept must be a "
                "strict superset of within_range (anything within_range implies accept=true).\n\n"
                "Rewrite instruction rules:\n"
                "- The downstream summarizer does not understand topic labels well; map each topic request to concrete sentence edits.\n"
                "- Each bullet must specify exact text operations using one of: KEEP, DELETE, REPLACE, ADD AFTER, or ADD BEFORE.\n"
                "- For DELETE or REPLACE, quote the exact sentence(s) from the current summary to edit.\n"
                "- For ADD AFTER/ADD BEFORE, quote the exact anchor sentence from the current summary where the new sentence should be inserted.\n"
                "- For ADD, provide one short replacement/addition sentence grounded in existing evidence only.\n"
                "- Keep each bullet minimal and local (one sentence-level change per bullet).\n"
                "- If exact sentence quoting is impossible, say so briefly and provide the closest unique anchor phrase.\n"
                "- Limit edits to only the specified topics; do not imply global rewriting.\n"
                "- When condensing, remove redundancy only and do NOT delete distinct facts or criteria.\n"
                "- Address each invisible topic with a SEPARATE bullet. Never request a single sentence that covers two topics at once.\n\n"
                "New-sentence quality rules (apply to every ADD / ADD AFTER / ADD BEFORE / REPLACE bullet):\n"
                "- The new sentence must be a single complete declarative sentence of at most 25 words.\n"
                "- Ground the sentence in the labeled_feedback items whose topic equals the target invisible topic. Reuse concrete nouns and phrasing that already appear in those items so the sentence is lexically close to its source evidence.\n"
                "- The sentence must clearly signal the target topic. Treat the topic identifier as a hint: split it on whitespace/underscores, use those words (or close synonyms drawn from the relevant feedback items) explicitly in the sentence.\n"
                "- Cite at least one feedback item id from labeled_feedback that supports the sentence.\n"
                "- Must NOT contain coordinating conjunctions ('and', 'or', 'but') joining two independent clauses.\n"
                "- Must NOT contain semicolons or em-dashes that bundle multiple topics together.\n"
                "- Must NOT start with an enumerator ('1.', '2.', '3.', 'First,', 'Second,').\n\n"
                "Output requirements:\n"
                "- Return valid JSON only.\n"
                "- Use exactly these keys: accept, review_feedback, rewrite_instructions, invisible_topics, metric_interpretation.\n"
                "- rewrite_instructions must contain no more than 4 bullet-like actions.\n"
                "- Each instruction must describe a minimal, bounded, sentence-level edit with explicit target text or anchor text.\n"
                "- invisible_topics should list only the top-priority topics you want addressed, not every invisible topic.\n"
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "iteration": iteration,
                    "max_iterations": max_iterations,
                    "summary_hash": summary_hash,
                    "summary": generated_response,
                    "equity_metrics": metrics,
                    "labeled_feedback": state.get("labeled_feedback", []),

                },
                ensure_ascii=False,
            ),
        },
    ]

    response = model.invoke(messages)
    raw_output = response.content

    accept = bool(metrics.get("within_range", False))
    review_feedback = "Equity appears acceptable: all input topics are visible in the summary."
    rewrite_instructions = ""
    invisible = list(metrics.get("invisible_topics", []))

    try:
        parsed = _extract_json_block(raw_output)
        if isinstance(parsed, dict):
            review_feedback = str(parsed.get("review_feedback", review_feedback))
            rewrite_instructions = str(parsed.get("rewrite_instructions", rewrite_instructions))
            accept = bool(parsed.get("accept", accept))
            invisible = parsed.get("invisible_topics", invisible) or invisible
    except Exception:
        if invisible:
            review_feedback = (
                f"Invisible topics (present in input but not covered in summary): {invisible}. "
                "The next draft should add at least one clear, topic-specific sentence for each invisible topic, "
                "grounded in the source feedback."
            )
            rewrite_instructions = (
                "Revise the summary by adding one focused, unambiguous sentence per invisible topic, "
                "grounded in the source feedback. Do not introduce unsupported claims."
            )

    next_iteration = iteration + 1
    done = accept or next_iteration >= max_iterations

    debug_event = {
        "iteration": iteration,
        "summary_hash_reviewed": summary_hash,
        "accept": accept,
        "invisible_topics": invisible,
        "visibility_rate_when_present": metrics.get("visibility_rate_when_present"),
        "rewrite_instructions_preview": _preview(rewrite_instructions),
    }
    metadata = _append_debug_event(state, "fairness_calls", debug_event)

    return {
        "review_feedback": review_feedback,
        "fairness_feedback": review_feedback,
        "rewrite_instructions": rewrite_instructions,
        "raw_fairness_output": raw_output,
        "iteration": next_iteration,
        "done": done,
        "final_response": generated_response,
        "metadata": metadata,
    }


def finalize_node(state: AgentState) -> AgentState:
    metrics = state.get("equity_metrics", {})
    metadata = _ensure_debug_metadata(state)
    debug = metadata.get("debug", {})

    return {
        "final_response": state.get("generated_response", ""),
        "metadata": {
            **metadata,
            "studio_summary": {
                "iterations_completed": state.get("iteration", 0),
                "invisible_topics": metrics.get("invisible_topics", []),
                "visible_topics": metrics.get("visible_topics", []),
                "visibility_rate_when_present": metrics.get("visibility_rate_when_present", 0.0),
                "minority_visible_topics": metrics.get("minority_visible_topics", []),
                "minority_invisible_topics": metrics.get("minority_invisible_topics", []),
                "minority_visibility_rate": metrics.get("minority_visibility_rate", 1.0),
                "majority_visible_topics": metrics.get("majority_visible_topics", []),
                "majority_invisible_topics": metrics.get("majority_invisible_topics", []),
                "majority_visibility_rate": metrics.get("majority_visibility_rate", 1.0),
                "min_minority_visibility_rate": metrics.get("min_minority_visibility_rate"),
                "min_majority_visibility_rate": metrics.get("min_majority_visibility_rate"),
                "unassigned_summary_count": metrics.get("unassigned_summary_count", 0),
                "within_range": metrics.get("within_range", False),
                "theta": metrics.get("theta"),
                "delta": metrics.get("delta"),
            },
            "debug": {
                **debug,
                "final_summary_hash": _stable_hash(state.get("generated_response", "")),
                "errors_count": len(state.get("errors", [])),
            },
        },
    }


def route_after_review(state: AgentState) -> str:
    return "finalize" if state.get("done", False) else "summarizer"
