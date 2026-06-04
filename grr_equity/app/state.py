from typing_extensions import TypedDict


class AgentState(TypedDict):
    prompt: str
    input_messages: list[dict[str, str]]
    generated_response: str
    review_feedback: str
    final_response: str