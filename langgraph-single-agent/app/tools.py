from langchain_core.tools import tool


@tool
def calculator(expression: str) -> str:
    """Evaluate a simple arithmetic expression like '2 + 2 * 10'."""
    allowed_chars = set("0123456789+-*/(). ")
    if not set(expression) <= allowed_chars:
        raise ValueError("Expression contains unsupported characters.")

    try:
        result = eval(expression, {"__builtins__": {}}, {})
    except Exception as e:
        raise ValueError(f"Could not evaluate expression: {e}")

    return str(result)