from tools import (
    create_file_tool,
    create_folder_tool,
    write_code_tool,
    save_file_tool,
    summarize_text_tool,
    general_chat_tool,
)

# Intents that write to disk — these require human confirmation.
FILE_WRITING_INTENTS = {"CREATE_FILE", "CREATE_FOLDER", "SAVE_FILE"}

# Max characters to send to Ollama for summarization.
# llama3.1:8b has a ~8k token context; ~4 chars/token → ~24k chars safe limit.
# We stay well under to leave room for the system prompt and history.
MAX_SUMMARIZE_CHARS = 12_000


def needs_confirmation(steps: list) -> bool:
    """Return True if any step in the plan will write to disk."""
    return any(s.get("intent") in FILE_WRITING_INTENTS for s in steps)


def describe_file_ops(steps: list) -> str:
    """Human-readable summary of disk-writing steps for the confirm panel."""
    lines = []
    for step in steps:
        intent = step.get("intent", "")
        meta   = step.get("meta") or {}
        if intent == "CREATE_FILE":
            lines.append(f"📄 Create file: **{meta.get('filename', 'new_file.txt')}**")
        elif intent == "CREATE_FOLDER":
            lines.append(f"📁 Create folder: **{meta.get('foldername', 'new_folder')}**")
        elif intent == "SAVE_FILE":
            lines.append(f"💾 Save file: **{meta.get('filename', 'output.txt')}**")
    return "\n".join(lines) if lines else ""


def _truncate(text: str, max_chars: int = MAX_SUMMARIZE_CHARS) -> str:
    """
    Truncate text to max_chars, appending a note so the LLM knows it was cut.
    This prevents context-window overflow crashes with very large inputs.
    """
    if len(text) <= max_chars:
        return text
    cutoff = text[:max_chars]
    # Try to cut at the last sentence boundary so we don't slice mid-word.
    last_period = max(cutoff.rfind(". "), cutoff.rfind(".\n"))
    if last_period > max_chars * 0.8:
        cutoff = cutoff[: last_period + 1]
    return cutoff + "\n\n[... text truncated for summarization ...]"


def execute_steps(steps: list, original_text: str, history: list | None = None) -> tuple[str, str]:
    """
    Run each classified step in order and collect logs and outputs.

    Returns
    -------
    logs    : newline-joined numbered list of executed intents.
    outputs : double-newline-joined results from each tool call.
    """
    history = history or []
    logs: list[str] = []
    outputs: list[str] = []

    previous_output = ""

    for i, step in enumerate(steps, start=1):
        intent = step.get("intent", "GENERAL_CHAT")
        query  = step.get("query", original_text)
        meta   = step.get("meta") or {}

        if intent == "CREATE_FILE":
            result = create_file_tool(meta.get("filename", "new_file.txt"), content="")

        elif intent == "CREATE_FOLDER":
            result = create_folder_tool(meta.get("foldername", "new_folder"))

        elif intent == "WRITE_CODE":
            result = write_code_tool(query, language=meta.get("language", "python"), history=history)
            previous_output = result

        elif intent == "SAVE_FILE":
            use_previous = meta.get("content_source") == "previous_step" and previous_output
            content = previous_output if use_previous else original_text
            result = save_file_tool(meta.get("filename", "output.txt"), content)

        elif intent == "SUMMARIZE_TEXT":
            # Use the full original text (bug fix), but truncate if too large
            # to prevent Ollama context-window overflow crashes.
            safe_text = _truncate(original_text)
            result = summarize_text_tool(safe_text, history=history)
            previous_output = result

        else:
            result = general_chat_tool(query, history=history)

        logs.append(f"{i}. {intent}")
        outputs.append(result)

    return "\n".join(logs), "\n\n".join(outputs)