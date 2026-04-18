import json
import re
import ollama

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

INTENT_PROMPT = """
ROLE: You are the Orchestration Engine for 'WinVoice', a local Windows AI Agent.
TASK: Convert a user's natural-language command into a sequence of executable tool steps, returned as JSON.

JSON SCHEMA:
{
  "steps": [
    {
      "intent": "STR",
      "query": "STR",
      "meta": {
        "filename": "STR",
        "foldername": "STR",
        "language": "STR",
        "content_source": "STR"
      }
    }
  ]
}

INTENT RULES:
1. WRITE_CODE      — Generate code or scripts. Always follow with SAVE_FILE when the user mentions saving.
2. SAVE_FILE       — Write content to disk. Set meta.filename. Use content_source: "previous_step" when saving generated code.
3. CREATE_FILE     — Create an empty file. Set meta.filename.
4. CREATE_FOLDER   — Create a directory. Set meta.foldername.
5. SUMMARIZE_TEXT  — Condense information into bullet points.
6. GENERAL_CHAT    — Greetings, help requests, or anything non-actionable.

CHAINING EXAMPLES:

User: "Write a python calculator and save it as calc.py"
Output:
{
  "steps": [
    {"intent": "WRITE_CODE",  "query": "simple python calculator", "meta": {"language": "python"}},
    {"intent": "SAVE_FILE",   "query": "save code",                "meta": {"filename": "calc.py", "content_source": "previous_step"}}
  ]
}

User: "Summarize this long text and save the summary to notes.txt"
Output:
{
  "steps": [
    {"intent": "SUMMARIZE_TEXT", "query": "user provided text", "meta": {}},
    {"intent": "SAVE_FILE",      "query": "save summary",       "meta": {"filename": "notes.txt", "content_source": "previous_step"}}
  ]
}

CONSTRAINT: Return ONLY valid JSON. No preamble, no explanation, no markdown fences.
""".strip()

ALLOWED_INTENTS = {"CREATE_FILE", "CREATE_FOLDER", "WRITE_CODE", "SAVE_FILE", "SUMMARIZE_TEXT", "GENERAL_CHAT"}

FALLBACK = {"steps": [{"intent": "GENERAL_CHAT", "query": "fallback", "meta": {}}]}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_json(raw: str) -> str:
    """Pull the first {...} block out of *raw*, handling any surrounding prose."""
    raw = raw.strip()
    if raw.startswith("{") and raw.endswith("}"):
        return raw
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    return match.group(0) if match else json.dumps(FALLBACK)


def _normalize(data: dict, user_text: str) -> dict:
    """
    Sanitise parsed JSON so every step has the required keys and only
    uses known intents. Unknown intents fall back to GENERAL_CHAT.
    """
    steps = data.get("steps", []) if isinstance(data, dict) else []

    if not isinstance(steps, list) or not steps:
        return {"steps": [{"intent": "GENERAL_CHAT", "query": user_text, "meta": {}}]}

    clean = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        intent = str(step.get("intent", "GENERAL_CHAT")).upper().strip()
        if intent not in ALLOWED_INTENTS:
            intent = "GENERAL_CHAT"
        meta = step.get("meta", {})
        clean.append({
            "intent": intent,
            "query": str(step.get("query", user_text)).strip(),
            "meta": meta if isinstance(meta, dict) else {},
        })

    return {"steps": clean}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify_intents(user_text: str, history: list | None = None, model: str = "llama3.1:8b") -> dict:
    """
    Ask the LLM to decompose *user_text* into an ordered list of tool steps.
    Returns a normalised dict with a 'steps' key. Always succeeds — falls back
    to GENERAL_CHAT on any error.
    """
    messages = [
        {"role": "system", "content": INTENT_PROMPT},
        {"role": "user",   "content": f"Classify this: {user_text}"},
    ]

    try:
        response = ollama.chat(model=model, messages=messages, options={"temperature": 0})
        raw = response["message"]["content"]
        parsed = json.loads(_extract_json(raw))
        return _normalize(parsed, user_text)
    except Exception as exc:
        print(f"[classifier] Error: {exc}")
        return {"steps": [{"intent": "GENERAL_CHAT", "query": user_text, "meta": {}}]}