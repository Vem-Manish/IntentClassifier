from pathlib import Path
import ollama

# All file operations are sandboxed to this directory.
OUTPUT_DIR = Path("output").resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_path(name: str) -> Path:
    """
    Resolve *name* relative to OUTPUT_DIR and raise if it escapes the sandbox.
    This prevents directory-traversal attacks (e.g. '../../etc/passwd').
    """
    name = (name or "untitled.txt").strip().lstrip("/\\")
    resolved = (OUTPUT_DIR / name).resolve()

    if not str(resolved).startswith(str(OUTPUT_DIR)):
        raise ValueError(f"Unsafe path '{name}'. All writes must stay inside output/.")

    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def _chat(prompt: str, history: list | None = None, model: str = "llama3.1:8b", temperature: float = 0.2) -> str:
    """Send *prompt* to Ollama, optionally prepending conversation *history*."""
    messages = (history or []) + [{"role": "user", "content": prompt}]
    response = ollama.chat(model=model, messages=messages, options={"temperature": temperature})
    return response["message"]["content"].strip()


# ---------------------------------------------------------------------------
# Public tool functions
# ---------------------------------------------------------------------------

def create_file_tool(filename: str, content: str = "") -> str:
    path = _safe_path(filename)
    path.write_text(content, encoding="utf-8")
    return f"Created file: {path.name}"


def create_folder_tool(foldername: str) -> str:
    path = _safe_path(foldername)
    path.mkdir(parents=True, exist_ok=True)
    return f"Created folder: {path.name}"


def write_code_tool(query: str, language: str = "python", history: list | None = None) -> str:
    prompt = f"Write {language} code for: {query}. Return ONLY the raw code, no explanations or markdown fences."
    return _chat(prompt, history=history, model="qwen2.5-coder:7b")


def save_file_tool(filename: str, content: str) -> str:
    """Alias for create_file_tool — semantically clearer when saving generated content."""
    return create_file_tool(filename, content)


def summarize_text_tool(query: str, history: list | None = None) -> str:
    return _chat(f"Summarize this concisely in bullet points:\n\n{query}", history=history)


def general_chat_tool(query: str, history: list | None = None) -> str:
    return _chat(query, history=history, temperature=0.7)