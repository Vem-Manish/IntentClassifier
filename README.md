# 🎙️ Voice-Controlled Local AI Agent with Intent Classification

A fully local, voice-controlled AI agent that listens to what you say, classifies your intent, and executes the right action — whether that's writing code, creating files, summarizing text, or just having a conversation. Everything runs on your machine. No cloud required.

---

## 📌 What It Can Do

| You say...                                        | What happens                                          |
|---------------------------------------------------|-------------------------------------------------------|
| "Create a Python file with a retry function"      | Generates the code and saves it to `output/`          |
| "Summarize this text and save it to notes.txt"    | Summarizes the text, then writes the file             |
| "Create a folder called projects"                 | Creates `output/projects/`                            |
| "Hey, what is recursion?"                         | Responds conversationally via the LLM                 |

---

## 🏗️ Architecture Overview

```
🎤 Audio Input (mic or file upload)
        ↓
🗣️  Speech-to-Text  [faster-whisper / Whisper base]
        ↓
🧠  Intent Classifier  [llama3.1:8b via Ollama]
        ↓
⚙️  Tool Executor
   ├── WRITE_CODE     → qwen2.5-coder:7b via Ollama
   ├── SAVE_FILE      → writes to output/
   ├── CREATE_FILE    → creates empty file in output/
   ├── CREATE_FOLDER  → creates directory in output/
   ├── SUMMARIZE_TEXT → llama3.1:8b via Ollama
   └── GENERAL_CHAT   → llama3.1:8b via Ollama
        ↓
🖥️  Gradio UI  (transcription + intent + result displayed)
```

### Key Components

- **`voice.py`** — Handles audio transcription using `faster-whisper` (Whisper base model, CPU-optimized with int8 quantization).
- **`intent_classifier.py`** — Sends the transcribed text to `llama3.1:8b` and parses a structured JSON plan of steps to execute.
- **`executor.py`** — Runs each step in the plan in order, passing outputs between steps (e.g., generated code → save to file).
- **`tools.py`** — The actual tool functions: file creation, code generation, summarization, chat. All file writes are sandboxed to `output/`.
- **`memory.py`** — Maintains a rolling 10-message conversation history within the session for context-aware responses.
- **`main.py`** — The Gradio UI that ties everything together.

---

## ✨ Features

- **Voice + Text input** — Speak into your mic or upload a `.wav`/`.mp3` file. Or just type.
- **Multi-step command chaining** — "Write a calculator in Python and save it as calc.py" executes two steps automatically.
- **Human-in-the-loop confirmation** — Before any file is written to disk, the UI shows a confirmation panel so you can review (and rename) the file before it's saved.
- **Session memory** — The agent remembers what was said earlier in the conversation for more coherent follow-ups.
- **Sandboxed file operations** — All file/folder creation is restricted to the `output/` directory. Path traversal attacks (like `../../etc/passwd`) are blocked.
- **Graceful error handling** — Unintelligible audio, unexpected model output, and unknown intents all fall back gracefully without crashing.

---

## 🖥️ System Requirements

- Python 3.10+
- [Ollama](https://ollama.com/) installed and running locally
- The following models pulled in Ollama:
  - `llama3.1:8b` (intent classification, chat, summarization)
  - `qwen2.5-coder:7b` (code generation)
- A CPU capable of running 8B parameter models (8–16 GB RAM recommended)
- Microphone (optional — you can also upload audio files)

---

## ⚙️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/your-username/intent-classifier-agent.git
cd intent-classifier-agent
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### 3. Install Python dependencies

```bash
pip install gradio faster-whisper ollama
```

### 4. Pull the required Ollama models

Make sure Ollama is running, then pull the models:

```bash
ollama pull llama3.1:8b
ollama pull qwen2.5-coder:7b
```

### 5. Run the app

```bash
python main.py
```

Then open your browser to `http://localhost:7860`.

---

## 📁 Project Structure

```
intent-classifier-agent/
├── main.py               # Gradio UI and event wiring
├── voice.py              # Speech-to-text with faster-whisper
├── intent_classifier.py  # LLM-based intent parsing → JSON steps
├── executor.py           # Step runner, chains tool outputs
├── tools.py              # Tool implementations (file ops, code gen, etc.)
├── memory.py             # Rolling conversation memory
├── output/               # All generated files go here (auto-created)
└── README.md
```

---

## 🔄 Example Flow (Step by Step)

**User says:** *"Write a Python retry function and save it as retry.py"*

1. `voice.py` transcribes the audio → `"Write a Python retry function and save it as retry.py"`
2. `intent_classifier.py` sends it to `llama3.1:8b`, which returns:
   ```json
   {
     "steps": [
       {"intent": "WRITE_CODE", "query": "Python retry function", "meta": {"language": "python"}},
       {"intent": "SAVE_FILE", "query": "save code", "meta": {"filename": "retry.py", "content_source": "previous_step"}}
     ]
   }
   ```
3. A **confirmation panel** appears in the UI showing: `💾 Save file: retry.py`
4. You click **Confirm** → `executor.py` runs both steps in order.
5. `qwen2.5-coder:7b` generates the code; it's written to `output/retry.py`.
6. The UI displays: transcription, detected intent, steps executed, and the generated code.

---

## 🛡️ Safety Design

All file operations go through `_safe_path()` in `tools.py`, which:
- Resolves the full path and checks it stays inside `output/`
- Strips leading slashes to prevent absolute path injection
- Raises an error if the resolved path escapes the sandbox

This means a command like *"save to ../../etc/hosts"* is rejected before any disk write happens.

---

## 🔩 Hardware Notes

**Speech-to-Text:** `faster-whisper` with the `base` model runs entirely on CPU using int8 quantization. It's fast enough for short voice commands on most modern laptops (typically 1–3 seconds for a 5-second clip).

**LLM (Ollama):** `llama3.1:8b` and `qwen2.5-coder:7b` are both 8B parameter models. They run on CPU but are faster with a GPU. On a machine with 16 GB RAM and no GPU, expect 10–30 seconds per generation. If this is too slow for your hardware, consider:
- Switching to `llama3.2:3b` (smaller, faster)
- Using an API-based LLM (OpenAI, Groq) by replacing the `ollama.chat()` calls in `tools.py` and `intent_classifier.py`

---

## 🚀 Bonus Features Implemented

- ✅ **Compound commands** — One audio input can trigger multiple chained steps (e.g., write + save).
- ✅ **Human-in-the-loop** — File operations require explicit confirmation before executing.
- ✅ **Graceful degradation** — Bad audio, empty input, and unknown intents all fall back to safe defaults.
- ✅ **Session memory** — Conversation context is maintained across turns within a session.

---

## 📄 License

MIT License — free to use, modify, and distribute.
