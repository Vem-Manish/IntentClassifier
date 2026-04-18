import os
import gradio as gr

from voice import STTService
from intent_classifier import classify_intents
from executor import execute_steps, needs_confirmation, describe_file_ops
from memory import ConversationMemory

# ---------------------------------------------------------------------------
# Service initialisation
# ---------------------------------------------------------------------------

stt = STTService()
memory = ConversationMemory(max_messages=10)


# ---------------------------------------------------------------------------
# Chat history helpers — Gradio 6 REQUIRES dict messages (No 'type' arg needed)
# ---------------------------------------------------------------------------

def _msg(role: str, content: str) -> dict:
    """Creates a message dictionary compliant with Gradio 6."""
    return {"role": role, "content": content}


def _chat_append(history: list, user: str, assistant: str) -> list:
    """Append user and assistant turns as separate dictionaries."""
    history = history or []
    history.append(_msg("user", user))
    history.append(_msg("assistant", assistant))
    return history


# ---------------------------------------------------------------------------
# Filename helpers
# ---------------------------------------------------------------------------

def _first_filename(steps: list) -> str:
    for step in steps:
        meta = step.get("meta") or {}
        intent = step.get("intent", "")
        if intent == "SAVE_FILE":
            return meta.get("filename", "output.txt")
        if intent == "CREATE_FILE":
            return meta.get("filename", "new_file.txt")
        if intent == "CREATE_FOLDER":
            return meta.get("foldername", "new_folder")
    return ""


def _patch_filename(steps: list, new_name: str) -> list:
    patched = []
    replaced = False
    for step in steps:
        s = dict(step)
        if not replaced and s.get("intent") in ("SAVE_FILE", "CREATE_FILE", "CREATE_FOLDER"):
            meta = dict(s.get("meta") or {})
            if s["intent"] in ("SAVE_FILE", "CREATE_FILE"):
                meta["filename"] = new_name
            else:
                meta["foldername"] = new_name
            s["meta"] = meta
            replaced = True
        patched.append(s)
    return patched


# ---------------------------------------------------------------------------
# Shared empty yield
# ---------------------------------------------------------------------------
def _blank(history):
    return ("", "", "", "", history,
            gr.update(visible=False), [], "", "", gr.update(value=""))


# ---------------------------------------------------------------------------
# Phase 1 — Classify + stream outputs
# ---------------------------------------------------------------------------

def run_classify(audio_path, text_input, chat_history):
    chat_history = chat_history or []

    yield _blank(chat_history)

    if text_input and text_input.strip():
        user_text = text_input.strip()
        transcription = "(manual input)"
    elif audio_path:
        if not os.path.exists(audio_path):
            yield ("Audio file not found.", "—", "—", "—", chat_history,
                   gr.update(visible=False), [], "", "", gr.update(value=""))
            return
        transcription = stt.transcribe_file(audio_path)
        user_text = transcription
    else:
        yield ("No input provided.", "—", "—", "—", chat_history,
               gr.update(visible=False), [], "", "", gr.update(value=""))
        return

    if not user_text:
        yield ("Nothing detected in audio.", "—", "—", "—", chat_history,
               gr.update(visible=False), [], "", "", gr.update(value=""))
        return

    yield (transcription, "", "", "", chat_history,
           gr.update(visible=False), [], "", "", gr.update(value=""))

    context = memory.get()
    routed = classify_intents(user_text, history=context)
    steps = routed.get("steps", [])
    intent_label = " -> ".join(s.get("intent", "GENERAL_CHAT") for s in steps)

    yield (transcription, intent_label, "", "", chat_history,
           gr.update(visible=False), [], "", "", gr.update(value=""))

    if not needs_confirmation(steps):
        logs, result = execute_steps(steps, user_text, history=context)
        memory.add("user", user_text)
        memory.add("assistant", result)
        chat_history = _chat_append(chat_history, user_text, result)

        yield (transcription, intent_label, logs, result, chat_history,
               gr.update(visible=False), [], "", "", gr.update(value=""))
        return

    file_ops_desc = describe_file_ops(steps)
    first_filename = _first_filename(steps)
    yield (transcription, intent_label, "Waiting for confirmation...", "",
           chat_history, gr.update(visible=True),
           steps, user_text, file_ops_desc, gr.update(value=first_filename))


# ---------------------------------------------------------------------------
# Phase 2a — User confirmed
# ---------------------------------------------------------------------------

def run_confirmed(steps, user_text, edited_filename, chat_history):
    chat_history = chat_history or []

    if edited_filename and edited_filename.strip():
        steps = _patch_filename(steps, edited_filename.strip())

    yield ("Running...", "", chat_history, gr.update(visible=False), [], "")

    context = memory.get()
    logs, result = execute_steps(steps, user_text, history=context)
    memory.add("user", user_text)
    memory.add("assistant", result)
    chat_history = _chat_append(chat_history, user_text, result)

    yield (logs, result, chat_history, gr.update(visible=False), [], "")


# ---------------------------------------------------------------------------
# Phase 2b — User cancelled
# ---------------------------------------------------------------------------

def run_cancelled(chat_history):
    chat_history = chat_history or []
    abort_msg = "Action cancelled. No files were created or modified."
    chat_history = _chat_append(chat_history, "(cancelled)", abort_msg)
    return ("Cancelled.", abort_msg, chat_history,
            gr.update(visible=False), [], "")


# ---------------------------------------------------------------------------
# Clear session
# ---------------------------------------------------------------------------

def clear_session():
    memory.clear()
    return (None, "", [], "", [],
            gr.update(visible=False), [], "", gr.update(value=""))


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

CSS = """
body, .gradio-container { font-family: 'Inter', system-ui, sans-serif; }
.section-label { font-size: 0.7rem; font-weight: 600; text-transform: uppercase; color: #888; margin-bottom: 4px; }
#run-btn { background: #1a1a1a; color: #fff; border: none; }
#confirm-panel { border: 1px solid #3a3a3a; border-left: 3px solid #f59e0b; border-radius: 8px; padding: 16px; background: #1e1e1e; margin-top: 14px; }
#confirm-btn { background: #16a34a !important; color: #fff !important; }
#cancel-btn { background: #1e1e1e !important; border: 1px solid #dc2626 !important; color: #dc2626 !important; }
"""

with gr.Blocks(title="Intent Classifier") as demo:
    pending_steps = gr.State([])
    pending_text = gr.State("")

    gr.Markdown("## Intent Classifier\nLocal AI agent pipeline.")

    with gr.Row():
        with gr.Column(scale=1):
            # FIXED: Added format="wav" and interactive=True to stabilize playback for longer recordings
            audio_in = gr.Audio(
                label="Voice command",
                type="filepath",
                sources=["microphone", "upload"],
                format="wav",
                interactive=True
            )
            text_in = gr.Textbox(label="Text command", placeholder="or type here...")
            with gr.Row():
                run_btn = gr.Button("Run", variant="primary", elem_id="run-btn")
                clear_btn = gr.Button("Clear")

            with gr.Group(elem_id="confirm-panel", visible=False) as confirm_panel:
                gr.Markdown("**⚠ Confirm File Operation**")
                file_ops_display = gr.Markdown("")
                filename_edit = gr.Textbox(label="Filename (edit if needed)", interactive=True)
                with gr.Row():
                    confirm_btn = gr.Button("Confirm", variant="primary", elem_id="confirm-btn")
                    cancel_btn = gr.Button("Cancel", elem_id="cancel-btn")

        with gr.Column(scale=1):
            out_transcription = gr.Textbox(label="Transcription", interactive=False)
            out_intent = gr.Textbox(label="Detected intent", interactive=False)
            out_logs = gr.Code(label="Steps executed", language="markdown", interactive=False)
            out_result = gr.Markdown(label="Result")

    history_display = gr.Chatbot(
        elem_classes=["chatbot-wrap"],
        show_label=False,
        height=320,
    )

    # Event wiring
    run_btn.click(
        fn=run_classify,
        inputs=[audio_in, text_in, history_display],
        outputs=[out_transcription, out_intent, out_logs, out_result, history_display, confirm_panel, pending_steps,
                 pending_text, file_ops_display, filename_edit]
    )

    confirm_btn.click(
        fn=run_confirmed,
        inputs=[pending_steps, pending_text, filename_edit, history_display],
        outputs=[out_logs, out_result, history_display, confirm_panel, pending_steps, pending_text]
    )

    cancel_btn.click(
        fn=run_cancelled,
        inputs=[history_display],
        outputs=[out_logs, out_result, history_display, confirm_panel, pending_steps, pending_text]
    )

    clear_btn.click(
        fn=clear_session,
        outputs=[audio_in, text_in, out_logs, out_result, history_display, confirm_panel, pending_steps, pending_text,
                 filename_edit]
    )

if __name__ == "__main__":
    demo.queue().launch(share=False, css=CSS)