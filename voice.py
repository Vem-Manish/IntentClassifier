from faster_whisper import WhisperModel


class STTService:
    """
    Speech-to-text using faster-whisper (Whisper base model, CPU-optimised).
    int8 quantisation keeps memory usage low on consumer hardware.
    """

    def __init__(self):
        self.model = WhisperModel("base", device="cpu", compute_type="int8")

    def transcribe_file(self, file_path: str) -> str:
        """Transcribe an audio file and return the joined transcript string."""
        if not file_path:
            return ""

        try:
            segments, _ = self.model.transcribe(
                file_path,
                language="en",
                beam_size=5,
                vad_filter=True,
                condition_on_previous_text=False,
            )
            return " ".join(seg.text.strip() for seg in segments).strip()
        except Exception as exc:
            return f"[Transcription error] {exc}"