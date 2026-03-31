"""Groq Whisper transcription client."""
import io
import wave
import numpy as np
from groq import Groq
from config import GROQ_API_KEY, WHISPER_MODEL, SAMPLE_RATE, CHANNELS


client = Groq(api_key=GROQ_API_KEY)


def transcribe_audio(audio_data: np.ndarray, language: str = "es") -> str:
    """Send audio to Groq Whisper and return transcription text."""
    # Convert numpy array to WAV bytes
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_data.tobytes())
    buf.seek(0)

    try:
        transcription = client.audio.transcriptions.create(
            file=("audio.wav", buf.read()),
            model=WHISPER_MODEL,
            language=language,
            response_format="text",
        )
        text = transcription.strip() if isinstance(transcription, str) else str(transcription).strip()
        return text if text else ""
    except Exception as e:
        return f"[Error: {e}]"
