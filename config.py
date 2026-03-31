"""VoiceScribe configuration."""
import os

# Groq API (Whisper transcription - free)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "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")

# Audio settings
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_DURATION_SEC = 5  # Transcribe every 5 seconds of audio
SILENCE_THRESHOLD = 300  # RMS below this = silence, skip transcription

# Output
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "transcriptions")
LATEST_FILE = os.path.join(OUTPUT_DIR, "latest.txt")
HISTORY_FILE = os.path.join(OUTPUT_DIR, "history.txt")

# Whisper model on Groq
WHISPER_MODEL = "whisper-large-v3-turbo"
