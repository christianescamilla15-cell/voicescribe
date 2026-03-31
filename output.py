"""Output manager — writes transcriptions to files for Claude Code to read."""
import os
from datetime import datetime
from config import OUTPUT_DIR, LATEST_FILE, HISTORY_FILE


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def write_latest(text: str, source: str = "mic"):
    """Overwrite latest.txt with the most recent transcription.
    Claude Code reads this file.
    """
    ensure_output_dir()
    timestamp = datetime.now().strftime("%H:%M:%S")
    with open(LATEST_FILE, "w", encoding="utf-8") as f:
        f.write(f"[{timestamp}] [{source}] {text}\n")


def append_history(text: str, source: str = "mic"):
    """Append to history.txt — full session log."""
    ensure_output_dir()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(HISTORY_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] [{source}] {text}\n")


def write_session(lines: list, session_name: str = None):
    """Write a complete session to its own file."""
    ensure_output_dir()
    if not session_name:
        session_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(OUTPUT_DIR, f"session_{session_name}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


def clear_latest():
    """Clear latest.txt."""
    ensure_output_dir()
    with open(LATEST_FILE, "w", encoding="utf-8") as f:
        f.write("")
