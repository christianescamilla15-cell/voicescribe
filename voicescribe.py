"""
VoiceScribe — Real-time voice transcription for Claude Code.

Modes:
  1. mic      — Capture your microphone only
  2. system   — Capture system audio only (Zoom/Meet other side)
  3. dual     — Capture both mic + system (phone call on speaker)
  4. devices  — List all audio devices

Usage:
  python voicescribe.py mic
  python voicescribe.py system
  python voicescribe.py dual
  python voicescribe.py devices
"""
import sys
import signal
import numpy as np
from datetime import datetime

from config import CHUNK_DURATION_SEC
from audio_capture import (
    list_devices, get_default_mic, find_loopback_device,
    record_chunk, record_dual, is_silence,
)
from transcriber import transcribe_audio
from output import write_latest, append_history, write_session, clear_latest


running = True
session_lines = []


def signal_handler(sig, frame):
    global running
    print("\n\nStopping VoiceScribe...")
    running = False


signal.signal(signal.SIGINT, signal_handler)


def run_mic_mode(language="es"):
    """Mode 1: Capture microphone only."""
    global running
    mic = get_default_mic()
    print(f"Mode: MICROPHONE | Device: {mic} | Language: {language}")
    print(f"Chunk: {CHUNK_DURATION_SEC}s | Press Ctrl+C to stop\n")
    clear_latest()

    while running:
        try:
            audio = record_chunk(CHUNK_DURATION_SEC, device=mic)
            if is_silence(audio):
                continue

            text = transcribe_audio(audio, language=language)
            if text and not text.startswith("[Error"):
                print(f"  YOU: {text}")
                write_latest(text, source="you")
                append_history(text, source="you")
                session_lines.append(f"[YOU] {text}")
        except Exception as e:
            print(f"  Error: {e}")


def run_system_mode(language="es", device_idx=None):
    """Mode 2: Capture system audio only (other side of Zoom/Meet)."""
    global running

    if device_idx is None:
        device_idx = find_loopback_device()

    if device_idx is None:
        print("ERROR: No loopback device found.")
        print("You need to enable 'Stereo Mix' in Windows Sound settings:")
        print("  1. Right-click speaker icon in taskbar → Sound settings")
        print("  2. Go to Recording tab")
        print("  3. Right-click → Show Disabled Devices")
        print("  4. Enable 'Stereo Mix'")
        print("\nOr install VB-Cable: https://vb-audio.com/Cable/")
        print("\nAvailable devices:")
        list_devices()
        return

    devices = list(filter(None, []))
    import sounddevice as sd
    dev_name = sd.query_devices(device_idx)["name"]
    print(f"Mode: SYSTEM AUDIO | Device: [{device_idx}] {dev_name} | Language: {language}")
    print(f"Chunk: {CHUNK_DURATION_SEC}s | Press Ctrl+C to stop\n")
    clear_latest()

    while running:
        try:
            audio = record_chunk(CHUNK_DURATION_SEC, device=device_idx)
            if is_silence(audio):
                continue

            text = transcribe_audio(audio, language=language)
            if text and not text.startswith("[Error"):
                print(f"  THEM: {text}")
                write_latest(text, source="them")
                append_history(text, source="them")
                session_lines.append(f"[THEM] {text}")
        except Exception as e:
            print(f"  Error: {e}")


def run_dual_mode(language="es", system_device_idx=None):
    """Mode 3: Capture both mic + system audio (phone call on speaker)."""
    global running

    mic = get_default_mic()

    if system_device_idx is None:
        system_device_idx = find_loopback_device()

    if system_device_idx is None:
        print("WARNING: No loopback device found. Will capture mic only.")
        print("To capture both sides, enable Stereo Mix or install VB-Cable.\n")
        run_mic_mode(language)
        return

    import sounddevice as sd
    sys_name = sd.query_devices(system_device_idx)["name"]
    print(f"Mode: DUAL (mic + system)")
    print(f"  Mic: [{mic}] | System: [{system_device_idx}] {sys_name}")
    print(f"  Language: {language} | Chunk: {CHUNK_DURATION_SEC}s")
    print(f"  Press Ctrl+C to stop\n")
    clear_latest()

    while running:
        try:
            mic_audio, sys_audio = record_dual(
                CHUNK_DURATION_SEC,
                mic_device=mic,
                system_device=system_device_idx,
            )

            # Transcribe mic (you)
            if mic_audio is not None and not is_silence(mic_audio):
                text = transcribe_audio(mic_audio, language=language)
                if text and not text.startswith("[Error"):
                    print(f"  YOU:  {text}")
                    write_latest(text, source="you")
                    append_history(text, source="you")
                    session_lines.append(f"[YOU] {text}")

            # Transcribe system (them)
            if sys_audio is not None and not is_silence(sys_audio):
                text = transcribe_audio(sys_audio, language=language)
                if text and not text.startswith("[Error"):
                    print(f"  THEM: {text}")
                    write_latest(text, source="them")
                    append_history(text, source="them")
                    session_lines.append(f"[THEM] {text}")

        except Exception as e:
            print(f"  Error: {e}")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "mic"
    language = sys.argv[2] if len(sys.argv) > 2 else "es"
    device = int(sys.argv[3]) if len(sys.argv) > 3 else None

    print("=" * 50)
    print("  VoiceScribe — Real-time Transcription")
    print("  Powered by Groq Whisper (free)")
    print("  Output: transcriptions/latest.txt")
    print("=" * 50)

    if mode == "devices":
        list_devices()
        return

    if mode == "mic":
        run_mic_mode(language)
    elif mode == "system":
        run_system_mode(language, device_idx=device)
    elif mode == "dual":
        run_dual_mode(language, system_device_idx=device)
    else:
        print(f"Unknown mode: {mode}")
        print("Usage: python voicescribe.py [mic|system|dual|devices] [es|en] [device_idx]")
        return

    # Save session on exit
    if session_lines:
        path = write_session(session_lines)
        print(f"\nSession saved: {path}")
        print(f"Total lines: {len(session_lines)}")


if __name__ == "__main__":
    main()
