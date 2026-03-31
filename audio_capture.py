"""Audio capture utilities for all modes."""
import sounddevice as sd
import numpy as np
from config import SAMPLE_RATE, CHANNELS, SILENCE_THRESHOLD


def list_devices():
    """List all audio devices with their index."""
    devices = sd.query_devices()
    print("\n=== Audio Devices ===")
    for i, d in enumerate(devices):
        direction = ""
        if d["max_input_channels"] > 0:
            direction += "IN "
        if d["max_output_channels"] > 0:
            direction += "OUT"
        print(f"  [{i}] {direction:6} {d['name']}")
    print()
    return devices


def get_default_mic():
    """Get default microphone device index."""
    return sd.default.device[0]


def find_loopback_device():
    """Find a loopback/stereo mix device for system audio capture."""
    devices = sd.query_devices()
    keywords = ["stereo mix", "loopback", "what u hear", "wave out", "cable output"]
    for i, d in enumerate(devices):
        name_lower = d["name"].lower()
        if d["max_input_channels"] > 0:
            for kw in keywords:
                if kw in name_lower:
                    return i
    return None


def record_chunk(duration: float, device=None) -> np.ndarray:
    """Record a chunk of audio. Returns int16 numpy array."""
    audio = sd.rec(
        int(duration * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="int16",
        device=device,
    )
    sd.wait()
    return audio.flatten()


def is_silence(audio_data: np.ndarray) -> bool:
    """Check if audio chunk is silence based on RMS."""
    rms = np.sqrt(np.mean(audio_data.astype(np.float64) ** 2))
    return rms < SILENCE_THRESHOLD


def record_dual(duration: float, mic_device=None, system_device=None):
    """Record from mic and system audio simultaneously.
    Returns (mic_audio, system_audio) as int16 numpy arrays.
    If system_device is None, returns (mic_audio, None).
    """
    import threading

    mic_audio = None
    sys_audio = None

    def rec_mic():
        nonlocal mic_audio
        mic_audio = record_chunk(duration, device=mic_device)

    def rec_sys():
        nonlocal sys_audio
        sys_audio = record_chunk(duration, device=system_device)

    t_mic = threading.Thread(target=rec_mic)
    t_mic.start()

    if system_device is not None:
        t_sys = threading.Thread(target=rec_sys)
        t_sys.start()
        t_sys.join()

    t_mic.join()
    return mic_audio, sys_audio
