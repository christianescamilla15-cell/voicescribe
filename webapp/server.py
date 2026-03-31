"""VoiceScribe Web App — FastAPI backend with WebSocket for real-time transcription."""
import io
import os
import json
import wave
import asyncio
import logging
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from groq import Groq

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Config
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
WHISPER_MODEL = "whisper-large-v3-turbo"
TRANSCRIPTIONS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "transcriptions")
os.makedirs(TRANSCRIPTIONS_DIR, exist_ok=True)

client = Groq(api_key=GROQ_API_KEY)

app = FastAPI(title="VoiceScribe", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store
sessions: dict[str, list] = {}
# Connected WebSocket clients (for real-time sync between phone and PC)
ws_clients: dict[str, list[WebSocket]] = {}


def transcribe_bytes(audio_bytes: bytes, language: str = "es") -> str:
    """Send audio bytes to Groq Whisper."""
    try:
        result = client.audio.transcriptions.create(
            file=("audio.wav", audio_bytes),
            model=WHISPER_MODEL,
            language=language,
            response_format="text",
        )
        text = result.strip() if isinstance(result, str) else str(result).strip()
        return text
    except Exception as e:
        logger.error("Whisper error: %s", e)
        return ""


def save_to_files(text: str, source: str, session_id: str):
    """Save transcription to files for Claude Code."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # latest.txt — Claude Code reads this
    with open(os.path.join(TRANSCRIPTIONS_DIR, "latest.txt"), "w", encoding="utf-8") as f:
        f.write(f"[{timestamp}] [{source}] {text}\n")

    # history.txt — full log
    with open(os.path.join(TRANSCRIPTIONS_DIR, "history.txt"), "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] [{source}] {text}\n")

    # session file
    session_file = os.path.join(TRANSCRIPTIONS_DIR, f"session_{session_id}.txt")
    with open(session_file, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] [{source}] {text}\n")


async def broadcast_to_session(session_id: str, message: dict):
    """Send transcription to all WebSocket clients in a session."""
    if session_id in ws_clients:
        dead = []
        for ws in ws_clients[session_id]:
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            ws_clients[session_id].remove(ws)


# ── REST Endpoints ──

@app.get("/api/health")
async def health():
    return {"status": "ok", "service": "VoiceScribe", "whisper": WHISPER_MODEL}


@app.post("/api/transcribe")
async def transcribe_upload(
    audio: UploadFile = File(...),
    language: str = Form("es"),
    source: str = Form("mic"),
    session_id: str = Form("default"),
):
    """Upload audio file and get transcription."""
    audio_bytes = await audio.read()
    text = transcribe_bytes(audio_bytes, language)

    if text:
        # Save to files
        save_to_files(text, source, session_id)

        # Store in memory
        entry = {
            "text": text,
            "source": source,
            "timestamp": datetime.now().isoformat(),
        }
        sessions.setdefault(session_id, []).append(entry)

        # Broadcast to WebSocket clients
        await broadcast_to_session(session_id, {"type": "transcription", **entry})

    return {"text": text, "source": source, "session_id": session_id}


@app.get("/api/sessions")
async def list_sessions():
    """List all sessions."""
    return {
        "sessions": [
            {"id": sid, "lines": len(lines), "last": lines[-1]["timestamp"] if lines else None}
            for sid, lines in sessions.items()
        ]
    }


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    """Get all transcriptions for a session."""
    lines = sessions.get(session_id, [])
    return {"session_id": session_id, "lines": lines, "total": len(lines)}


@app.get("/api/latest")
async def get_latest():
    """Get latest transcription (what Claude Code reads)."""
    path = os.path.join(TRANSCRIPTIONS_DIR, "latest.txt")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return {"text": f.read().strip()}
    return {"text": ""}


# ── WebSocket for real-time sync ──

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """Real-time transcription stream. Phone records, PC sees text live."""
    await websocket.accept()
    ws_clients.setdefault(session_id, []).append(websocket)
    logger.info("WS connected: session=%s, clients=%d", session_id, len(ws_clients[session_id]))

    try:
        while True:
            data = await websocket.receive()

            # Text message (ping/control)
            if "text" in data:
                msg = json.loads(data["text"])
                if msg.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})

            # Binary message (audio chunk from browser)
            elif "bytes" in data:
                audio_bytes = data["bytes"]
                language = "es"  # default

                # Transcribe
                text = transcribe_bytes(audio_bytes, language)
                if text:
                    source = "phone"
                    entry = {
                        "text": text,
                        "source": source,
                        "timestamp": datetime.now().isoformat(),
                    }
                    sessions.setdefault(session_id, []).append(entry)
                    save_to_files(text, source, session_id)

                    # Broadcast to ALL clients in session (including sender)
                    await broadcast_to_session(session_id, {"type": "transcription", **entry})

    except WebSocketDisconnect:
        pass
    finally:
        if session_id in ws_clients:
            ws_clients[session_id] = [c for c in ws_clients[session_id] if c != websocket]
        logger.info("WS disconnected: session=%s", session_id)


# ── WhatsApp webhook ──
from whatsapp_handler import router as whatsapp_router
app.include_router(whatsapp_router, prefix="/api")

# ── Serve frontend ──
frontend_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    cert = os.path.join(os.path.dirname(__file__), "cert.pem")
    key = os.path.join(os.path.dirname(__file__), "key.pem")
    if os.path.exists(cert) and os.path.exists(key):
        uvicorn.run(app, host="0.0.0.0", port=8888, ssl_certfile=cert, ssl_keyfile=key)
    else:
        uvicorn.run(app, host="0.0.0.0", port=8888)
