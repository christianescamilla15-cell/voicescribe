"""WhatsApp webhook handler for Twilio — receives voice notes, transcribes with Groq Whisper."""
import os
import io
import logging
import requests
from datetime import datetime
from fastapi import APIRouter, Form, Response
from typing import Optional

from groq import Groq

logger = logging.getLogger(__name__)

router = APIRouter()

# Twilio config
TWILIO_SID = "AC2df6900d836269f47"
TWILIO_TOKEN = "35112b7b9a8962d06127b"
WHISPER_MODEL = "whisper-large-v3-turbo"
GROQ_API_KEY = os.environ.get(
    "GROQ_API_KEY",
    "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",
)

groq_client = Groq(api_key=GROQ_API_KEY)

TRANSCRIPTIONS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "transcriptions")
os.makedirs(TRANSCRIPTIONS_DIR, exist_ok=True)


def save_transcription(text: str, source: str = "whatsapp"):
    """Save to files for Claude Code."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(os.path.join(TRANSCRIPTIONS_DIR, "latest.txt"), "w", encoding="utf-8") as f:
        f.write(f"[{timestamp}] [{source}] {text}\n")

    with open(os.path.join(TRANSCRIPTIONS_DIR, "history.txt"), "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] [{source}] {text}\n")


def download_twilio_media(media_url: str) -> bytes:
    """Download media from Twilio (requires auth)."""
    resp = requests.get(media_url, auth=(TWILIO_SID, TWILIO_TOKEN), timeout=30)
    resp.raise_for_status()
    return resp.content


def transcribe_audio_bytes(audio_bytes: bytes, language: str = "es") -> str:
    """Send audio to Groq Whisper."""
    try:
        result = groq_client.audio.transcriptions.create(
            file=("audio.ogg", audio_bytes),
            model=WHISPER_MODEL,
            language=language,
            response_format="text",
        )
        return result.strip() if isinstance(result, str) else str(result).strip()
    except Exception as e:
        logger.error("Whisper error: %s", e)
        return ""


@router.post("/webhook/whatsapp")
async def whatsapp_webhook(
    Body: str = Form(""),
    From: str = Form(""),
    NumMedia: str = Form("0"),
    MediaUrl0: Optional[str] = Form(None),
    MediaContentType0: Optional[str] = Form(None),
):
    """Receive WhatsApp messages from Twilio.
    - Text messages: save directly
    - Voice notes: download, transcribe with Whisper, save
    """
    sender = From.replace("whatsapp:", "")
    num_media = int(NumMedia)
    response_text = ""

    logger.info("WhatsApp from %s: body='%s', media=%d", sender, Body[:50], num_media)

    # Case 1: Voice note
    if num_media > 0 and MediaUrl0 and "audio" in (MediaContentType0 or ""):
        try:
            audio_bytes = download_twilio_media(MediaUrl0)
            text = transcribe_audio_bytes(audio_bytes)

            if text:
                save_transcription(text, source="whatsapp")
                response_text = f"Transcrito: {text}"
                logger.info("WhatsApp voice transcribed: %s", text[:100])
            else:
                response_text = "No se pudo transcribir el audio."
        except Exception as e:
            logger.error("WhatsApp voice error: %s", e)
            response_text = f"Error: {e}"

    # Case 2: Text message
    elif Body.strip():
        save_transcription(Body.strip(), source="whatsapp-text")
        response_text = f"Guardado: {Body.strip()[:100]}"
        logger.info("WhatsApp text saved: %s", Body[:100])

    # Case 3: Other media (image, doc, etc)
    elif num_media > 0:
        response_text = "Solo proceso notas de voz y texto por ahora."

    else:
        response_text = "Envia una nota de voz o texto para transcribir."

    # Respond with TwiML
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Message>{response_text}</Message>
</Response>"""

    return Response(content=twiml, media_type="application/xml")
