"""WhatsApp webhook handler for Twilio — receives voice notes, files, transcribes with Groq Whisper."""
import os
import io
import logging
import requests
from datetime import datetime
from fastapi import APIRouter, Form, Response
from typing import Optional

from groq import Groq

try:
    from PyPDF2 import PdfReader
    HAS_PDF = True
except ImportError:
    HAS_PDF = False

try:
    from docx import Document as DocxDocument
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False

logger = logging.getLogger(__name__)

router = APIRouter()

# Twilio config
TWILIO_SID = os.environ.get("TWILIO_SID", "")
TWILIO_TOKEN = os.environ.get("TWILIO_TOKEN", "")
WHISPER_MODEL = "whisper-large-v3-turbo"
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

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

    # Case 2: Document (PDF, Word, TXT, images) — check BEFORE text since PDFs send filename as Body
    elif num_media > 0 and MediaUrl0 and "audio" not in (MediaContentType0 or ""):
        content_type = MediaContentType0 or ""
        content_type = MediaContentType0 or ""
        try:
            file_bytes = download_twilio_media(MediaUrl0)
            extracted = ""
            file_type = "unknown"

            # PDF
            if "pdf" in content_type:
                file_type = "pdf"
                if HAS_PDF:
                    reader = PdfReader(io.BytesIO(file_bytes))
                    extracted = "\n".join(page.extract_text() or "" for page in reader.pages)
                else:
                    extracted = "[PyPDF2 not installed — PDF received but cannot extract]"

            # Word (.docx)
            elif "wordprocessingml" in content_type or "msword" in content_type:
                file_type = "docx"
                if HAS_DOCX:
                    doc = DocxDocument(io.BytesIO(file_bytes))
                    extracted = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
                else:
                    extracted = "[python-docx not installed — Word file received but cannot extract]"

            # Plain text / CSV
            elif "text" in content_type:
                file_type = "txt"
                extracted = file_bytes.decode("utf-8", errors="replace")

            # Image
            elif "image" in content_type:
                file_type = "image"
                # Save image to disk for Claude Code to read
                img_ext = content_type.split("/")[-1].split(";")[0]
                img_name = f"wa_image_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{img_ext}"
                img_path = os.path.join(TRANSCRIPTIONS_DIR, img_name)
                with open(img_path, "wb") as f:
                    f.write(file_bytes)
                extracted = f"[Imagen guardada: {img_name}]"

            # Other file — save raw
            else:
                file_type = content_type.split("/")[-1] if "/" in content_type else "file"
                file_name = f"wa_file_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{file_type}"
                file_path = os.path.join(TRANSCRIPTIONS_DIR, file_name)
                with open(file_path, "wb") as f:
                    f.write(file_bytes)
                extracted = f"[Archivo guardado: {file_name}]"

            if extracted:
                # Save to latest.txt with file content
                save_transcription(f"[FILE:{file_type}] {extracted[:5000]}", source="whatsapp-file")
                # Also save full content in a separate file
                doc_name = f"wa_doc_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                doc_path = os.path.join(TRANSCRIPTIONS_DIR, doc_name)
                with open(doc_path, "w", encoding="utf-8") as f:
                    f.write(extracted)
                response_text = f"Archivo {file_type} recibido ({len(extracted)} chars). Guardado como {doc_name}"
                logger.info("WhatsApp file processed: type=%s, chars=%d", file_type, len(extracted))
            else:
                response_text = f"Archivo {file_type} recibido pero no se pudo extraer contenido."

        except Exception as e:
            logger.error("WhatsApp file error: %s", e)
            response_text = f"Error procesando archivo: {e}"

    # Case 3: Text message
    elif Body.strip():
        save_transcription(Body.strip(), source="whatsapp-text")
        response_text = f"Guardado: {Body.strip()[:100]}"
        logger.info("WhatsApp text saved: %s", Body[:100])

    else:
        response_text = "Envia una nota de voz o texto para transcribir."

    # Respond with TwiML
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Message>{response_text}</Message>
</Response>"""

    return Response(content=twiml, media_type="application/xml")
