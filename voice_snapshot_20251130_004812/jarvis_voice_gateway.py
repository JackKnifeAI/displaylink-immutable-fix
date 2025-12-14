from __future__ import annotations

import asyncio
import logging
import os
from typing import Dict, Optional

import httpx
import numpy as np
import sounddevice as sd
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pythonjsonlogger import jsonlogger

from hotword import HotwordListener
from stt import STTEngine
from tts import TTSEngine
from vad import DummyVAD

SERVICE_VERSION = "0.1.0"
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
ORCHESTRATOR_URL = os.environ.get("ORCHESTRATOR_URL", "http://orchestrator:8080")
VOSK_MODEL_PATH = os.environ.get("VOSK_MODEL_PATH", "/models/vosk")
PIPER_BINARY = os.environ.get("PIPER_BINARY", "/usr/local/bin/piper")
PIPER_VOICE = os.environ.get("PIPER_VOICE", "/voices/en_US-amy-low.onnx")
HOTWORD_MODEL_PATH = os.environ.get("HOTWORD_MODEL_PATH", "/models/openwakeword/jarvis.tflite")
MAX_VOICE_TEXT_CHARS = int(os.environ.get("MAX_VOICE_TEXT_CHARS", "512"))


def _token_set(*env_names: str) -> set[str]:
    tokens: set[str] = set()
    for name in env_names:
        raw = os.environ.get(name, "")
        if raw:
            for part in raw.split(","):
                token = part.strip()
                if token:
                    tokens.add(token)
    return tokens

ALLOWED_TOKENS = _token_set("VOICE_GATEWAY_TOKEN", "JARVIS_INTERNAL_TOKEN", "JARVIS_ALLOWED_TOKENS")
ORCHESTRATOR_TOKEN = os.environ.get("ORCHESTRATOR_TOKEN") or os.environ.get("JARVIS_INTERNAL_TOKEN", "")


def setup_logging() -> logging.Logger:
    handler = logging.StreamHandler()
    formatter = jsonlogger.JsonFormatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    handler.setFormatter(formatter)
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(LOG_LEVEL)
    logger = logging.getLogger("jarvis.voice")
    logger.info("logging configured", extra={"level": LOG_LEVEL})
    return logger


LOGGER = setup_logging()

stt_engine = STTEngine(VOSK_MODEL_PATH)
tts_engine = TTSEngine(PIPER_BINARY, PIPER_VOICE)
vad_engine = DummyVAD()
http_client = httpx.AsyncClient(timeout=60.0)
hotword_listener: Optional[HotwordListener] = None
main_loop: Optional[asyncio.AbstractEventLoop] = None


class AskTextRequest(BaseModel):
    text: str


class AskTextResponse(BaseModel):
    reply: str


class VoiceInteractionResponse(BaseModel):
    heard: str
    reply: str
    note: Optional[str] = None


class HotwordStatus(BaseModel):
    running: bool


def require_internal_token(
    x_internal_token: Optional[str] = Header(default=None, alias="X-Internal-Token")
) -> None:
    if not ALLOWED_TOKENS:
        return
    if not x_internal_token or x_internal_token not in ALLOWED_TOKENS:
        raise HTTPException(status_code=401, detail="invalid internal token")


async def ask_orchestrator(payload: Dict[str, str]) -> str:
    headers = {"X-Internal-Token": ORCHESTRATOR_TOKEN} if ORCHESTRATOR_TOKEN else {}
    try:
        resp = await http_client.post(f"{ORCHESTRATOR_URL}/api/ask", json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        return data.get("reply", "")
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Orchestrator error: {exc}")


def get_hotword_listener(create: bool) -> Optional[HotwordListener]:
    global hotword_listener
    if hotword_listener is None and create:
        if not os.path.exists(HOTWORD_MODEL_PATH):
            raise HTTPException(status_code=500, detail="hotword model not present")
        hotword_listener = HotwordListener(
            HOTWORD_MODEL_PATH,
            on_hotword=hotword_callback,
        )
    return hotword_listener


def hotword_callback() -> None:  # pragma: no cover - invoked from audio thread
    if not main_loop:
        return
    asyncio.run_coroutine_threadsafe(run_hotword_interaction(), main_loop)


async def run_hotword_interaction() -> None:
    try:
        freq = 880.0
        duration = 0.15
        samplerate = 16000
        t = np.arange(int(duration * samplerate)) / samplerate
        beep = 0.2 * np.sin(2 * np.pi * freq * t)
        sd.play(beep.astype(np.float32), samplerate)
        sd.wait()

        user_text = stt_engine.listen_once(timeout=10.0)
        if not user_text:
            print("[HOTWORD] Silence detected")
            return
        print(f"[HOTWORD] Heard: {user_text}")
        reply = await ask_orchestrator({"input": user_text, "mode": "voice", "source": "hotword"})
        tts_engine.speak(reply)
    except Exception as exc:
        print(f"[HOTWORD] interaction error: {exc}")


app = FastAPI(title="Jarvis Voice Gateway", version=SERVICE_VERSION)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.on_event("startup")
async def startup_event() -> None:
    global main_loop
    main_loop = asyncio.get_running_loop()


@app.on_event("shutdown")
async def shutdown_event() -> None:
    await http_client.aclose()
    if hotword_listener and hotword_listener.is_running():
        hotword_listener.stop()


@app.get("/health")
async def health() -> Dict[str, str]:
    running = hotword_listener.is_running() if hotword_listener else False
    return {"status": "ok", "hotword_running": str(running)}


@app.post("/api/ask-text", response_model=AskTextResponse, dependencies=[Depends(require_internal_token)])
async def api_ask_text(payload: AskTextRequest) -> AskTextResponse:
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="empty text")
    if len(text) > MAX_VOICE_TEXT_CHARS:
        raise HTTPException(status_code=400, detail="text too long")
    reply = await ask_orchestrator({"input": text, "mode": "chat", "source": "voice_gateway-text"})
    tts_engine.speak(reply)
    LOGGER.info("ask-text processed", extra={"chars_in": len(text), "chars_out": len(reply)})
    return AskTextResponse(reply=reply)


@app.post("/api/voice-interaction", response_model=VoiceInteractionResponse, dependencies=[Depends(require_internal_token)])
async def api_voice_interaction() -> VoiceInteractionResponse:
    user_text = stt_engine.listen_once(timeout=10.0)
    if not user_text:
        LOGGER.warning("voice interaction heard nothing")
        return VoiceInteractionResponse(heard="", reply="", note="no speech detected")
    if len(user_text) > MAX_VOICE_TEXT_CHARS:
        LOGGER.warning("voice input too long", extra={"chars_in": len(user_text)})
        raise HTTPException(status_code=400, detail="voice input too long")
    reply = await ask_orchestrator({"input": user_text, "mode": "voice", "source": "voice_gateway"})
    tts_engine.speak(reply)
    LOGGER.info("voice interaction complete", extra={"chars_in": len(user_text), "chars_out": len(reply)})
    return VoiceInteractionResponse(heard=user_text, reply=reply)


@app.post("/api/hotword/start", response_model=HotwordStatus, dependencies=[Depends(require_internal_token)])
async def api_hotword_start() -> HotwordStatus:
    listener = get_hotword_listener(create=True)
    if not listener.is_running():
        listener.start()
    LOGGER.info("hotword listener started")
    return HotwordStatus(running=True)


@app.post("/api/hotword/stop", response_model=HotwordStatus, dependencies=[Depends(require_internal_token)])
async def api_hotword_stop() -> HotwordStatus:
    listener = get_hotword_listener(create=True)
    if listener.is_running():
        listener.stop()
    LOGGER.info("hotword listener stopped")
    return HotwordStatus(running=False)


@app.get("/api/hotword/status", response_model=HotwordStatus, dependencies=[Depends(require_internal_token)])
async def api_hotword_status() -> HotwordStatus:
    listener = get_hotword_listener(create=False)
    running = listener.is_running() if listener else False
    return HotwordStatus(running=running)


@app.get("/")
async def root() -> Dict[str, str]:
    return {"service": "jarvis-voice-gateway", "version": SERVICE_VERSION}
