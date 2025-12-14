#!/usr/bin/env python3
"""
astraeus_voice.py

Continuous Voice REPL for SHAI/Astraeus with always-on listening:

- Uses Vosk for continuous speech recognition (no fixed recording time)
- Falls back to Whisper for 8-second recordings if Vosk unavailable
- Type text directly for text input
- All responses are spoken via TTS (espeak-ng)
- Wake word: "hey shai" or "okay shai" to trigger

Commands:
  /q, /quit  - Exit
  /mute      - Toggle TTS on/off
  /wake      - Toggle wake word requirement
  /help      - Show help

Adjust VOSK_MODEL_PATH, WHISPER_CMD, and TTS_CMD for your system.
"""
from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

# Try to import Vosk
try:
    from vosk import Model, KaldiRecognizer
    VOSK_AVAILABLE = True
except ImportError:
    VOSK_AVAILABLE = False

BASE_URL = os.environ.get("SHAI_BASE_URL", "http://localhost:8000")
QA_ENDPOINT = f"{BASE_URL}/qa"

HOME = Path(os.environ.get("HOME", str(Path("~").expanduser()))).resolve()
VOICE_LOG_DIR = HOME / ".astraeus_voice_logs"
VOICE_LOG_DIR.mkdir(parents=True, exist_ok=True)

# Paths
REPO_DIR = Path(__file__).parent.resolve()

# Vosk model path - auto-select based on available memory
VOSK_MODEL_LARGE = REPO_DIR / "models" / "vosk-model-en-us-0.22"
VOSK_MODEL_SMALL = REPO_DIR / "models" / "vosk-model-small-en-us-0.15"

def get_available_memory_gb():
    """Get available memory in GB."""
    try:
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                if line.startswith('MemAvailable:'):
                    kb = int(line.split()[1])
                    return kb / (1024 * 1024)
    except:
        pass
    return 2.0  # Conservative default

# Large model needs ~3GB RAM to load, use small if less than 4GB available
available_mem = get_available_memory_gb()
if available_mem >= 4.0 and VOSK_MODEL_LARGE.exists():
    DEFAULT_VOSK = str(VOSK_MODEL_LARGE)
    print(f"[voice] {available_mem:.1f}GB RAM available - using LARGE Vosk model (better accuracy)")
elif VOSK_MODEL_SMALL.exists():
    DEFAULT_VOSK = str(VOSK_MODEL_SMALL)
    print(f"[voice] {available_mem:.1f}GB RAM available - using SMALL Vosk model (low memory mode)")
else:
    DEFAULT_VOSK = str(VOSK_MODEL_LARGE)
VOSK_MODEL_PATH = os.environ.get("ASTRAEUS_VOSK_MODEL", DEFAULT_VOSK)

# Recording parameters for Whisper fallback
ARECORD_CMD = [
    "arecord",
    "-f", "S16_LE",
    "-r", "16000",
    "-c", "1",
    "-d", "8",
]

# Whisper CLI (fallback)
WHISPER_CMD = os.environ.get("ASTRAEUS_WHISPER_CMD", "whisper")

# TTS command - default to Piper/Jenny, fallback to espeak-ng
PIPER_SPEAK = REPO_DIR / "piper-speak"
# Always prefer piper-speak for Jenny voice if available
if PIPER_SPEAK.exists():
    TTS_CMD = str(PIPER_SPEAK)  # Force piper-speak, ignore env var
    print(f"[voice] Using Piper TTS (Jenny): {TTS_CMD}")
else:
    TTS_CMD = os.environ.get("ASTRAEUS_TTS_CMD", "espeak-ng")
    print(f"[voice] Piper not found, using: {TTS_CMD}")

# Auto-speak responses
AUTO_SPEAK = os.environ.get("ASTRAEUS_AUTO_SPEAK", "1") == "1"

# Wake words (case insensitive)
WAKE_WORDS = ["hey shai", "okay shai", "hey shy", "okay shy", "a shai", "hey chai"]

# Silence timeout - how long to wait after speech stops before processing
SILENCE_TIMEOUT = 1.5  # seconds


class ContinuousVoiceListener:
    """Continuous voice listener using Vosk."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model: Optional[Model] = None
        self.recognizer: Optional[KaldiRecognizer] = None
        self.audio_queue: queue.Queue = queue.Queue()
        self.result_queue: queue.Queue = queue.Queue()
        self.running = False
        self.recording_thread: Optional[threading.Thread] = None
        self.processing_thread: Optional[threading.Thread] = None
        self.require_wake_word = True
        self.wake_word_detected = False
        self.partial_text = ""
        self.last_speech_time = 0.0

    def initialize(self) -> bool:
        """Initialize Vosk model."""
        if not VOSK_AVAILABLE:
            print("[voice] Vosk not available")
            return False

        if not Path(self.model_path).exists():
            print(f"[voice] Vosk model not found at {self.model_path}")
            return False

        try:
            print("[voice] Loading Vosk model...")
            self.model = Model(self.model_path)
            self.recognizer = KaldiRecognizer(self.model, 16000)
            self.recognizer.SetWords(True)
            print("[voice] Vosk model loaded successfully")
            return True
        except Exception as e:
            print(f"[voice] Failed to load Vosk model: {e}")
            return False

    def _record_audio(self):
        """Record audio using arecord and feed to queue."""
        import subprocess

        cmd = [
            "arecord",
            "-f", "S16_LE",
            "-r", "16000",
            "-c", "1",
            "-t", "raw",
            "-q",  # quiet
        ]

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL
            )

            while self.running:
                data = process.stdout.read(4000)  # ~125ms chunks
                if data:
                    self.audio_queue.put(data)
                else:
                    break

            process.terminate()
        except Exception as e:
            print(f"[voice] Recording error: {e}")
            self.running = False

    def _process_audio(self):
        """Process audio from queue with Vosk."""
        accumulated_text = ""
        last_final_time = time.time()

        while self.running:
            try:
                data = self.audio_queue.get(timeout=0.1)
            except queue.Empty:
                # Check for silence timeout
                if accumulated_text and (time.time() - last_final_time > SILENCE_TIMEOUT):
                    # Process accumulated text
                    self._handle_utterance(accumulated_text.strip())
                    accumulated_text = ""
                continue

            if self.recognizer.AcceptWaveform(data):
                result = json.loads(self.recognizer.Result())
                text = result.get("text", "").strip()

                if text:
                    accumulated_text += " " + text
                    last_final_time = time.time()
                    self.last_speech_time = time.time()
            else:
                # Partial result - show listening indicator
                partial = json.loads(self.recognizer.PartialResult())
                partial_text = partial.get("partial", "")
                if partial_text and partial_text != self.partial_text:
                    self.partial_text = partial_text
                    # Show partial on same line
                    print(f"\r[listening] {partial_text[:60]}..." + " " * 20, end="", flush=True)

    def _handle_utterance(self, text: str):
        """Handle a complete utterance."""
        if not text:
            return

        # Clear the listening line
        print("\r" + " " * 80 + "\r", end="", flush=True)

        # Check for wake word if required
        text_lower = text.lower()

        if self.require_wake_word:
            wake_found = False
            for wake in WAKE_WORDS:
                if wake in text_lower:
                    wake_found = True
                    # Remove wake word from query
                    idx = text_lower.find(wake)
                    text = text[idx + len(wake):].strip()
                    break

            if not wake_found:
                # No wake word, ignore
                return

        if text:
            self.result_queue.put(text)

    def start(self):
        """Start continuous listening."""
        if not self.model:
            return False

        self.running = True
        self.recording_thread = threading.Thread(target=self._record_audio, daemon=True)
        self.processing_thread = threading.Thread(target=self._process_audio, daemon=True)

        self.recording_thread.start()
        self.processing_thread.start()
        return True

    def stop(self):
        """Stop listening."""
        self.running = False
        if self.recording_thread:
            self.recording_thread.join(timeout=1)
        if self.processing_thread:
            self.processing_thread.join(timeout=1)

    def get_utterance(self, timeout: float = 0.1) -> Optional[str]:
        """Get next recognized utterance."""
        try:
            return self.result_queue.get(timeout=timeout)
        except queue.Empty:
            return None


def check_whisper_available() -> bool:
    """Check if Whisper is available."""
    try:
        result = subprocess.run(
            ["which", WHISPER_CMD],
            capture_output=True,
            check=False
        )
        return result.returncode == 0
    except Exception:
        return False


def check_arecord_available() -> bool:
    """Check if arecord is available."""
    try:
        result = subprocess.run(
            ["which", "arecord"],
            capture_output=True,
            check=False
        )
        return result.returncode == 0
    except Exception:
        return False


def record_audio_whisper(temp_path: Path) -> bool:
    """Record 8 seconds of audio for Whisper."""
    cmd = ARECORD_CMD + [str(temp_path)]
    print("[voice] Recording 8 seconds... (speak now)")
    try:
        subprocess.run(cmd, check=True, stderr=subprocess.DEVNULL)
        print("[voice] Recording complete.")
        return True
    except Exception as e:
        print(f"[voice] ERROR: recording failed: {e}")
        return False


def run_whisper(temp_wav: Path) -> Optional[str]:
    """Run Whisper on audio file."""
    try:
        proc = subprocess.run(
            [WHISPER_CMD, str(temp_wav)],
            text=True,
            capture_output=True,
            check=False,
        )
    except FileNotFoundError:
        print("[voice] ERROR: Whisper command not found.")
        return None

    if proc.returncode != 0:
        print("[voice] Whisper error:")
        print(proc.stderr, file=sys.stderr)
        return None

    text = proc.stdout.strip()
    if not text:
        print("[voice] Whisper produced empty transcript.")
        return None

    return text


def qa_request(query: str) -> Dict[str, Any]:
    """Send query to SHAI."""
    payload: Dict[str, Any] = {
        "query": query,
        "top_k": 8,
        "namespace": "all",
        "model": "tinyllama",  # Use smaller model for low memory
        "use_tools": False,  # Disable for faster response
        "use_chaos": False,
        "chaos_level": 1,
        "use_paradox": False,  # Disable for faster response
        "use_nightside": False,  # Disable for faster response
    }
    llm_endpoint = os.environ.get("ASTRAEUS_LLM_ENDPOINT", "http://localhost:11434")
    if llm_endpoint:
        payload["llm_endpoint"] = llm_endpoint

    resp = requests.post(QA_ENDPOINT, json=payload, timeout=600)
    resp.raise_for_status()
    return resp.json()


def extract_speakable_lines(answer: str) -> List[str]:
    """Extract key lines from answer for TTS."""
    lines = answer.splitlines()
    speak: List[str] = []
    in_summary = False

    for ln in lines:
        stripped = ln.strip()
        if stripped.startswith("SUMMARY:"):
            in_summary = True
            speak.append(stripped)
            continue
        if stripped.startswith("Recovery Complete:"):
            in_summary = True
            speak.append(stripped)
            continue
        if in_summary:
            speak.append(stripped)
            continue
        if stripped.startswith("* "):
            speak.append(stripped)

    # If no structured content, speak first few lines
    if not speak and lines:
        for line in lines:
            if line.strip():
                speak.append(line.strip())
                if sum(len(s) for s in speak) > 500:
                    break

    return speak


def tts_speak(text: str) -> None:
    """Speak text using TTS."""
    if not text.strip():
        return

    # Clean text for TTS
    text = text.replace('"', '').replace("'", "")

    try:
        if "piper" in TTS_CMD.lower():
            # Piper takes text via stdin
            subprocess.run(
                [TTS_CMD, text],
                check=False,
                stderr=subprocess.DEVNULL
            )
        elif "espeak" in TTS_CMD:
            subprocess.run(
                [TTS_CMD, "-s", "150", "-p", "50", text],
                check=False,
                stderr=subprocess.DEVNULL
            )
        else:
            subprocess.run([TTS_CMD, text], check=False)
    except FileNotFoundError:
        print("[voice] WARNING: TTS command not found. Install espeak-ng or piper.")


def log_interaction(query: str, response: Dict[str, Any]) -> None:
    """Log interaction to file."""
    ts = int(time.time())
    log_path = VOICE_LOG_DIR / f"session_{time.strftime('%Y%m%d')}.jsonl"
    record = {"ts": ts, "query": query, "response": response}
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def process_query(query: str, auto_speak: bool) -> None:
    """Process a query and display/speak the response."""
    print(f"\nyou> {query}")
    print("\n[thinking...]")

    try:
        data = qa_request(query)
    except Exception as e:
        error_msg = f"Error contacting SHAI: {e}"
        print(f"[voice] {error_msg}")
        if auto_speak:
            tts_speak("Error contacting SHAI.")
        return

    # Display response
    answer = data.get("answer") or data.get("prompt") or ""
    if not answer:
        answer = json.dumps(data, ensure_ascii=False, indent=2)

    print("\nshai>")
    print(answer)

    # Speak response
    if auto_speak and answer:
        speak_lines = extract_speakable_lines(answer)
        if speak_lines:
            to_say = " ".join(speak_lines)
            tts_speak(to_say)

    log_interaction(query, data)


def main() -> int:
    print("=" * 60)
    print("SHAI CONTINUOUS VOICE REPL")
    print("=" * 60)

    has_arecord = check_arecord_available()

    # Try to initialize Vosk for continuous listening
    listener: Optional[ContinuousVoiceListener] = None
    vosk_mode = False

    if VOSK_AVAILABLE and has_arecord:
        listener = ContinuousVoiceListener(VOSK_MODEL_PATH)
        if listener.initialize():
            vosk_mode = True
            print("Voice input: CONTINUOUS (Vosk - always listening)")
            print(f"Wake words: {', '.join(WAKE_WORDS[:3])}...")
        else:
            listener = None

    # Fallback to Whisper
    has_whisper = check_whisper_available()
    whisper_mode = not vosk_mode and has_whisper and has_arecord

    if whisper_mode:
        print("Voice input: WHISPER (press Enter to record 8 seconds)")
    elif not vosk_mode:
        if not has_arecord:
            print("Voice input: DISABLED (arecord not found)")
        else:
            print("Voice input: DISABLED (no STT engine available)")

    print("Text input: Type your query and press Enter")
    print("TTS output: Responses will be spoken aloud")
    print("Commands: /q = quit, /mute = toggle TTS, /wake = toggle wake word, /help")
    print("=" * 60)

    temp_wav = Path("/tmp/astraeus_voice.wav")
    auto_speak = AUTO_SPEAK
    require_wake = True

    # Greeting
    if auto_speak:
        if vosk_mode:
            tts_speak("SHAI voice interface ready. Say hey shai to begin.")
        else:
            tts_speak("SHAI voice interface ready.")

    # Start continuous listening if available
    if vosk_mode and listener:
        listener.require_wake_word = require_wake
        listener.start()
        print("\n[Listening continuously... say 'hey shai' followed by your question]")

    try:
        while True:
            # Check for voice input from continuous listener
            if vosk_mode and listener:
                utterance = listener.get_utterance(timeout=0.05)
                if utterance:
                    process_query(utterance, auto_speak)
                    print("\n[Listening...]")
                    continue

            # Check for keyboard input (non-blocking in vosk mode)
            if vosk_mode:
                import select
                if select.select([sys.stdin], [], [], 0.05)[0]:
                    try:
                        line = sys.stdin.readline().strip()
                    except:
                        continue
                else:
                    continue
            else:
                # Blocking input for non-vosk mode
                try:
                    line = input("\nshai> ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\n[voice] Goodbye.")
                    if auto_speak:
                        tts_speak("Goodbye.")
                    return 0

            # Handle commands
            if line == "/q" or line == "/quit":
                print("[voice] Goodbye.")
                if auto_speak:
                    tts_speak("Goodbye.")
                return 0

            if line == "/mute":
                auto_speak = not auto_speak
                status = "enabled" if auto_speak else "disabled"
                print(f"[voice] TTS {status}")
                continue

            if line == "/wake":
                require_wake = not require_wake
                if listener:
                    listener.require_wake_word = require_wake
                status = "required" if require_wake else "not required"
                print(f"[voice] Wake word {status}")
                if not require_wake:
                    print("[voice] All speech will now be processed as queries")
                continue

            if line == "/help":
                print("\nCommands:")
                print("  /q, /quit  - Exit")
                print("  /mute      - Toggle TTS on/off")
                print("  /wake      - Toggle wake word requirement")
                print("  /help      - Show this help")
                print("\nUsage:")
                print("  - Type a question and press Enter")
                if vosk_mode:
                    print("  - Or speak: 'hey shai, [your question]'")
                    print("  - Use /wake to disable wake word requirement")
                elif whisper_mode:
                    print("  - Press Enter alone to record voice (8 seconds)")
                continue

            # Whisper fallback: empty line = record
            if not line and whisper_mode:
                if not record_audio_whisper(temp_wav):
                    continue

                transcript = run_whisper(temp_wav)
                if not transcript:
                    print("[voice] Could not transcribe. Try typing instead.")
                    continue

                process_query(transcript, auto_speak)
                continue
            elif not line and not vosk_mode:
                print("[voice] Type a query or install Vosk/Whisper for voice input.")
                continue
            elif line:
                # Text input
                process_query(line, auto_speak)

    except KeyboardInterrupt:
        print("\n[voice] Goodbye.")
        if auto_speak:
            tts_speak("Goodbye.")
    finally:
        if listener:
            listener.stop()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
