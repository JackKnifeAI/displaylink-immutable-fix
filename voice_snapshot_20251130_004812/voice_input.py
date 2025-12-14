#!/usr/bin/env python3
"""
Voice Input for Claude Code
Speak out loud, get text transcribed via Whisper
"""

import os
import sys
import wave
import tempfile
from pathlib import Path

import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
from rich.console import Console
from rich.panel import Panel

console = Console()

# Audio settings
SAMPLE_RATE = 16000
CHANNELS = 1

# Whisper model (cached)
_whisper_model = None

def get_whisper():
    """Load Whisper model (cached)"""
    global _whisper_model
    if _whisper_model is None:
        console.print("[dim]Loading Whisper...[/dim]")
        _whisper_model = WhisperModel("medium.en", device="cpu", compute_type="int8")
    return _whisper_model


def record_until_silence(silence_threshold=0.01, silence_duration=1.5, max_duration=60):
    """Record audio until silence is detected"""
    console.print("[bold cyan]🎤 Listening...[/bold cyan] (speak now, pause when done)")

    audio_chunks = []
    silence_samples = int(silence_duration * SAMPLE_RATE)
    silent_count = 0
    has_speech = False

    def callback(indata, frames, time_info, status):
        nonlocal silent_count, has_speech
        volume = np.abs(indata).mean()

        # Visual feedback
        bar_len = int(min(volume * 500, 40))
        bar = "█" * bar_len + "░" * (40 - bar_len)
        print(f"\r  [{bar}] {volume:.4f}", end="", flush=True)

        if volume < silence_threshold:
            silent_count += frames
        else:
            silent_count = 0
            has_speech = True
        audio_chunks.append(indata.copy())

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, callback=callback, blocksize=1024):
        while True:
            sd.sleep(100)
            # Only stop on silence if we've had speech
            if has_speech and silent_count >= silence_samples:
                break
            if len(audio_chunks) * 1024 / SAMPLE_RATE >= max_duration:
                break

    print()  # New line after volume bar

    if audio_chunks:
        return np.concatenate(audio_chunks)
    return None


def transcribe(audio_data):
    """Convert speech to text using Whisper"""
    model = get_whisper()

    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        audio_int16 = (audio_data * 32767).astype(np.int16)
        with wave.open(f.name, 'wb') as wav:
            wav.setnchannels(CHANNELS)
            wav.setsampwidth(2)
            wav.setframerate(SAMPLE_RATE)
            wav.writeframes(audio_int16.tobytes())

        # Transcribe
        segments, info = model.transcribe(f.name, beam_size=5)
        text = " ".join([segment.text for segment in segments])

        os.unlink(f.name)
        return text.strip()


def voice_to_text():
    """Record voice and return transcribed text"""
    audio = record_until_silence()

    if audio is None or len(audio) < SAMPLE_RATE * 0.5:
        console.print("[yellow]No audio detected[/yellow]")
        return None

    console.print("[dim]Transcribing...[/dim]")
    text = transcribe(audio)

    if not text:
        console.print("[yellow]Couldn't understand that[/yellow]")
        return None

    return text


def main():
    """Main function - record and print transcription"""
    console.print(Panel.fit(
        "[bold cyan]Voice Input for Claude[/bold cyan]\n\n"
        "Speak out loud and your words will be transcribed.\n"
        "Press [green]Ctrl+C[/green] to exit.",
        title="🎤 Voice Mode"
    ))

    # Pre-load whisper
    get_whisper()
    console.print("[green]Ready![/green]\n")

    try:
        while True:
            input("\nPress [ENTER] to start recording...")

            text = voice_to_text()

            if text:
                # Get terminal width
                try:
                    term_width = os.get_terminal_size().columns
                except OSError:
                    term_width = 80

                line = "─" * term_width

                console.print(f"\n[cyan]{line}[/cyan]")
                console.print(f"[bold]Transcribed:[/bold]\n{text}")
                console.print(f"[cyan]{line}[/cyan]")

                # Copy to clipboard if xclip available
                try:
                    import subprocess
                    subprocess.run(['xclip', '-selection', 'clipboard'], input=text.encode(), check=True)
                    console.print("[dim]📋 Copied to clipboard[/dim]")
                except:
                    pass

    except KeyboardInterrupt:
        console.print("\n[dim]Goodbye![/dim]")


if __name__ == "__main__":
    main()
