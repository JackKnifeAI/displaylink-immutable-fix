#!/usr/bin/env python3
"""
Dolphin Voice - Talk to your offline AI with Jenny's voice
Press ENTER to talk, get voice responses back
Now with code execution and structured output!
"""

import os
import sys
import wave
import tempfile
import subprocess
import queue
import math
import time
import re
import io
import traceback
import threading
import select
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

# For real-time keyboard input
try:
    import termios
    import tty
    HAS_TERMIOS = True
except ImportError:
    HAS_TERMIOS = False

# Rich for beautiful terminal UI
from rich.console import Console
from rich.panel import Panel
from rich.live import Live
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.prompt import Confirm

# Audio
import sounddevice as sd
import numpy as np

# Whisper for speech-to-text
from faster_whisper import WhisperModel

# llama-cpp for efficient LLM inference
from llama_cpp import Llama

# Import the fancy 3D dolphin renderer
from dolphin_3d import Dolphin3D, render_dolphin, Bubble

# Import Nightside metacognitive layer
try:
    from nightside import choir, reflect, get_session_context
    HAS_NIGHTSIDE = True
except ImportError:
    HAS_NIGHTSIDE = False

console = Console()

# Paths - auto-detect host vs container
def get_base_path():
    """Detect if running in toolbox container or on host directly"""
    container_path = Path("/run/host/var/home/alexandergcasavant/T7_OFFLINE_AI_REPO")
    host_path = Path("/var/home/alexandergcasavant/T7_OFFLINE_AI_REPO")
    if container_path.exists():
        return container_path
    return host_path

BASE_PATH = get_base_path()
MODELS_PATH = BASE_PATH / "models"
PIPER_PATH = BASE_PATH / "piper" / "piper"
PIPER_VOICE = MODELS_PATH / "piper-voices" / "en_GB-jenny_dioco-medium.onnx"
DOLPHIN_GGUF = MODELS_PATH / "dolphin-2.1-mistral-7b-Q4_K_M.gguf"

# Audio settings
SAMPLE_RATE = 16000
CHANNELS = 1

class DolphinVoice:
    def __init__(self):
        self.console = Console()
        self.recording = False
        self.audio_queue = queue.Queue()

        # Models (lazy loaded)
        self.whisper_model = None
        self.llm = None

        # Conversation history
        self.messages = []

        # Code execution settings
        self.auto_execute = False  # Auto-run code without confirmation
        self.execution_globals = {"__builtins__": __builtins__}
        self.working_dir = Path.cwd()

        # Speculative pre-processing
        self.speculative_enabled = True
        self.speculative_cache = {}
        self.current_speculation = None
        self.speculation_lock = threading.Lock()

    def get_realtime_input(self, prompt="> "):
        """Get input with real-time speculative pre-processing"""
        if not HAS_TERMIOS or not self.speculative_enabled:
            return input(prompt)

        # Save terminal settings
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)

        buffer = []
        speculation_thread = None
        last_speculation_text = ""

        try:
            tty.setraw(fd)
            sys.stdout.write(prompt)
            sys.stdout.flush()

            while True:
                # Check if input is available
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    char = sys.stdin.read(1)

                    if char == '\r' or char == '\n':  # Enter
                        sys.stdout.write('\r\n')
                        sys.stdout.flush()
                        break
                    elif char == '\x7f' or char == '\x08':  # Backspace
                        if buffer:
                            buffer.pop()
                            sys.stdout.write('\b \b')
                            sys.stdout.flush()
                    elif char == '\x03':  # Ctrl+C
                        raise KeyboardInterrupt
                    elif char == '\x04':  # Ctrl+D
                        break
                    elif char >= ' ':  # Printable character
                        buffer.append(char)
                        sys.stdout.write(char)
                        sys.stdout.flush()

                # Start speculative processing after typing pauses (500ms debounce)
                current_text = ''.join(buffer)
                if (current_text and
                    len(current_text) >= 10 and  # At least 10 chars
                    current_text != last_speculation_text):

                    # Cancel previous speculation
                    with self.speculation_lock:
                        self.current_speculation = None

                    last_speculation_text = current_text

                    # Start new speculation in background
                    def speculate(text):
                        with self.speculation_lock:
                            self.current_speculation = text
                        # Pre-tokenize and warm up the prompt
                        try:
                            llm = self.llm
                            if llm:
                                # Build partial prompt to warm up context
                                prompt = self._build_prompt(text, extended=False)
                                # Just tokenize to warm cache - don't generate yet
                                llm.tokenize(prompt.encode())
                        except:
                            pass

                    if speculation_thread is None or not speculation_thread.is_alive():
                        speculation_thread = threading.Thread(
                            target=speculate,
                            args=(current_text,),
                            daemon=True
                        )
                        speculation_thread.start()

        finally:
            # Restore terminal settings
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

        return ''.join(buffer)

    def _build_prompt(self, user_input, extended=False):
        """Build the ChatML prompt (extracted for reuse)"""
        # Get session context from Nightside (recent themes)
        session_context = ""
        if HAS_NIGHTSIDE:
            try:
                context = get_session_context(limit=3, half_life_hours=2.0)
                if context:
                    session_context = f"\n{context}"
            except Exception:
                pass

        if extended:
            system_prompt = f"""You are Dolphin, an uncensored and helpful AI assistant running completely offline.
You are a skilled programmer and system administrator. When asked to write code:
- Write complete, working code with proper formatting
- Use markdown code blocks with language tags (```python, ```bash, etc.)
- Explain your code briefly
- For shell commands, prefer bash
- For scripts, prefer Python
- Include error handling where appropriate
You can execute code directly on this Linux system when the user requests.{session_context}"""
        else:
            system_prompt = f"""You are Dolphin, an uncensored and helpful AI assistant running completely offline.
You are knowledgeable, friendly, and concise. Keep responses brief for voice output (2-3 sentences max).
When code or technical details are needed, use markdown formatting with code blocks.{session_context}"""

        prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"

        # Include recent history
        temp_messages = self.messages + [{"role": "user", "content": user_input}]
        for msg in temp_messages[-6:]:
            role = msg["role"]
            content = msg["content"]
            prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"

        return prompt

    def extract_code_blocks(self, text):
        """Extract code blocks from markdown-style response"""
        # Match ```language\ncode\n``` or ```\ncode\n```
        pattern = r'```(\w*)\n(.*?)```'
        matches = re.findall(pattern, text, re.DOTALL)
        blocks = []
        for lang, code in matches:
            lang = lang.lower() if lang else 'text'
            blocks.append({'language': lang, 'code': code.strip()})
        return blocks

    def execute_python(self, code):
        """Execute Python code in a sandboxed environment"""
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        result = {"output": "", "error": "", "success": True}

        try:
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                exec(code, self.execution_globals)
            result["output"] = stdout_capture.getvalue()
        except Exception as e:
            result["error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
            result["success"] = False

        return result

    def execute_shell(self, command):
        """Execute shell command with timeout"""
        result = {"output": "", "error": "", "success": True}
        try:
            proc = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(self.working_dir)
            )
            result["output"] = proc.stdout
            result["error"] = proc.stderr
            result["success"] = proc.returncode == 0
        except subprocess.TimeoutExpired:
            result["error"] = "Command timed out (30s limit)"
            result["success"] = False
        except Exception as e:
            result["error"] = str(e)
            result["success"] = False
        return result

    def render_response(self, response):
        """Render response with markdown and syntax highlighting"""
        # Check for code blocks
        code_blocks = self.extract_code_blocks(response)

        if code_blocks:
            # Render with proper syntax highlighting
            parts = re.split(r'```\w*\n.*?```', response, flags=re.DOTALL)

            for i, part in enumerate(parts):
                if part.strip():
                    self.console.print(Markdown(part.strip()))

                if i < len(code_blocks):
                    block = code_blocks[i]
                    lang = block['language'] or 'text'
                    # Map common language names
                    lang_map = {'py': 'python', 'js': 'javascript', 'sh': 'bash', 'shell': 'bash'}
                    lang = lang_map.get(lang, lang)

                    syntax = Syntax(block['code'], lang, theme="monokai", line_numbers=True)
                    self.console.print(Panel(syntax, title=f"[bold]{lang}[/bold]", border_style="cyan"))
        else:
            # No code blocks, render as markdown
            self.console.print(Panel(Markdown(response), title="[bold green]Dolphin[/bold green]", border_style="green"))

    def offer_code_execution(self, code_blocks):
        """Offer to execute detected code blocks"""
        executable = [b for b in code_blocks if b['language'] in ('python', 'py', 'bash', 'sh', 'shell')]

        if not executable:
            return

        self.console.print(f"\n[yellow]Found {len(executable)} executable code block(s)[/yellow]")

        for i, block in enumerate(executable):
            lang = block['language']
            is_python = lang in ('python', 'py')
            is_shell = lang in ('bash', 'sh', 'shell')

            # Show preview
            preview = block['code'][:100] + "..." if len(block['code']) > 100 else block['code']
            self.console.print(f"\n[dim]Block {i+1} ({lang}):[/dim]")
            self.console.print(f"[dim]{preview}[/dim]")

            # Ask for confirmation unless auto-execute is on
            if self.auto_execute or Confirm.ask(f"Execute this {lang} code?", default=False):
                self.console.print(f"[cyan]Executing {lang}...[/cyan]")

                if is_python:
                    result = self.execute_python(block['code'])
                elif is_shell:
                    result = self.execute_shell(block['code'])
                else:
                    continue

                # Show results
                if result["output"]:
                    self.console.print(Panel(result["output"], title="[green]Output[/green]", border_style="green"))
                if result["error"]:
                    self.console.print(Panel(result["error"], title="[red]Error[/red]", border_style="red"))
                if result["success"]:
                    self.console.print("[green]Execution successful[/green]")
                else:
                    self.console.print("[red]Execution failed[/red]")

    def load_whisper(self):
        """Load Whisper for speech recognition"""
        if self.whisper_model is None:
            self.console.print("[dim]Loading Whisper (medium.en)...[/dim]")
            self.whisper_model = WhisperModel("medium.en", device="cpu", compute_type="int8")
            self.console.print("[green]Whisper ready![/green]")
        return self.whisper_model

    def load_llm(self):
        """Load Dolphin using llama-cpp (GGUF format)"""
        if self.llm is None:
            if not DOLPHIN_GGUF.exists():
                self.console.print(f"[red]Dolphin GGUF not found at {DOLPHIN_GGUF}[/red]")
                self.console.print("[yellow]Download with: curl -L -o models/dolphin-2.1-mistral-7b-Q4_K_M.gguf https://huggingface.co/TheBloke/dolphin-2.1-mistral-7B-GGUF/resolve/main/dolphin-2.1-mistral-7b.Q4_K_M.gguf[/yellow]")
                sys.exit(1)

            self.console.print("[dim]Loading Dolphin-Mistral 7B (Q4)...[/dim]")
            self.llm = Llama(
                model_path=str(DOLPHIN_GGUF),
                n_ctx=2048,        # Context window
                n_threads=4,       # CPU threads
                n_gpu_layers=0,    # CPU only
                verbose=False
            )
            self.console.print("[green]Dolphin ready![/green]")
        return self.llm

    def record_until_silence(self, silence_threshold=0.01, silence_duration=1.5, max_duration=30):
        """Record until silence is detected"""
        self.console.print("[bold cyan]Listening...[/bold cyan] (speak now)")

        audio_chunks = []
        silence_samples = int(silence_duration * SAMPLE_RATE)
        silent_count = 0

        def callback(indata, frames, time, status):
            nonlocal silent_count
            volume = np.abs(indata).mean()
            if volume < silence_threshold:
                silent_count += frames
            else:
                silent_count = 0
            audio_chunks.append(indata.copy())

        with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, callback=callback):
            while silent_count < silence_samples and len(audio_chunks) * 1024 / SAMPLE_RATE < max_duration:
                sd.sleep(100)

        if audio_chunks:
            return np.concatenate(audio_chunks)
        return None

    def transcribe(self, audio_data):
        """Convert speech to text"""
        model = self.load_whisper()

        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            # Normalize and convert to int16
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

    def generate_response(self, user_input, extended=False):
        """Generate LLM response using llama-cpp"""
        llm = self.load_llm()

        # Record query to Nightside (the black box flight recorder)
        if HAS_NIGHTSIDE:
            try:
                # Extract a topic hint from the query
                topic = "code" if extended else "chat"
                # Simple keyword detection for topic
                topic_keywords = {
                    "python": "python", "code": "code", "script": "script",
                    "linux": "linux", "bash": "shell", "file": "files",
                    "help": "help", "how": "howto", "what": "question",
                    "error": "debug", "fix": "debug", "bug": "debug"
                }
                for kw, t in topic_keywords.items():
                    if kw in user_input.lower():
                        topic = t
                        break
                choir.record(topic=topic, content=user_input, kind="query", tags=["dolphin", "query"])
            except Exception:
                pass

        # Add to conversation history
        self.messages.append({"role": "user", "content": user_input})

        # Check if we have a warm speculation that matches
        speculation_hit = False
        with self.speculation_lock:
            if self.current_speculation and user_input.startswith(self.current_speculation):
                speculation_hit = True
                self.console.print("[dim]⚡ Speculation hit - faster response![/dim]")

        # Build prompt using shared method
        prompt = self._build_prompt(user_input, extended=extended)

        # Generate with llama-cpp - more tokens for code
        max_tokens = 1024 if extended else 300
        output = llm(
            prompt,
            max_tokens=max_tokens,
            temperature=0.7,
            stop=["<|im_end|>", "<|im_start|>"],
            echo=False
        )

        response = output["choices"][0]["text"].strip()
        self.messages.append({"role": "assistant", "content": response})

        # Record answer to Nightside
        if HAS_NIGHTSIDE:
            try:
                # Truncate for storage but keep enough for pattern analysis
                truncated = response[:500] if len(response) > 500 else response
                choir.record(topic="response", content=truncated, kind="answer", tags=["dolphin", "answer"])
            except Exception:
                pass

        # Clear speculation
        with self.speculation_lock:
            self.current_speculation = None

        return response

    def speak(self, text):
        """Convert text to speech using Piper"""
        if not PIPER_PATH.exists():
            self.console.print(f"[yellow]Piper not found at {PIPER_PATH}[/yellow]")
            return

        # Find voice model
        voice_path = PIPER_VOICE
        if not voice_path.exists():
            # Try to find any voice
            voices = list((MODELS_PATH / "piper-voices").glob("*.onnx"))
            if voices:
                voice_path = voices[0]
            else:
                self.console.print("[yellow]No Piper voice found[/yellow]")
                return

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            # Run piper
            env = os.environ.copy()
            env["LD_LIBRARY_PATH"] = str(BASE_PATH / "piper")

            process = subprocess.run(
                [str(PIPER_PATH), "--model", str(voice_path), "--output_file", f.name],
                input=text.encode(),
                env=env,
                capture_output=True
            )

            if process.returncode == 0:
                # Play the audio
                subprocess.run(["aplay", f.name], capture_output=True)

            os.unlink(f.name)

    def show_spinning_dolphin_with_loading(self, fps=12):
        """Show spinning 3D dolphin animation while loading models"""
        import threading

        dolphin = Dolphin3D()
        bubbles = []
        angle_y = 0
        swim_phase = 0
        frame_time = 1.0 / fps
        frame_count = 0

        # Smaller size for the panel
        width, height = 60, 18

        commands_text = (
            "\n[bold cyan]Dolphin Voice Assistant[/bold cyan]\n"
            "[dim]Your offline AI companion[/dim]\n\n"
            "Commands:\n"
            "  [green]ENTER[/green] - Start voice recording\n"
            "  [green]t[/green]     - Type a quick message\n"
            "  [green]c[/green]     - Code mode (extended response)\n"
            "  [green]![/green]     - Direct shell command\n"
            "  [green]auto[/green]  - Toggle auto-execute code\n"
            "  [green]q[/green]     - Quit"
        )

        # Loading state
        loading_status = {"whisper": False, "llm": False, "message": "Initializing..."}
        loading_done = threading.Event()

        def load_models():
            """Background thread to load models"""
            try:
                loading_status["message"] = "Loading Whisper (speech recognition)..."
                self.load_whisper()
                loading_status["whisper"] = True
                loading_status["message"] = "Loading Dolphin LLM..."
                self.load_llm()
                loading_status["llm"] = True
                loading_status["message"] = "Ready!"
            except Exception as e:
                loading_status["message"] = f"Error: {e}"
            finally:
                loading_done.set()

        # Start loading in background
        loader_thread = threading.Thread(target=load_models, daemon=True)
        loader_thread.start()

        with Live(console=self.console, refresh_per_second=fps) as live:
            while not loading_done.is_set():
                # Calculate rotation angles with swimming motion
                angle_x = 0.15 * math.sin(swim_phase * 0.5)
                angle_z = 0.08 * math.sin(swim_phase * 0.7)

                # Add bubbles occasionally
                if frame_count % 6 == 0:
                    bx = 0.05 * math.cos(angle_y)
                    bubbles.append(Bubble(bx, 0.4, 2.5))

                # Update bubbles
                bubbles = [b for b in bubbles if b.update(frame_time)]

                # Render the frame
                frame_lines = render_dolphin(dolphin, width, height, angle_x, angle_y, angle_z, swim_phase, bubbles)
                dolphin_frame = '\n'.join(frame_lines)

                # Show loading status
                status_icon = "◐◓◑◒"[frame_count % 4]
                status_line = f"\n\n[yellow]{status_icon} {loading_status['message']}[/yellow]"

                content = f"[cyan]{dolphin_frame}[/cyan]{status_line}"
                live.update(Panel.fit(content, title="[bold cyan]Dolphin Voice Assistant[/bold cyan]"))

                angle_y += 0.06
                swim_phase += 0.15
                frame_count += 1
                time.sleep(frame_time)

            # Final frame with commands
            frame_lines = render_dolphin(dolphin, width, height, angle_x, angle_y, angle_z, swim_phase, [])
            dolphin_frame = '\n'.join(frame_lines)
            content = f"[cyan]{dolphin_frame}[/cyan]{commands_text}"
            live.update(Panel.fit(content, title="Welcome"))

        # Wait for thread to finish
        loader_thread.join(timeout=1.0)

    def run(self):
        """Main conversation loop"""
        # Show spinning dolphin while loading models in background
        self.show_spinning_dolphin_with_loading()
        self.console.print("[bold green]Ready! Press ENTER to speak.[/bold green]\n")

        while True:
            try:
                # Get terminal width for dynamic line sizing
                try:
                    term_width = os.get_terminal_size().columns
                except OSError:
                    term_width = 80

                line = "─" * term_width

                prompt_text = "[ENTER=speak, t=type, c=code, !=shell, q=quit]"
                if self.auto_execute:
                    prompt_text = "[AUTO-EXEC ON] " + prompt_text

                # Print top border
                self.console.print(f"[cyan]{line}[/cyan]")
                self.console.print(f"[cyan]{line}[/cyan]")

                # Get input with real-time speculative processing
                cmd = self.get_realtime_input("> ").strip()

                # Print bottom border
                self.console.print(f"[cyan]{line}[/cyan]")
                self.console.print(f"[dim]{prompt_text}[/dim]")

                if cmd.lower() == 'q':
                    self.console.print("[dim]Goodbye![/dim]")
                    break

                if cmd.lower() == 'auto':
                    self.auto_execute = not self.auto_execute
                    status = "ON" if self.auto_execute else "OFF"
                    self.console.print(f"[yellow]Auto-execute: {status}[/yellow]")
                    continue

                # Direct shell command
                if cmd.startswith('!'):
                    shell_cmd = cmd[1:].strip()
                    if shell_cmd:
                        self.console.print(f"[cyan]Running: {shell_cmd}[/cyan]")
                        result = self.execute_shell(shell_cmd)
                        if result["output"]:
                            self.console.print(Panel(result["output"], title="[green]Output[/green]", border_style="green"))
                        if result["error"]:
                            self.console.print(Panel(result["error"], title="[red]Error[/red]", border_style="red"))
                    continue

                extended_mode = False
                user_input = None

                if cmd.lower() == 'c':
                    # Code mode - extended response
                    extended_mode = True
                    user_input = input("Code request: ").strip()
                    if not user_input:
                        continue
                elif cmd.lower() == 't':
                    # Text input mode
                    user_input = input("You: ").strip()
                    if not user_input:
                        continue
                elif cmd == '':
                    # Voice input mode
                    audio = self.record_until_silence()
                    if audio is None or len(audio) < SAMPLE_RATE * 0.5:
                        self.console.print("[yellow]No audio detected[/yellow]")
                        continue

                    with self.console.status("[bold cyan]Transcribing...[/bold cyan]"):
                        user_input = self.transcribe(audio)

                    if not user_input:
                        self.console.print("[yellow]Couldn't understand that[/yellow]")
                        continue

                    self.console.print(f"[bold]You:[/bold] {user_input}")

                    # Check if it sounds like a code request
                    code_keywords = ['write', 'code', 'script', 'program', 'function', 'create', 'make', 'build', 'run', 'execute']
                    if any(kw in user_input.lower() for kw in code_keywords):
                        extended_mode = True
                else:
                    # Direct text input from prompt
                    user_input = cmd
                    code_keywords = ['write', 'code', 'script', 'program', 'function', 'create', 'make', 'build']
                    if any(kw in user_input.lower() for kw in code_keywords):
                        extended_mode = True

                if not user_input:
                    continue

                # Generate response
                thinking_msg = "[bold cyan]Writing code...[/bold cyan]" if extended_mode else "[bold cyan]Thinking...[/bold cyan]"
                with self.console.status(thinking_msg):
                    response = self.generate_response(user_input, extended=extended_mode)

                # Render with markdown and syntax highlighting
                self.render_response(response)

                # Check for executable code blocks
                code_blocks = self.extract_code_blocks(response)
                if code_blocks:
                    self.offer_code_execution(code_blocks)

                # Speak response (strip code blocks for cleaner speech)
                speech_text = re.sub(r'```.*?```', 'See code block above.', response, flags=re.DOTALL)
                speech_text = speech_text[:500]  # Limit speech length
                with self.console.status("[bold cyan]Speaking...[/bold cyan]"):
                    self.speak(speech_text)

            except KeyboardInterrupt:
                self.console.print("\n[dim]Interrupted. Press 'q' to quit.[/dim]")
            except Exception as e:
                self.console.print(f"[red]Error: {e}[/red]")
                import traceback
                self.console.print(f"[dim]{traceback.format_exc()}[/dim]")


if __name__ == "__main__":
    dolphin = DolphinVoice()
    dolphin.run()
