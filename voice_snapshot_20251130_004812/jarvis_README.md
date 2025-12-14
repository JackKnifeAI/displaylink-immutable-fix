# Jarvis-ALX

Jarvis-ALX is a multi-service local AI assistant stack that runs entirely on your own hardware. It combines an orchestrator, episodic memory, hardware agent, voice gateway, and a Next.js dashboard.

## Stack Overview

| Service | Description |
| --- | --- |
| `llm_core` | OpenAI-compatible vLLM backend (external image). |
| `memory_service` | Chroma-backed episodic memory API. |
| `hw_agent` | FastAPI daemon that manages RF relays, sensors, and shutdowns. |
| `orchestrator` | FastAPI brain that routes between LLM, memory, and hardware. |
| `voice_gateway` | Offline STT/TTS bridge with hotword support. |
| `web_ui` | Next.js dashboard for status, chat, and controls. |
| `termux/jarvisctl.py` | Mobile CLI for quick control. |

Every service enforces the `X-Internal-Token` header. You can set a single string via `JARVIS_INTERNAL_TOKEN` **or** give each service its own secret (`ORCHESTRATOR_TOKEN`, `HW_AGENT_TOKEN`, `MEMORY_SERVICE_TOKEN`, `VOICE_GATEWAY_TOKEN`). The orchestrator uses the downstream vars to authenticate to the hardware + memory APIs. The dashboard/Termux clients should use `NEXT_PUBLIC_JARVIS_TOKEN` / `JARVIS_TOKEN`.

## Quick Start

```bash
cp .env.example .env
# edit JARVIS_INTERNAL_TOKEN and paths
# Preferred portable start (no manual compose):
# - runs hardware preflight
# - auto-selects models based on GPU VRAM (writes .env.autoselect)
# - uses detected model store (e.g., Samsung_T7/models)
./scripts/start_portable.sh

# Legacy start
# ./scripts/dev_up.sh
# runs a hardware preflight; set SKIP_PREFLIGHT=1 to bypass
# set AUTO_SELECT_MODELS=1 to emit .env.autoselect with recommended model/env overrides
```

Point your browser to http://localhost:3000 for the dashboard. The orchestrator API stays on http://localhost:8080.

Termux client example:

```bash
export JARVIS_HOST=192.168.1.50
export JARVIS_PORT=8080
export JARVIS_TOKEN=YourSharedToken
python termux/jarvisctl.py status
```

## Host Setup

For a fresh Ubuntu host with NVIDIA GPUs, run:

```bash
sudo ./scripts/host_install_ubuntu.sh
```

This installs Docker, NVIDIA container toolkit, and prepares `/opt/jarvis-alx`.

## Development Lifecycle

- `./scripts/dev_up.sh` – build and start the full compose stack.
- `./scripts/dev_down.sh` – stop and remove containers/volumes.
- `./scripts/host_update.sh` – pull updates and rebuild.

## Configuration

- `config/jarvis_personality.yaml` – defines system prompt traits.
- `config/hw_pins.yaml` – describes GPIO lines, sensors, and UPS metadata.
- `config/services.yaml` – placeholder routing map for future expansion.
- `SENTENCE_TRANSFORMER_MODEL` (env) – optional HF encoder (e.g., intfloat/e5-small-v2) for better memory embeddings.
- `LLM_MODEL_PATH` (env) – path for vLLM `--model` (default `/models/llama-3-70b`); adjusted by auto-select.
- `MODEL_STORE_ROOT` (env) – host directory where models live (defaults: /run/media/$USER/Samsung_T7/models → ./models → /srv/models).

## Desktop launcher (optional)

`launchers/jarvis-alx.desktop` can be copied to `~/.local/share/applications` after replacing `REPO_ROOT_PLACEHOLDER` with the repo path. It invokes `scripts/start_portable.sh` so everything comes up from a single click.

## Security Notes

- **Never** leave the default `JARVIS_INTERNAL_TOKEN` in production.
- The web UI uses `NEXT_PUBLIC_JARVIS_TOKEN` to communicate directly with the orchestrator/voice gateway. Treat access to the dashboard as trusted.
- RF and shutdown endpoints require the shared token in every request.

## Deploying with systemd

Copy `deploy/systemd/jarvis-docker.service` to `/etc/systemd/system/`, update the working directory if needed, then enable:

```bash
sudo systemctl enable --now jarvis-docker.service
```

## Directory Map

See `jarvis ai build` spec for the canonical structure; this repository mirrors it exactly.
