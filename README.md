<div align="center">

# S.K.Y.E.
### Sophisticated Knowledge Yielding Entity

*A private, locally-run AI assistant built natively on Apple Silicon*

![Architecture](assets/architecture.png)

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python)](https://python.org)
[![MLX](https://img.shields.io/badge/MLX-Apple%20Silicon-000000?style=flat-square&logo=apple)](https://github.com/ml-explore/mlx)
[![Gemma-4](https://img.shields.io/badge/Model-Gemma--4%20E4B-purple?style=flat-square)](https://huggingface.co/mlx-community/gemma-4-e4b-it-4bit)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

</div>

---

S.K.Y.E. is not a chatbot wrapper. It is a locally-running intelligence engine built on **Google Gemma 4 E4B** that remembers everything you discuss, calls real tools, and holds a consistent personality — all without sending a single byte of data to the cloud.

---

## ✨ Core Capabilities

| Capability | Status |
|---|---|
| 🔍 Web Search (Tavily, feeds long-term memory) | ✅ Live |
| 🌤️ Live Weather | ✅ Live |
| 📰 News Summarisation | ✅ Live |
| ⏰ Alarms & Reminders | ✅ Live |
| 🎭 Consistent Persona (system prompt) | ✅ Live |
| 💾 3-Layer Hybrid Memory (RAG) | ✅ Live |
| 🌙 Nightly Memory Consolidation | ✅ Live |
| 🛡️ Runtime Guardrails | ✅ Live |

---

## 🏛️ Architecture

S.K.Y.E. runs the **stock 4-bit Llama-3 8B Instruct** model. No fine-tuned weights, no adapters. Everything that makes her *her* lives in three places.

### 1 — Persona (System Prompt)

Personality, tone and the tool-calling contract are defined in a single file: [`prompts/skye_persona.txt`](prompts/skye_persona.txt). It is loaded at startup and prepended to every conversation as the system message.

This replaced an earlier approach that baked personality into fine-tuned LoRA weights. The prompt is deliberately terse — every token in it is paid for on every single turn, so it has been tuned for size as well as behaviour.

### 2 — Tools

When the persona decides a request needs real data, the model emits a single line:

```
CALL_FUNC: {"name": "tell_time", "arguments": {}}
```

`core_v2.py` parses that, dispatches to the registered Python function, and feeds the result back into the next turn as a system note. A blocklist rejects anything destructive before dispatch.

### 3 — Hybrid RAG Memory (3-Layer)

A persistent, hierarchical memory that survives across sessions:

| Tier | Storage | Role |
|---|---|---|
| **1 — Short Term** | Active `SHARED_MESSAGES` list | System message + last 6 turns |
| **2 — Long Term Semantics** | SQLite + `sentence-transformers` vector store | Semantic search over all historical conversations |
| **3 — Persistent Profile** | `memory/persistent_profile.json` | Structured facts: your name, preferences, habits |

On every message, S.K.Y.E. runs a **hybrid search** (70% vector cosine similarity + 30% keyword boost) over Tier 2 and injects the top-3 memories into the system prompt. Results below a relevance floor are discarded rather than padding the prompt with noise, and a `UNIQUE` index on content keeps the store free of duplicates.

---

## 🌙 Nightly Memory Consolidation

Memory consolidation runs via a single command (or a scheduled Cron job):

```bash
python scripts/nightly_agi_cron.py
```

Two phases run sequentially:

```
Phase 1 │ Fact Extraction      → proposed_facts.py   → updates memory/persistent_profile.json
Phase 2 │ Semantic Ingestion   → ingest_history.py   → indexes logs into memory/semantics.db
```

> **Note:** this pipeline previously had three further phases that generated DPO pairs from guardrail failures, fine-tuned LoRA adapters on-device, and hot-swapped the weights into the running server. That loop treated the assistant's own failures as correct training targets and progressively degraded the model. It has been removed. S.K.Y.E. learns *facts* overnight, not weights.

---

## 📁 Project Structure

```
jarvis/
│
├── core/
│   ├── core_v2.py              # Main production engine (Socket Server + CLI)
│   ├── protocol.py             # Newline-delimited JSON frames (client <-> server)
│   ├── stt.py                  # Whisper speech-to-text (MLX)
│   ├── tts_client.py           # Talks to tts_server/ (Chatterbox Turbo voice)
│   ├── mood.py                 # Picks a speaking mood per reply
│   └── skills.py               # Routes a request to a skill (its tools + UI colour)
│
├── mcp_server/
│   ├── server.py               # Tool server (MCP): time, weather, alarms, web search, tasks...
│   ├── notion_tools.py         # Notion: finance tracker, to-do list, learning notes
│   └── calendar_tools.py       # Google Calendar: list, add, move, cancel, free time, undo
│
├── tts_server/
│   └── server.py               # Chatterbox Turbo TTS subprocess (runs in .venv)
│
├── prompts/
│   └── skye_persona.txt        # The personality. Loaded as the system message.
│
├── scripts/
│   ├── nightly_agi_cron.py     # Nightly memory consolidation orchestrator
│   ├── ingest_history.py       # Semantic log ingestor (Tier 2)
│   └── clean_memory.py         # One-time semantics.db dedupe/cleanup
│
├── memory/
│   ├── manager.py              # MemoryManager: hybrid vector + keyword search
│   ├── semantics.db            # SQLite vector store (Tier 2 - generated)
│   ├── persistent_profile.json # Structured user facts (Tier 3 - generated)
│   └── short-term/
│       └── proposed_facts.py   # LLM-powered fact extractor
│
├── clients/
│   ├── browser.py              # HTTP + WebSocket bridge to the socket server
│   └── browser.html            # Browser UI
│
└── logs/                       # Session telemetry JSONL files (gitignored)
```

---

## 🚀 Getting Started

### Prerequisites
- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.11 via `pyenv`
- ~10GB free unified memory

### 1. Clone & Setup Environment
```bash
git clone <your-repo-url>
cd jarvis
pyenv shell jarvis-py311
```

### 2. Install Dependencies
```bash
pip install mlx mlx-lm sentence-transformers numpy scikit-learn tavily-python google-generativeai python-dotenv google-auth-oauthlib google-api-python-client dateparser
```

**Voice (TTS) runs in its own environment.** Chatterbox Turbo (via `mlx-audio`) needs a newer `mlx` than the LLM's pinned runtime, so `tts_server/` is spawned as a subprocess using the repo's `.venv`:
```bash
python3 -m venv .venv && .venv/bin/pip install mlx-audio soundfile
```
Override the interpreter with `SKYE_TTS_PYTHON`. The voice is cloned from `assets/reference_voice_short.wav` (a ~13 s slice of `reference_voice.wav`; regenerate it if you change the reference). Optional per-mood clips — `assets/reference_voice_happy.wav`, `_sad.wav`, `_concerned.wav` — are picked up automatically. The first start downloads the model.

**Notion (optional).** Create an internal integration at notion.so/profile/integrations, share your pages with it (`...` → Connections), and put its secret in `.env` as `NOTION_TOKEN`. SKYE finds the finance tracker, to-do list and learning notes by name/shape, so nothing else needs configuring. `python scripts/notion_discovery.py` prints everything the integration can see. Tools are added to the model's menu per request by `core/skills.py` (finance / to-do / knowledge), and each skill recolours the UI.

**Google Calendar (optional).** In Google Cloud, enable the Calendar API, create an OAuth *Desktop app* client (add yourself as a test user) and save its JSON as `gcp-oauth.keys.json` in the repo root. Then run `python scripts/google_login.py` once and approve access; the token is kept in `memory/google_token.json` (both files are gitignored).

### 3. Configure Environment
Create a `.env` file in the project root:
```env
GOOGLE_API_KEY=your_key_here   # For web search fallback (Gemini)
WEATHER_API_KEY=your_key_here  # For live weather
```

### 4. Run S.K.Y.E.
```bash
python core/core_v2.py
```
Select **[1]** for an interactive terminal session or **[2]** to expose a Socket Server on port `12345`.

### 5. Consolidate Memory
After a conversation session, run the overnight pipeline so S.K.Y.E. remembers it:
```bash
python scripts/nightly_agi_cron.py
```

> **Tip**: To make this automatic, add this to your Mac's Crontab (runs at 3 AM nightly):
> ```
> 0 3 * * * cd /path/to/jarvis && /path/to/envs/jarvis-py311/bin/python scripts/nightly_agi_cron.py
> ```

---

## ⚙️ Technical Stack

| Component | Technology |
|---|---|
| **Base LLM** | `mlx-community/gemma-4-e4b-it-4bit` (stock, no adapters) |
| **Inference** | `mlx-lm` (Apple Silicon native) |
| **Personality** | System prompt — `prompts/skye_persona.txt` |
| **Embedding Model** | `all-MiniLM-L6-v2` (sentence-transformers) |
| **Vector Store** | SQLite + NumPy cosine similarity |
| **Concurrency** | `threading.Lock()` serialises generation across socket clients |
| **Telemetry** | JSONL session logs with guardrail annotations |

---

## 🛡️ Guardrail System

A layered runtime defence against common LLM failure modes. Every guardrail that fires is printed as `[guardrails fired: ...]` and recorded in the session telemetry.

| Guardrail | Trigger | Action |
|---|---|---|
| `role_integrity_reset` | Role-play leakage detected | Hard reset to SKYE persona |
| `repetition_abort` | Repeated sentences / identical prior turn | Prompt to rephrase |
| `verbosity_truncate` | Response > 200 words or > 8 sentences | Truncate to 8 sentences |
| `personality_dampen` | Over-saturation of butler tokens (`Sir`, `Certainly`) | Suppress excess tokens |
| `low_info_suppression` | Filler phrases detected multiple times | Deduplicate |

Ahead of these, `sanitise_raw()` strips malformed tool calls, dangling JSON and status filler from the raw output.

Several of these were written to compensate for the removed fine-tune and may now be unreachable. They are being kept under observation and retired on logged evidence rather than on inspection.

---

<div align="center">
  <sub>Built with 🖤 on Apple Silicon</sub>
</div>
