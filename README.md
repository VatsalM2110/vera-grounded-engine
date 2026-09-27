# Vera Grounded AI Message Engine

Vera uses Groq to generate natural messages grounded in the context supplied by the challenge. Set
`GROQ_API_KEY` in the deployment environment to enable AI generation. If Groq is unavailable, the
service automatically falls back to its deterministic templates so the API remains reliable.

A deterministic, stateful submission for the magicpin Vera AI Challenge. It turns category, merchant, trigger, and optional customer context into a single grounded WhatsApp action—without an external LLM, latency, or fabricated facts.

## Approach

- Prioritizes active triggers by urgency, allows at most one send per merchant per tick, and enforces suppression keys.
- Uses category-aware strategies for research, compliance, performance, lifecycle, seasonal, competitive, planning, and customer triggers.
- Grounds numbers and claims only in received context. New context versions atomically replace old ones, so adaptive judge injections are immediately used.
- Stores context, conversations, turns, and suppressions in SQLite (WAL mode) for restart-safe behavior.
- Handles explicit intent, auto-replies, opt-outs, delays, hostile/off-topic replies, and unknown questions through a deterministic state machine.

## Run locally

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
uvicorn bot:app --host 0.0.0.0 --port 8000
```

Then run `python self_test.py`. For the official simulator, set its `BOT_URL` to `http://localhost:8000`, configure its judge LLM credentials, and run `python judge_simulator.py`.

## Deploy

The included `render.yaml` and `Procfile` support Render and other Procfile hosts. Set `TEAM_NAME`, `TEAM_MEMBER`, `CONTACT_EMAIL`, and `SUBMITTED_AT` in the host environment. For multi-instance production, point `VERA_DB_PATH` at persistent storage or keep one web instance during judging.

Tradeoff: deterministic composition is less stylistically flexible than an LLM, but is fast, reproducible, auditable, and robust under fresh context injection.
