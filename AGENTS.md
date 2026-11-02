# AGENTS.md

Canonical agent instructions for this repo. `CLAUDE.md` imports this file, so edit here only.

## Project context

UniBot is a campus-orientation chatbot for new Heriot-Watt students (BSc dissertation; supervisor Marta Romeo). It's a FastAPI backend in `agentic_rag/` with RAG over scraped university pages (ChromaDB) and OpenRouteService walking routes, plus a single-file React frontend in `index.html`. It is deployed at https://uni-bot.duckdns.org/ on an Oracle Free Tier Ubuntu host.

- **Models: NVIDIA hosted API** (`LLM_PROVIDER=nvidia`). Chat uses `nvidia/nemotron-3-ultra-550b-a55b` and embeddings use `nvidia/nemotron-3.5-embed-1b`. All provider quirks live in `agentic_rag/llm.py` (`LLMGateway`). The Azure key was revoked on 2 Nov 2026. Changing the embedding model means rebuilding the Chroma index into a new snapshot directory.
- **Answers must stay cited.** Every `[n]` in an answer maps to citation n (title and URL), and the UI shows a Sources list. `agentic_rag/tests/test_chat_citations.py` pins this behaviour, so keep it green.
- **Deploying.** Production code lives in `/opt/unibot/app`, config in `/etc/unibot.env` (root, 600) and state in `/var/lib/unibot`. The steps are: back up to `/var/backups/unibot`, stage, diff, run the tests on the server, restart, then verify the public pages and a cited answer. The private `HANDOFF.md` (transfer package, outside this repo) has the full procedure and SSH access. Ask the user where it is.
- **Secrets never enter git.** That covers `.env` files, `PRIVATE/`, SSH keys and API keys. `.env` is already gitignored. Don't paste secret values into chats, issues or PRs.
- **Production is a shared host.** Another site (Lotus Bun) runs alongside UniBot. Preserve its services and its Caddy block, and treat server changes as needing explicit user approval.
- Tests: `python -m pytest agentic_rag/tests -q -p no:cacheprovider`. They use fake LLM and translator adapters and a temporary Chroma, so they need no API keys.

## Agent skills

### Issue tracker

Issues live in GitHub Issues for `nm2064/Supporting-new-students-orienting-themselves-on-campus` (via `gh`). See `docs/agents/issue-tracker.md`.

### Triage labels

Default five-role vocabulary: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: `GLOSSARY.md` + `docs/adr/` at the repo root. See `docs/agents/domain.md`.
