# AGENTS.md

Canonical agent instructions for this repo. `CLAUDE.md` imports this file, so edit here only.

## Project context

UniBot is a campus-orientation chatbot for new Heriot-Watt students (BSc dissertation; supervisor Marta Romeo). It's a FastAPI backend in `agentic_rag/` with RAG over scraped university pages (ChromaDB), Azure OpenAI generation and OpenRouteService walking routes, plus a single-file React frontend in `index.html`. It is deployed at https://uni-bot.duckdns.org/.

- **GitHub is behind production.** The live server runs source that was never pushed (`public_access.py`, `deploy/`, Azure v1 Responses support, extra tests). The deployed baseline is in the private transfer package (`HANDOFF.md` plus `source/`), kept outside this repo. Ask the user where it is. Before changing anything that production also touches, diff against that baseline. Commit only reviewed source, never credentials.
- **Secrets never enter git.** That covers `.env` files, `PRIVATE/`, SSH keys and API keys. `.env` is already gitignored. Don't paste secret values into chats, issues or PRs.
- **Production is a shared host.** Another site runs alongside UniBot. Read `HANDOFF.md` before any server or deployment work, and treat server changes as needing explicit user approval.
- Tests: `python -m pytest agentic_rag/tests -q -p no:cacheprovider`.

## Agent skills

### Issue tracker

Issues live in GitHub Issues for `nm2064/Supporting-new-students-orienting-themselves-on-campus` (via `gh`). See `docs/agents/issue-tracker.md`.

### Triage labels

Default five-role vocabulary: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: `GLOSSARY.md` + `docs/adr/` at the repo root. See `docs/agents/domain.md`.
