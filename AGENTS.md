# Agent Instructions

Issue tracking uses **bd** (beads). See "Session completion" below for the mandatory end-of-session workflow. Run `bd onboard` if bd has not been set up.

## Commands

```bash
pip install -e ".[dev]"        # install with pytest/black/isort/flake8
python3 -m pytest tests/       # run tests (import-only, no network needed)
python3 -m pytest tests/test_import.py::test_import_factory   # single test
python3 test_basic.py          # legacy structure check — gitignored, may not exist on fresh clones
```

- Provider SDKs are optional extras: `pip install -e ".[openai]"`, `.[gemini]`, `.[claude]`, or `.[all]`. Core deps are in `requirements.txt`.
- Examples in `examples/` make **live AI calls** — they need a local Ollama/LM Studio server or real API keys.
- Formatting is black (line-length 100) + isort (profile=black); there is no lint config or CI.

## Architecture

- Library entrypoint: `create_ai_client()` in `modulle/factory.py` → returns `(client, text_processor, vision_processor)`.
- Providers live in `modulle/providers/<name>/` as `client.py`, `text_processor.py`, `vision_processor.py`. Adding a provider requires: the new package dir, a branch in `factory.py` (config defaults + API-key validation), and **adding every new package to the explicit `packages` list in `[tool.setuptools]` (pyproject.toml)** — there is no auto-discovery, forgotten packages silently won't install.
- Core API is deliberately generic: only `generate()`, `chat()`, `analyze_image()` on the base classes. Do NOT add domain-specific methods to the core; build features via prompts or examples (README "Contributing").
- Tool calling is per-provider: `client.chat_with_tools(...)` exists on each provider's client but **not** on `BaseAIClient`. Tool schemas come from `ToolRegistry.to_ollama_format() / to_openai_format() / to_claude_format() / to_gemini_format()` — there is no `to_lm_studio_format()`; LM Studio consumes tools in **OpenAI format**.
- Decision models (clef / clef-flash) are also per-provider: `OllamaClient.system_one()` hits Ollama's `/v1/systemone` endpoint (requires Ollama >= 0.35.1). This is a separate endpoint from `generate()`/`chat()` — decision models do not work through those.
- **Cloud model names differ from local**: on `ollama_cloud` (https://ollama.com, Bearer auth via `api_key`/`OLLAMA_API_KEY`) use the bare `/api/tags` name (e.g. `glm-5.3-flash`, **not** `glm-5.3-flash:cloud`). Decision models and `/v1/systemone` are local-only; no `Ollama-Cloud /v1/systemone` endpoint exists. Before any live test against Ollama Cloud, ask the user which model to use — list models via `GET https://ollama.com/api/tags`.
- `modulle/web/` provides direct methods (search/fetch) via `WebAccessor`, plus prebuilt `SearchWebTool`/`FetchPageTool` in `modulle/web/tools.py` for a ToolRegistry.
- Version appears in both `pyproject.toml` and `modulle/__init__.py __version__` — keep in sync.
- `DailyFeedSanity/`, `test_basic.py`, and old docs (`OVERVIEW.md`, `EXTRACTION_SUMMARY.md`, etc.) are **gitignored** — don't build on them; tracked docs are `README.md`, `REFACTORING.md`, `WEB_AND_TOOLS_README.md`, `docs/`, `examples/README.md`.

## Configuration

Priority: `~/.modulle.json` (highest; see `.modulle.json.example`) > environment variables > defaults in `modulle/config.py`. Env vars are read at import time. `.modulle.json` is gitignored; never commit real API keys.

## Session completion

Use bd for task tracking:

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --status in_progress  # Claim work
bd close <id>         # Complete work
bd sync               # Sync with git
```

**When ending a work session**, complete ALL steps below. Work is NOT complete until `git push` succeeds.

1. File issues for remaining work
2. Run quality gates (if code changed) — tests, linters, builds
3. Update issue status — close finished work, update in-progress items
4. Push to remote (mandatory):
   ```bash
   git pull --rebase
   bd sync
   git push
   git status  # MUST show "up to date with origin"
   ```
5. Clean up — clear stashes, prune remote branches
6. Verify all changes committed AND pushed
7. Hand off context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing — that leaves work stranded locally
- NEVER say "ready to push when you are" — YOU must push
- If push fails, resolve and retry until it succeeds