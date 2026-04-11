# Robustness Audit (2026-04-11)

Inspired by Noah Dasanaike's socOCRbench and "What I've Learned From Digitizing 20 Million Historical Documents."

## What changed

### Robustness (stop the hangs)

1. **Retry with exponential backoff** (`core/retry.py`) — shared utility used by Gemini API engine. Retries on 429/500/502/503/504 and transient network errors. Respects `Retry-After` headers. Configurable via `RetryConfig`.

2. **Figure extraction timeouts** — per-page SIGALRM guard (30s default) prevents malformed PDFs from hanging indefinitely. Bare `except: pass` replaced with logged warnings.

3. **Smart engine auto-selection** (`EngineType.AUTO`) — probes engines in priority order (gemini-api > gemini > mistral > deepseek > glm > nougat > marker). When no GPU/Ollama available, automatically falls through to cloud engines instead of failing.

4. **Concurrent page processing** — `ThreadPoolExecutor` with configurable `max_concurrent_pages` (default 4) for per-page API engines. Pre-renders all pages, then processes in parallel.

5. **Batch command unified pipeline** — `socr batch` now supports `--unified`, `--multi-engine`, and auto engine selection.

### Quality (inspired by Noah)

6. **Task-specific prompts** — Gemini API engine now accepts `prompt_hint` ("table", "complex", or default). Native-first pipeline passes "complex" for pages with tables/figures/equations. Based on socOCRbench finding that different prompts significantly affect quality.

7. **Markdown fence stripping** — normalizer now strips `\`\`\`markdown....\`\`\`` wrappers that VLMs add. Based on a bug Noah found where "entire response stripped instead of just the fence markers."

8. **Repetition loop repair** — normalizer detects lines repeated 5+ times consecutively and deduplicates. This is Noah's "partially degenerate" pattern: salvageable output that shouldn't be thrown away entirely.

9. **DeepSeek prompt change** — switched from `--task ocr` to `--task format` for structured markdown output. The `ocr` mode triggers formatting instruction hallucinations.

10. **DeepSeek/GLM availability checks** — `is_available()` now verifies the Ollama model is actually pulled, not just that the CLI exists. Prevents slow failures.

### Config

11. **Configurable render DPI** — `render_dpi` in PipelineConfig (default 200). Noah: "image resolution matters a lot."

12. **Fixed timeout default mismatch** — `build_config()` default was 300 but Click's was 1800.

13. **Fixed dependency conflict** — removed `marker` from `all` extras (pillow <11 vs >=11 conflict with deepseek).

## Decisions

- **AUTO as default** — changed `primary_engine` default from `deepseek` to `auto`. This means socr works out-of-the-box on machines without Ollama/GPU.
- **Retry, not circuit breaker** — simple retry with backoff is sufficient for the scale socr operates at. Circuit breakers add complexity without benefit for single-document or small-batch use.
- **ThreadPoolExecutor, not asyncio** — simpler, and httpx sync client is already used everywhere. No need to rewrite the engine interface.
