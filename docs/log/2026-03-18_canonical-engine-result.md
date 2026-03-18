# L0C-01: Canonical EngineResult contract

**Date:** 2026-03-18
**Branch:** `feat/canonical-engine-result`

## What

Replaced `DocumentResult`/`PageResult` with `EngineResult`/`PageOutput` as the canonical types all engines return. Added structured metadata fields: `failure_mode`, `model_version`, `cost`.

## Key decisions

- **`EngineResult.pages: list[PageOutput]` is the source of truth** for content. `markdown` is a computed property that assembles from pages.
- **CLI engines produce a single `PageOutput(page_num=0)`** with the full document text. This is a transitional state — L0C-02 (born-digital detection) and L0C-03 (normalize outputs) will add proper per-page splitting.
- **`FailureMode` enum** with 11 variants (`timeout`, `cli_error`, `empty_output`, `hallucination`, `refusal`, etc.) — enables programmatic repair routing in L1B-01.
- **`pages_processed` kept as stored field** (not derived from `len(pages)`) because CLI engines don't know page count — the pipeline sets it from `doc.page_count`.
- **No backward-compat aliases.** Clean rename throughout; old `smart_ocr/` package untouched (legacy).

## Files changed

| File | Change |
|------|--------|
| `core/result.py` | New `FailureMode`, renamed `DocumentResult→EngineResult`, `PageResult→PageOutput` |
| `core/__init__.py`, `__init__.py` | Updated exports |
| `engines/base.py` | `BaseEngine.process_document()` returns `EngineResult` with `PageOutput`; `BaseHTTPEngine.process_image()` returns `PageOutput`; added `model_version` property |
| `engines/deepseek.py`, `engines/glm.py` | Updated inline imports |
| `engines/deepseek_vllm.py`, `engines/vllm.py` | Updated types |
| `pipeline/processor.py` | `StandardPipeline` uses `EngineResult` |
| `pipeline/hpc_pipeline.py` | `HPCPipeline` builds `EngineResult` from `PageOutput` list |
| `pipeline/reconciler.py` | Renamed `create_page_result_from_reconciliation` → `create_page_output_from_reconciliation` |
| `tests/test_engine_result.py` | New: 8 tests for the contract |
| `tests/test_audit_heuristics.py` | Updated `PageResult` → `PageOutput` |

## What this unblocks

- **L0C-03** (Normalize engine output formats) — can now populate per-page PageOutputs
- **L0C-04** (DocumentState blackboard) — can build state from structured EngineResult
- **L1B-01** (Failure-mode detection in scorer) — can read `FailureMode` enum to route repairs
