"""Tests for the canonical EngineResult/PageOutput contract."""

from pathlib import Path

from socr.core.result import (
    DocumentStatus,
    EngineResult,
    FailureMode,
    PageOutput,
    PageStatus,
)


def test_single_page_markdown() -> None:
    """CLI engines produce a single PageOutput; markdown returns it directly."""
    result = EngineResult(
        document_path=Path("test.pdf"),
        engine="gemini",
        status=DocumentStatus.SUCCESS,
        pages=[PageOutput(page_num=0, text="Full document text", status=PageStatus.SUCCESS)],
    )
    assert result.markdown == "Full document text"
    assert result.word_count == 3
    assert result.success


def test_multi_page_markdown_assembly() -> None:
    """HPC engines produce per-page outputs; markdown joins with separators."""
    result = EngineResult(
        document_path=Path("test.pdf"),
        engine="hpc-sequential",
        status=DocumentStatus.SUCCESS,
        pages=[
            PageOutput(page_num=1, text="Page one content", status=PageStatus.SUCCESS),
            PageOutput(page_num=2, text="Page two content", status=PageStatus.SUCCESS),
        ],
    )
    assert result.markdown == "Page one content\n\n---\n\nPage two content"
    assert result.word_count == 7  # includes "---" separator


def test_empty_pages_produce_empty_markdown() -> None:
    result = EngineResult(
        document_path=Path("test.pdf"),
        engine="deepseek",
        status=DocumentStatus.ERROR,
        failure_mode=FailureMode.TIMEOUT,
    )
    assert result.markdown == ""
    assert result.word_count == 0
    assert not result.success


def test_failure_mode_on_result() -> None:
    result = EngineResult(
        document_path=Path("test.pdf"),
        engine="deepseek",
        status=DocumentStatus.ERROR,
        failure_mode=FailureMode.CLI_ERROR,
        error="CLI exited 1: segfault",
    )
    assert result.failure_mode == FailureMode.CLI_ERROR
    assert result.failure_mode == "cli_error"  # str enum comparison


def test_failure_mode_on_page() -> None:
    page = PageOutput(
        page_num=3,
        status=PageStatus.ERROR,
        failure_mode=FailureMode.HALLUCINATION,
        error="Repeated sentence loop detected",
    )
    assert page.failure_mode == FailureMode.HALLUCINATION
    assert page.needs_reprocessing()


def test_model_version_and_cost() -> None:
    result = EngineResult(
        document_path=Path("test.pdf"),
        engine="gemini",
        status=DocumentStatus.SUCCESS,
        pages=[PageOutput(page_num=0, text="ok", status=PageStatus.SUCCESS)],
        model_version="gemini-3-flash-preview",
        cost=0.003,
    )
    assert result.model_version == "gemini-3-flash-preview"
    assert result.cost == 0.003


def test_skips_empty_pages_in_markdown() -> None:
    """Pages with empty text are excluded from assembled markdown."""
    result = EngineResult(
        document_path=Path("test.pdf"),
        engine="hpc",
        status=DocumentStatus.SUCCESS,
        pages=[
            PageOutput(page_num=1, text="Good page", status=PageStatus.SUCCESS),
            PageOutput(page_num=2, text="", status=PageStatus.ERROR),
            PageOutput(page_num=3, text="Another good page", status=PageStatus.SUCCESS),
        ],
    )
    assert result.markdown == "Good page\n\n---\n\nAnother good page"


def test_page_output_defaults() -> None:
    page = PageOutput(page_num=1)
    assert page.status == PageStatus.PENDING
    assert page.failure_mode == FailureMode.NONE
    assert page.text == ""
    assert page.confidence == 0.0
    assert page.audit_passed is True
    assert not page.needs_reprocessing()  # PENDING doesn't trigger reprocessing
