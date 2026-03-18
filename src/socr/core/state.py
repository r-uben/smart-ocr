"""Central document state blackboard for the OCR pipeline.

DocumentState is the single mutable data structure that accumulates results
as a document flows through pipeline stages (born-digital detection, primary
engine, fallback engines, reconciliation, figure extraction).

Design principles:
  - DUMB DATA: no pipeline logic, no audit rules, no engine calls.
  - The orchestrator calls an engine, gets an EngineResult, then merges it
    via ``state.apply_result(result)``.
  - All attempts are stored per page for reconciliation / voting.
  - ``best_output`` is auto-selected naively (first passing attempt); the
    reconciler will override it later.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from socr.core.born_digital import DocumentAssessment
from socr.core.document import DocumentHandle
from socr.core.result import DocumentStatus, EngineResult, PageOutput


@dataclass
class PageState:
    """Per-page processing state (1-indexed)."""

    page_num: int
    is_born_digital: bool = False
    native_text: str | None = None
    attempts: list[PageOutput] = field(default_factory=list)  # all engine attempts
    best_output: PageOutput | None = None  # selected/reconciled best

    @property
    def needs_repair(self) -> bool:
        """Whether this page still needs (re)processing.

        Born-digital pages with native text never need repair.
        Otherwise, a page needs repair if there is no best_output yet or
        the best output failed its audit.
        """
        if self.is_born_digital and self.native_text:
            return False
        return not self.best_output or not self.best_output.audit_passed


@dataclass
class DocumentState:
    """Central blackboard for the OCR pipeline.

    Constructed from a ``DocumentHandle``; pre-populates one ``PageState``
    per page.  Engine results are merged via ``apply_result``.
    """

    handle: DocumentHandle
    status: DocumentStatus = DocumentStatus.PENDING
    pages: dict[int, PageState] = field(default_factory=dict)
    whole_doc_attempts: list[PageOutput] = field(default_factory=list)  # page_num=0 from CLI engines
    engine_runs: list[EngineResult] = field(default_factory=list)  # all EngineResult objects for telemetry

    def __post_init__(self) -> None:
        for i in range(1, self.handle.page_count + 1):
            if i not in self.pages:
                self.pages[i] = PageState(page_num=i)

    # ------------------------------------------------------------------
    # Mutation helpers
    # ------------------------------------------------------------------

    def apply_result(self, result: EngineResult) -> None:
        """Merge an engine's output into the blackboard."""
        self.engine_runs.append(result)
        for page_out in result.pages:
            if page_out.page_num == 0:
                self.whole_doc_attempts.append(page_out)
            else:
                page_state = self.pages.get(page_out.page_num)
                if page_state:
                    page_state.attempts.append(page_out)
                    if not page_state.best_output and page_out.audit_passed:
                        page_state.best_output = page_out

    def apply_born_digital(self, assessment: DocumentAssessment) -> None:
        """Apply born-digital detection results."""
        for pa in assessment.pages:
            if pa.page_num in self.pages:
                self.pages[pa.page_num].is_born_digital = pa.is_born_digital
                if pa.is_born_digital:
                    self.pages[pa.page_num].native_text = pa.native_text

    # ------------------------------------------------------------------
    # Read-only derived properties
    # ------------------------------------------------------------------

    @property
    def text(self) -> str:
        """Assemble current best document text.

        If only whole-doc attempts exist (CLI engines), return the last one.
        Otherwise, stitch per-page best outputs, preferring native text for
        born-digital pages.
        """
        # If only whole-doc attempts exist (CLI engines)
        if not any(p.best_output for p in self.pages.values()) and self.whole_doc_attempts:
            return self.whole_doc_attempts[-1].text

        texts: list[str] = []
        for i in range(1, self.handle.page_count + 1):
            p = self.pages[i]
            if p.is_born_digital and p.native_text:
                texts.append(p.native_text)
            elif p.best_output:
                texts.append(p.best_output.text)
        return "\n\n---\n\n".join(texts)

    @property
    def pages_needing_repair(self) -> list[int]:
        """Page numbers (sorted) that still need (re)processing."""
        return [i for i, p in sorted(self.pages.items()) if p.needs_repair]

    @property
    def total_cost(self) -> float:
        """Sum of cost across all engine runs."""
        return sum(r.cost for r in self.engine_runs)

    @property
    def engines_used(self) -> list[str]:
        """Ordered unique list of engine names used so far."""
        return list(dict.fromkeys(r.engine for r in self.engine_runs))
