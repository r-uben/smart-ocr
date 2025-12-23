"""Rich terminal UI components for OCR CLI."""

from ocr_cli.ui.console import AgentConsole
from ocr_cli.ui.panels import StagePanel, SummaryPanel
from ocr_cli.ui.progress import AgentProgress

__all__ = ["AgentConsole", "StagePanel", "SummaryPanel", "AgentProgress"]
