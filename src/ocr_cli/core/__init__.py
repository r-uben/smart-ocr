"""Core data models for OCR CLI."""

from ocr_cli.core.config import AgentConfig
from ocr_cli.core.document import Document, PageImage
from ocr_cli.core.result import OCRResult, PageResult, ProcessingStats

__all__ = [
    "AgentConfig",
    "Document",
    "PageImage",
    "OCRResult",
    "PageResult",
    "ProcessingStats",
]
