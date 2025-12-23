"""Core data models for OCR Agent."""

from ocr_agent.core.config import AgentConfig
from ocr_agent.core.document import Document, PageImage
from ocr_agent.core.result import OCRResult, PageResult, ProcessingStats

__all__ = [
    "AgentConfig",
    "Document",
    "PageImage",
    "OCRResult",
    "PageResult",
    "ProcessingStats",
]
