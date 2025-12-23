"""OCR CLI - Multi-agent document processing with beautiful terminal UI."""

__version__ = "0.1.0"

from ocr_cli.core.config import AgentConfig
from ocr_cli.core.document import Document, DocumentType
from ocr_cli.core.result import OCRResult, PageResult
from ocr_cli.pipeline.processor import OCRPipeline

__all__ = [
    "AgentConfig",
    "Document",
    "DocumentType",
    "OCRResult",
    "PageResult",
    "OCRPipeline",
]
