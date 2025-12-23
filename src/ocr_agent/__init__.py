"""OCR Agent - Multi-agent document processing with beautiful terminal UI."""

__version__ = "0.1.0"

from ocr_agent.core.config import AgentConfig
from ocr_agent.core.document import Document, DocumentType
from ocr_agent.core.result import OCRResult, PageResult
from ocr_agent.pipeline.processor import OCRPipeline

__all__ = [
    "AgentConfig",
    "Document",
    "DocumentType",
    "OCRResult",
    "PageResult",
    "OCRPipeline",
]
