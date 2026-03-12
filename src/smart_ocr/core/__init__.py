"""Core data models for smart-ocr."""

from smart_ocr.core.config import EngineType, HPCConfig, PipelineConfig
from smart_ocr.core.document import DocumentHandle
from smart_ocr.core.metadata import MetadataManager
from smart_ocr.core.result import DocumentResult, DocumentStatus, PageResult, PageStatus

__all__ = [
    "DocumentHandle",
    "DocumentResult",
    "MetadataManager",
    "DocumentStatus",
    "EngineType",
    "HPCConfig",
    "PageResult",
    "PageStatus",
    "PipelineConfig",
]
