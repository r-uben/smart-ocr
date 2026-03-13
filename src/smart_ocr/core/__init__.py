"""Core data models for smart-ocr."""

from smart_ocr.core.config import EngineType, HPCConfig, PipelineConfig
from smart_ocr.core.document import DocumentHandle
from smart_ocr.core.metadata import MetadataManager
from smart_ocr.core.result import (
    DocumentResult,
    DocumentStatus,
    FigureInfo,
    PageResult,
    PageStatus,
)

__all__ = [
    "DocumentHandle",
    "DocumentResult",
    "DocumentStatus",
    "EngineType",
    "FigureInfo",
    "HPCConfig",
    "MetadataManager",
    "PageResult",
    "PageStatus",
    "PipelineConfig",
]
