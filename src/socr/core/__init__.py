"""Core data models for socr."""

from socr.core.config import EngineType, HPCConfig, PipelineConfig
from socr.core.document import DocumentHandle
from socr.core.metadata import MetadataManager
from socr.core.result import (
    DocumentStatus,
    EngineResult,
    FailureMode,
    FigureInfo,
    PageOutput,
    PageStatus,
)

__all__ = [
    "DocumentHandle",
    "DocumentStatus",
    "EngineResult",
    "EngineType",
    "FailureMode",
    "FigureInfo",
    "HPCConfig",
    "MetadataManager",
    "PageOutput",
    "PageStatus",
    "PipelineConfig",
]
