"""Core data models for socr."""

from socr.core.born_digital import (
    BornDigitalDetector,
    DocumentAssessment,
    PageAssessment,
)
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
from socr.core.state import DocumentState, PageState

__all__ = [
    "BornDigitalDetector",
    "DocumentAssessment",
    "DocumentHandle",
    "DocumentState",
    "DocumentStatus",
    "EngineResult",
    "EngineType",
    "FailureMode",
    "FigureInfo",
    "HPCConfig",
    "MetadataManager",
    "PageAssessment",
    "PageOutput",
    "PageState",
    "PageStatus",
    "PipelineConfig",
]
