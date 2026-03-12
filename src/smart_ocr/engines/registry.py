"""Engine registry — maps EngineType to engine instances."""

from smart_ocr.core.config import EngineType
from smart_ocr.engines.base import BaseEngine
from smart_ocr.engines.deepseek import DeepSeekEngine
from smart_ocr.engines.gemini import GeminiEngine
from smart_ocr.engines.marker import MarkerEngine
from smart_ocr.engines.mistral import MistralEngine
from smart_ocr.engines.nougat import NougatEngine

_ENGINES: dict[EngineType, type[BaseEngine]] = {
    EngineType.NOUGAT: NougatEngine,
    EngineType.DEEPSEEK: DeepSeekEngine,
    EngineType.MISTRAL: MistralEngine,
    EngineType.GEMINI: GeminiEngine,
    EngineType.MARKER: MarkerEngine,
}


def get_engine(engine_type: EngineType) -> BaseEngine:
    """Get an engine instance by type."""
    cls = _ENGINES.get(engine_type)
    if cls is None:
        raise ValueError(f"No CLI engine for {engine_type.value}")
    return cls()
