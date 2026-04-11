"""Engine registry — maps EngineType to engine instances."""

import logging

from socr.core.config import AUTO_ENGINE_ORDER, EngineType
from socr.engines.base import BaseEngine
from socr.engines.deepseek import DeepSeekEngine
from socr.engines.gemini import GeminiEngine
from socr.engines.glm import GLMEngine
from socr.engines.marker import MarkerEngine
from socr.engines.mistral import MistralEngine
from socr.engines.nougat import NougatEngine

logger = logging.getLogger(__name__)

_ENGINES: dict[EngineType, type[BaseEngine]] = {
    EngineType.GLM: GLMEngine,
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


def resolve_auto_engine() -> EngineType:
    """Probe CLI engines in priority order and return the first available one.

    Returns the first engine whose CLI is installed and whose dependencies
    (API keys, Ollama models, etc.) are satisfied. Falls back to GEMINI
    if nothing is available, letting downstream code produce a clear error.
    """
    for engine_type in AUTO_ENGINE_ORDER:
        try:
            cli_engine = _ENGINES.get(engine_type)
            if cli_engine is None:
                continue
            instance = cli_engine()
            if instance.is_available():
                logger.info(f"Auto-selected engine: {engine_type.value}")
                return engine_type
        except Exception:
            continue

    logger.warning("No engines available; falling back to gemini (will likely fail)")
    return EngineType.GEMINI
