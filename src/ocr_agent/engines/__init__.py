"""OCR engine adapters."""

from ocr_agent.engines.base import BaseEngine, EngineCapabilities
from ocr_agent.engines.deepseek import DeepSeekEngine
from ocr_agent.engines.gemini import GeminiEngine
from ocr_agent.engines.mistral import MistralEngine
from ocr_agent.engines.nougat import NougatEngine

__all__ = [
    "BaseEngine",
    "EngineCapabilities",
    "NougatEngine",
    "DeepSeekEngine",
    "MistralEngine",
    "GeminiEngine",
]
