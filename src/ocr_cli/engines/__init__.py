"""OCR engine adapters."""

from ocr_cli.engines.base import BaseEngine, EngineCapabilities
from ocr_cli.engines.deepseek import DeepSeekEngine
from ocr_cli.engines.gemini import GeminiEngine
from ocr_cli.engines.mistral import MistralEngine
from ocr_cli.engines.nougat import NougatEngine

__all__ = [
    "BaseEngine",
    "EngineCapabilities",
    "NougatEngine",
    "DeepSeekEngine",
    "MistralEngine",
    "GeminiEngine",
]
