"""OCR engine adapters."""

from smart_ocr.engines.base import BaseEngine
from smart_ocr.engines.deepseek import DeepSeekEngine
from smart_ocr.engines.gemini import GeminiEngine
from smart_ocr.engines.marker import MarkerEngine
from smart_ocr.engines.mistral import MistralEngine
from smart_ocr.engines.nougat import NougatEngine

__all__ = [
    "BaseEngine",
    "DeepSeekEngine",
    "GeminiEngine",
    "MarkerEngine",
    "MistralEngine",
    "NougatEngine",
]
