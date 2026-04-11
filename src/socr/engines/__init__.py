"""OCR engine adapters.

Each engine wraps a sibling CLI tool (gemini-ocr, deepseek-ocr, etc.).
socr is a thin orchestrator — retry, concurrency, and prompts are handled
by the CLIs themselves.
"""

from socr.engines.base import BaseEngine, BaseHTTPEngine
from socr.engines.deepseek import DeepSeekEngine
from socr.engines.gemini import GeminiEngine
from socr.engines.glm import GLMEngine
from socr.engines.marker import MarkerEngine
from socr.engines.mistral import MistralEngine
from socr.engines.nougat import NougatEngine

__all__ = [
    "BaseEngine",
    "BaseHTTPEngine",
    "DeepSeekEngine",
    "GeminiEngine",
    "GLMEngine",
    "MarkerEngine",
    "MistralEngine",
    "NougatEngine",
]
