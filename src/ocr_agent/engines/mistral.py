"""Mistral OCR engine adapter.

Mistral OCR is a cloud-based service with excellent accuracy.
Cost: ~$0.001 per page.
"""

import base64
import io
import time
from pathlib import Path

from PIL import Image

from ocr_agent.core.config import MistralConfig
from ocr_agent.core.result import FigureResult, PageResult
from ocr_agent.engines.base import BaseEngine, EngineCapabilities


class MistralEngine(BaseEngine):
    """Adapter for Mistral OCR API."""

    COST_PER_PAGE = 0.001  # $0.001 per page

    OCR_PROMPT = """Extract all text from this document image.
Preserve the original formatting and structure:
- Maintain paragraph breaks
- Keep heading hierarchy
- Format tables as markdown
- Preserve lists (bulleted and numbered)
- Include footnotes and references

Output clean, well-formatted markdown."""

    FIGURE_PROMPT = """Describe this figure in detail:
1. What type of visualization is this? (chart, table, diagram, etc.)
2. What does it represent?
3. What are the key data points or findings?
4. What labels, legends, or annotations are present?

Be specific and quantitative where possible."""

    def __init__(self, config: MistralConfig | None = None) -> None:
        super().__init__()
        self.config = config or MistralConfig()
        self._client = None

    @property
    def name(self) -> str:
        return "mistral"

    @property
    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            name="mistral",
            supports_pdf=True,
            supports_images=True,
            supports_batch=False,
            supports_figures=True,
            is_local=False,
            cost_per_page=self.COST_PER_PAGE,
            best_for=["general", "high-accuracy", "multilingual"],
        )

    def initialize(self) -> bool:
        """Initialize Mistral client."""
        if self._initialized:
            return True

        if not self.config.api_key:
            return False

        try:
            from mistralai import Mistral

            self._client = Mistral(api_key=self.config.api_key)
            self._initialized = True
            return True

        except ImportError:
            return False
        except Exception:
            return False

    def _image_to_base64_url(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 data URL."""
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=95)
        b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"

    def _call_mistral(self, image: Image.Image, prompt: str) -> str:
        """Call Mistral API with image."""
        if not self._client:
            raise RuntimeError("Client not initialized")

        response = self._client.chat.complete(
            model=self.config.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": self._image_to_base64_url(image)},
                        },
                    ],
                }
            ],
        )

        return response.choices[0].message.content if response.choices else ""

    def process_image(self, image: Image.Image, page_num: int = 1) -> PageResult:
        """Process a single image with Mistral."""
        if not self._initialized and not self.initialize():
            return self._create_error_result(page_num, "Mistral not initialized (check API key)")

        start_time = time.time()

        try:
            text = self._call_mistral(image, self.OCR_PROMPT)
            processing_time = time.time() - start_time

            return self._create_success_result(
                page_num=page_num,
                text=text,
                processing_time=processing_time,
                cost=self.COST_PER_PAGE,
            )

        except Exception as e:
            return self._create_error_result(page_num, str(e))

    def describe_figure(
        self,
        image: Image.Image,
        figure_type: str = "unknown",
        context: str = "",
    ) -> FigureResult:
        """Describe a figure using Mistral vision."""
        if not self._initialized and not self.initialize():
            return FigureResult(
                figure_num=0,
                page_num=0,
                figure_type=figure_type,
                description="Mistral not initialized",
            )

        try:
            prompt = self.FIGURE_PROMPT
            if context:
                prompt += f"\n\nContext from surrounding text: {context}"

            description = self._call_mistral(image, prompt)

            # Try to extract figure type from response
            detected_type = figure_type
            for t in ["chart", "graph", "table", "diagram", "photo", "image"]:
                if t in description.lower():
                    detected_type = t
                    break

            return FigureResult(
                figure_num=0,
                page_num=0,
                figure_type=detected_type,
                description=description,
                engine=self.name,
            )

        except Exception as e:
            return FigureResult(
                figure_num=0,
                page_num=0,
                figure_type=figure_type,
                description=f"Error: {e}",
            )
