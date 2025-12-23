"""Gemini OCR engine adapter.

Gemini Flash is a cloud-based multimodal model with excellent OCR capabilities.
Cost: ~$0.0002 per page (very cheap).
"""

import io
import time
from pathlib import Path

from PIL import Image

from ocr_cli.core.config import GeminiConfig
from ocr_cli.core.result import FigureResult, PageResult
from ocr_cli.engines.base import BaseEngine, EngineCapabilities


class GeminiEngine(BaseEngine):
    """Adapter for Google Gemini API."""

    COST_PER_PAGE = 0.0002  # ~$0.0002 per page (Flash is very cheap)

    OCR_PROMPT = """Extract all text from this document image with high accuracy.
Preserve the original formatting and structure:
- Maintain paragraph breaks and spacing
- Keep heading hierarchy (use markdown # headers)
- Format tables as proper markdown tables
- Preserve lists (bulleted and numbered)
- Include footnotes, references, and citations
- Maintain any special formatting (bold, italic)

Output clean, well-structured markdown. Be thorough and accurate."""

    FIGURE_PROMPT = """Analyze this figure in detail and provide:

1. **Type**: What kind of visualization is this? (bar chart, line graph, pie chart,
   scatter plot, table, diagram, flowchart, photograph, etc.)

2. **Description**: What does this figure show or represent?

3. **Key findings**: What are the main data points, trends, or conclusions?

4. **Details**: List any labels, axes, legends, annotations, or notable features.

Be specific and quantitative. Extract any numbers, percentages, or values visible."""

    def __init__(self, config: GeminiConfig | None = None) -> None:
        super().__init__()
        self.config = config or GeminiConfig()
        self._client = None

    @property
    def name(self) -> str:
        return "gemini"

    @property
    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            name="gemini",
            supports_pdf=True,
            supports_images=True,
            supports_batch=False,
            supports_figures=True,
            is_local=False,
            cost_per_page=self.COST_PER_PAGE,
            best_for=["general", "figures", "multilingual", "high-quality"],
        )

    def initialize(self) -> bool:
        """Initialize Gemini client."""
        if self._initialized:
            return True

        if not self.config.api_key:
            return False

        try:
            from google import genai

            self._client = genai.Client(api_key=self.config.api_key)
            self._initialized = True
            return True

        except ImportError:
            return False
        except Exception:
            return False

    def _pil_to_part(self, image: Image.Image):
        """Convert PIL Image to Gemini Part."""
        from google.genai import types

        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=95)
        return types.Part.from_bytes(data=buffer.getvalue(), mime_type="image/jpeg")

    def _call_gemini(self, image: Image.Image, prompt: str) -> str:
        """Call Gemini API with image."""
        if not self._client:
            raise RuntimeError("Client not initialized")

        from google.genai import types

        response = self._client.models.generate_content(
            model=self.config.model,
            contents=[prompt, self._pil_to_part(image)],
            config=types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=8192,
            ),
        )

        return response.text.strip() if response.text else ""

    def process_image(self, image: Image.Image, page_num: int = 1) -> PageResult:
        """Process a single image with Gemini."""
        if not self._initialized and not self.initialize():
            return self._create_error_result(page_num, "Gemini not initialized (check API key)")

        start_time = time.time()

        try:
            text = self._call_gemini(image, self.OCR_PROMPT)
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
        """Describe a figure using Gemini vision."""
        if not self._initialized and not self.initialize():
            return FigureResult(
                figure_num=0,
                page_num=0,
                figure_type=figure_type,
                description="Gemini not initialized",
            )

        try:
            prompt = self.FIGURE_PROMPT
            if context:
                prompt += f"\n\nContext from surrounding text:\n{context}"

            description = self._call_gemini(image, prompt)

            # Try to extract figure type from response
            detected_type = figure_type
            type_keywords = {
                "bar chart": "bar_chart",
                "line graph": "line_graph",
                "pie chart": "pie_chart",
                "scatter": "scatter_plot",
                "table": "table",
                "diagram": "diagram",
                "flowchart": "flowchart",
                "photograph": "photo",
                "map": "map",
            }
            for keyword, t in type_keywords.items():
                if keyword in description.lower():
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
