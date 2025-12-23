"""DeepSeek OCR engine adapter.

DeepSeek VL2 runs locally via Ollama. Good for general documents
and supports multiple languages.
"""

import base64
import io
import re
import time
from pathlib import Path

import httpx
from PIL import Image

from ocr_agent.core.config import DeepSeekConfig
from ocr_agent.core.result import FigureResult, PageResult
from ocr_agent.engines.base import BaseEngine, EngineCapabilities


def _html_table_to_markdown(html_table: str) -> str:
    """Convert HTML table to markdown format."""
    rows = []
    row_matches = re.findall(r'<tr[^>]*>(.*?)</tr>', html_table, re.DOTALL | re.IGNORECASE)

    for row_html in row_matches:
        cells = re.findall(r'<t[dh][^>]*>(.*?)</t[dh]>', row_html, re.DOTALL | re.IGNORECASE)
        cleaned_cells = []
        for cell in cells:
            cell = re.sub(r'<[^>]+>', '', cell)
            cell = ' '.join(cell.split())
            cleaned_cells.append(cell)
        if cleaned_cells:
            rows.append(cleaned_cells)

    if not rows:
        return ""

    md_lines = []
    for idx, row in enumerate(rows):
        md_lines.append("| " + " | ".join(row) + " |")
        if idx == 0:
            md_lines.append("|" + "|".join(["---"] * len(row)) + "|")

    return "\n".join(md_lines)


def clean_ocr_output(text: str) -> str:
    """Remove grounding annotations, convert HTML tables to markdown, decode entities."""
    # Remove grounding annotations
    text = re.sub(r'<\|ref\|>.*?<\|/ref\|>', '', text)
    text = re.sub(r'<\|det\|>\[\[.*?\]\]<\|/det\|>', '', text)
    text = re.sub(r'<\|[^|]+\|>', '', text)

    # Convert HTML tables to markdown
    def replace_table(match: re.Match) -> str:
        return _html_table_to_markdown(match.group(0))

    text = re.sub(r'<table[^>]*>.*?</table>', replace_table, text, flags=re.DOTALL | re.IGNORECASE)

    # Handle other HTML elements
    text = re.sub(r'<(sup|sub)>([^<]*)</\1>', r'^\2', text, flags=re.IGNORECASE)
    text = re.sub(r'<center>([^<]*)</center>', r'\1', text, flags=re.IGNORECASE)
    text = re.sub(r'<br\s*/?>', '\n', text, flags=re.IGNORECASE)
    text = re.sub(r'<[^>]+>', '', text)

    # Decode HTML entities
    html_entities = {
        '&amp;': '&',
        '&lt;': '<',
        '&gt;': '>',
        '&quot;': '"',
        '&apos;': "'",
        '&nbsp;': ' ',
        '&#39;': "'",
        '&#x27;': "'",
    }
    for entity, char in html_entities.items():
        text = text.replace(entity, char)

    # Clean up whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)
    return text.strip()


class DeepSeekEngine(BaseEngine):
    """Adapter for DeepSeek OCR via Ollama.

    Uses the deepseek-ocr model which requires specific prompt formats.
    """

    # DeepSeek OCR uses special prompt tokens
    OCR_PROMPT = "<|grounding|>Convert the document to markdown."

    FIGURE_PROMPT = "Describe this figure in detail. What does the chart/graph/diagram show? Explain the axes, data, and key findings."

    def __init__(self, config: DeepSeekConfig | None = None) -> None:
        super().__init__()
        self.config = config or DeepSeekConfig()
        self._client: httpx.Client | None = None

    @property
    def name(self) -> str:
        return "deepseek"

    @property
    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            name="deepseek",
            supports_pdf=False,  # Process via images
            supports_images=True,
            supports_batch=False,
            supports_figures=True,  # Can describe figures
            is_local=True,
            cost_per_page=0.0,
            best_for=["general", "multilingual", "tables"],
        )

    def initialize(self) -> bool:
        """Initialize Ollama client and check model availability."""
        if self._initialized:
            return True

        try:
            self._client = httpx.Client(
                base_url=self.config.ollama_host,
                timeout=self.config.timeout,
            )

            # Check if model is available
            response = self._client.get("/api/tags")
            if response.status_code != 200:
                return False

            models = response.json().get("models", [])
            model_names = [m.get("name", "").split(":")[0] for m in models]

            if self.config.model.split(":")[0] not in model_names:
                # Try to pull the model
                return False

            self._initialized = True
            return True

        except Exception:
            return False

    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string."""
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=95)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _call_ollama(self, image: Image.Image, prompt: str, clean: bool = True) -> str:
        """Call Ollama API with image."""
        if not self._client:
            raise RuntimeError("Client not initialized")

        response = self._client.post(
            "/api/generate",
            json={
                "model": self.config.model,
                "prompt": prompt,
                "images": [self._image_to_base64(image)],
                "stream": False,
                "options": {
                    "num_ctx": 8192,
                    "temperature": 0.1,
                },
            },
        )

        if response.status_code != 200:
            raise RuntimeError(f"Ollama error: {response.status_code}")

        raw_text = response.json().get("response", "")
        return clean_ocr_output(raw_text) if clean else raw_text

    def process_image(self, image: Image.Image, page_num: int = 1) -> PageResult:
        """Process a single image with DeepSeek."""
        if not self._initialized and not self.initialize():
            return self._create_error_result(page_num, "DeepSeek not initialized")

        start_time = time.time()

        try:
            text = self._call_ollama(image, self.OCR_PROMPT)
            processing_time = time.time() - start_time

            return self._create_success_result(
                page_num=page_num,
                text=text,
                processing_time=processing_time,
            )

        except Exception as e:
            return self._create_error_result(page_num, str(e))

    def describe_figure(
        self,
        image: Image.Image,
        figure_type: str = "unknown",
        context: str = "",
    ) -> FigureResult:
        """Describe a figure using DeepSeek vision."""
        if not self._initialized and not self.initialize():
            return FigureResult(
                figure_num=0,
                page_num=0,
                figure_type=figure_type,
                description="DeepSeek not initialized",
            )

        try:
            prompt = self.FIGURE_PROMPT
            if context:
                prompt += f"\n\nContext from surrounding text: {context}"

            description = self._call_ollama(image, prompt)

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
