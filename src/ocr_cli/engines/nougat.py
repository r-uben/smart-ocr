"""Nougat OCR engine adapter.

Nougat is optimized for scientific/academic papers with complex layouts,
equations, and figures. It's free and runs locally.
"""

import tempfile
import time
from pathlib import Path

from PIL import Image

from ocr_cli.core.config import NougatConfig
from ocr_cli.core.result import PageResult, PageStatus
from ocr_cli.engines.base import BaseEngine, EngineCapabilities


class NougatEngine(BaseEngine):
    """Adapter for Nougat OCR (via nougat-ocr-cli)."""

    def __init__(self, config: NougatConfig | None = None) -> None:
        super().__init__()
        self.config = config or NougatConfig()
        self._model = None

    @property
    def name(self) -> str:
        return "nougat"

    @property
    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            name="nougat",
            supports_pdf=True,
            supports_images=True,
            supports_batch=True,
            supports_figures=False,  # No built-in figure description
            is_local=True,
            cost_per_page=0.0,
            best_for=["academic", "scientific", "equations", "papers"],
        )

    def initialize(self) -> bool:
        """Initialize Nougat model."""
        if self._initialized:
            return True

        try:
            from nougat_ocr import NougatOCR

            self._model = NougatOCR(model=self.config.model)
            self._initialized = True
            return True
        except ImportError:
            return False
        except Exception:
            return False

    def process_image(self, image: Image.Image, page_num: int = 1) -> PageResult:
        """Process a single image with Nougat."""
        if not self._initialized and not self.initialize():
            return self._create_error_result(page_num, "Nougat not initialized")

        start_time = time.time()

        try:
            # Save image to temp file (Nougat works better with files)
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                image.save(tmp.name, format="PNG")
                tmp_path = Path(tmp.name)

            # Process with Nougat
            result = self._model.process_image(str(tmp_path))
            text = result.get("text", "") if isinstance(result, dict) else str(result)

            # Cleanup
            tmp_path.unlink()

            processing_time = time.time() - start_time
            return self._create_success_result(
                page_num=page_num,
                text=text,
                processing_time=processing_time,
            )

        except Exception as e:
            return self._create_error_result(page_num, str(e))

    def process_pdf(self, pdf_path: Path) -> list[PageResult]:
        """Process entire PDF with Nougat (batch mode)."""
        if not self._initialized and not self.initialize():
            return [self._create_error_result(1, "Nougat not initialized")]

        start_time = time.time()

        try:
            # Use Nougat's batch PDF processing
            results = self._model.process_pdf(
                str(pdf_path),
                batch_size=self.config.batch_size,
                no_skipping=self.config.no_skipping,
            )

            page_results = []
            if isinstance(results, list):
                for i, result in enumerate(results):
                    text = result.get("text", "") if isinstance(result, dict) else str(result)
                    page_results.append(
                        self._create_success_result(
                            page_num=i + 1,
                            text=text,
                            processing_time=(time.time() - start_time) / len(results),
                        )
                    )
            else:
                # Single result for entire document
                text = results.get("text", "") if isinstance(results, dict) else str(results)
                page_results.append(
                    self._create_success_result(
                        page_num=1,
                        text=text,
                        processing_time=time.time() - start_time,
                    )
                )

            return page_results

        except Exception as e:
            return [self._create_error_result(1, str(e))]
