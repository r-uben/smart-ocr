"""GLM-OCR engine adapter.

CLI: glm-ocr process <path> -o <dir> [--backend ollama|transformers|vllm]
Click group — requires explicit 'process' subcommand.
"""

import logging
import subprocess
from pathlib import Path

from socr.core.config import PipelineConfig
from socr.engines.base import BaseEngine

logger = logging.getLogger(__name__)

OLLAMA_MODEL = "glm-ocr"


class GLMEngine(BaseEngine):
    """Adapter for glm-ocr-cli."""

    @property
    def name(self) -> str:
        return "glm"

    @property
    def cli_command(self) -> str:
        return "glm-ocr"

    def is_available(self) -> bool:
        """Check CLI is installed."""
        return super().is_available()

    def check_ollama_model(self) -> str | None:
        """Check if the Ollama model is available. Returns error message or None."""
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                return "Ollama is not running or not installed"
            model_names = []
            for line in result.stdout.strip().splitlines()[1:]:
                parts = line.split()
                if parts:
                    name = parts[0].split(":")[0]
                    model_names.append(name)
            if OLLAMA_MODEL not in model_names:
                return (
                    f"Ollama model '{OLLAMA_MODEL}' not found. "
                    f"Pull it with: ollama pull {OLLAMA_MODEL}"
                )
        except FileNotFoundError:
            return "Ollama is not installed (ollama command not found)"
        except subprocess.TimeoutExpired:
            return "Ollama did not respond (timeout)"
        return None

    def process_document(self, pdf_path, output_dir, config):
        """Process document, with Ollama model pre-check."""
        if config.glm_backend == "ollama":
            error = self.check_ollama_model()
            if error:
                from socr.core.result import DocumentStatus, EngineResult, FailureMode
                logger.error(f"[{self.name}] {error}")
                return EngineResult(
                    document_path=pdf_path,
                    engine=self.name,
                    status=DocumentStatus.ERROR,
                    failure_mode=FailureMode.MODEL_UNAVAILABLE,
                    error=error,
                    processing_time=0.0,
                )
        return super().process_document(pdf_path, output_dir, config)

    def _build_command(
        self,
        pdf_path: Path,
        output_dir: Path,
        config: PipelineConfig,
    ) -> list[str]:
        cmd = [
            self.cli_command,
            "process",
            str(pdf_path),
            "-o", str(output_dir),
            "--backend", config.glm_backend,
        ]
        return cmd
