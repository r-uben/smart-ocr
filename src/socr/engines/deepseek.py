"""DeepSeek OCR engine adapter.

CLI: deepseek-ocr process <path> -o <dir> [--backend ollama|vllm] [--vllm-url] [--task ocr|format]
Click group — requires explicit 'process' subcommand.

Note on task modes:
  - "ocr": Raw transcription. DeepSeek-OCR hallucinates formatting instructions
    (font sizes, page margins, submission guidelines) with this mode.
  - "format": Markdown-structured output. Better for academic papers but
    may add unwanted structure to simple documents.

The default is "format" since the normalizer strips hallucinated instructions
anyway, and structured output is more useful downstream.
"""

import logging
import subprocess
from pathlib import Path

from socr.core.config import PipelineConfig
from socr.engines.base import BaseEngine

logger = logging.getLogger(__name__)

OLLAMA_MODEL = "deepseek-ocr"


def _check_ollama_model(model_name: str) -> str | None:
    """Check if an Ollama model is available. Returns error message or None."""
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
        if model_name not in model_names:
            return (
                f"Ollama model '{model_name}' not found. "
                f"Pull it with: ollama pull {model_name}"
            )
    except FileNotFoundError:
        return "Ollama is not installed (ollama command not found)"
    except subprocess.TimeoutExpired:
        return "Ollama did not respond (timeout)"
    return None


class DeepSeekEngine(BaseEngine):
    """Adapter for deepseek-ocr-cli."""

    @property
    def name(self) -> str:
        return "deepseek"

    @property
    def cli_command(self) -> str:
        return "deepseek-ocr"

    def is_available(self) -> bool:
        """Check CLI is installed AND Ollama model is available (for ollama backend)."""
        if not super().is_available():
            return False
        # Quick pre-check: if using ollama backend, verify the model is pulled
        # to avoid slow failures at process time.
        error = _check_ollama_model(OLLAMA_MODEL)
        if error:
            logger.debug(f"[{self.name}] {error}")
            return False
        return True

    def process_document(self, pdf_path, output_dir, config):
        """Process document, with Ollama model pre-check for ollama backend."""
        if config.deepseek_backend != "vllm":
            error = _check_ollama_model(OLLAMA_MODEL)
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
            # Use "format" for structured markdown output.
            # "ocr" mode triggers formatting hallucinations.
            "--task", "format",
        ]
        if config.deepseek_backend == "vllm":
            cmd.extend(["--vllm-url", config.deepseek_vllm_url])
        return cmd
