"""DeepSeek OCR engine adapter.

CLI: deepseek-ocr process <path> -o <dir> [--backend ollama|vllm] [--vllm-url]
Click group — requires explicit 'process' subcommand.
"""

from pathlib import Path

from smart_ocr.core.config import PipelineConfig
from smart_ocr.engines.base import BaseEngine


class DeepSeekEngine(BaseEngine):
    """Adapter for deepseek-ocr-cli."""

    @property
    def name(self) -> str:
        return "deepseek"

    @property
    def cli_command(self) -> str:
        return "deepseek-ocr"

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
        ]
        if config.deepseek_backend == "vllm":
            cmd.extend(["--vllm-url", config.deepseek_vllm_url])
        return cmd
