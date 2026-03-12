"""DeepSeek OCR engine adapter.

CLI: deepseek-ocr <path> -o <dir> [--backend ollama|vllm] [--vllm-url] [-q]
Click group with auto-insert of 'process' — `deepseek-ocr paper.pdf` works.
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
            str(pdf_path),
            "-o", str(output_dir),
            "--backend", config.deepseek_backend,
        ]
        if config.deepseek_backend == "vllm":
            cmd.extend(["--vllm-url", config.deepseek_vllm_url])
        if config.quiet:
            cmd.append("-q")
        return cmd
