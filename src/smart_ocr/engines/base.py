"""Base engine adapter for smart-ocr v1.0.

Engines call their CLI once per document (one subprocess per PDF).
No more per-page subprocess calls or PIL image passing.
"""

import logging
import subprocess
import tempfile
import time
from abc import ABC, abstractmethod
from pathlib import Path

from smart_ocr.core.config import PipelineConfig
from smart_ocr.core.result import DocumentResult, DocumentStatus

logger = logging.getLogger(__name__)


def sanitize_filename(name: str) -> str:
    """Sanitize a filename for use as a directory name."""
    return "".join(c if c.isalnum() or c in "._- " else "_" for c in name).strip()


class BaseEngine(ABC):
    """Abstract base class for CLI-based OCR engines.

    Each engine wraps a sibling CLI tool (gemini-ocr, nougat-ocr, etc.).
    The contract: call CLI once per PDF, read output markdown from
    {output_dir}/{stem}/{stem}.md.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Engine identifier (matches EngineType value)."""
        ...

    @property
    @abstractmethod
    def cli_command(self) -> str:
        """The CLI binary name (e.g., 'gemini-ocr', 'nougat-ocr')."""
        ...

    def is_available(self) -> bool:
        """Check if the CLI tool is installed and callable."""
        try:
            result = subprocess.run(
                [self.cli_command, "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.returncode == 0
        except (subprocess.SubprocessError, FileNotFoundError):
            return False

    def process_document(
        self,
        pdf_path: Path,
        output_dir: Path,
        config: PipelineConfig,
    ) -> DocumentResult:
        """Process a PDF document via CLI subprocess.

        Calls the CLI once on the whole PDF, reads the output markdown.
        """
        start_time = time.time()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_out = Path(tmpdir)
            cmd = self._build_command(pdf_path, tmp_out, config)

            logger.info(f"[{self.name}] Running: {' '.join(cmd)}")

            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=config.timeout,
                )

                if result.returncode != 0:
                    stderr = result.stderr.strip() if result.stderr else "Unknown error"
                    logger.error(f"[{self.name}] CLI failed: {stderr}")
                    return DocumentResult(
                        document_path=pdf_path,
                        engine=self.name,
                        status=DocumentStatus.ERROR,
                        error=f"CLI exited {result.returncode}: {stderr[:500]}",
                        processing_time=time.time() - start_time,
                    )

                # Read output markdown
                markdown = self._read_output(pdf_path, tmp_out)
                if markdown is None:
                    return DocumentResult(
                        document_path=pdf_path,
                        engine=self.name,
                        status=DocumentStatus.ERROR,
                        error="CLI produced no output markdown",
                        processing_time=time.time() - start_time,
                    )

                return DocumentResult(
                    document_path=pdf_path,
                    engine=self.name,
                    status=DocumentStatus.SUCCESS,
                    markdown=markdown,
                    processing_time=time.time() - start_time,
                )

            except subprocess.TimeoutExpired:
                return DocumentResult(
                    document_path=pdf_path,
                    engine=self.name,
                    status=DocumentStatus.ERROR,
                    error=f"Timeout after {config.timeout}s",
                    processing_time=time.time() - start_time,
                )

    @abstractmethod
    def _build_command(
        self,
        pdf_path: Path,
        output_dir: Path,
        config: PipelineConfig,
    ) -> list[str]:
        """Build the CLI command for this engine."""
        ...

    def _read_output(self, pdf_path: Path, output_dir: Path) -> str | None:
        """Read the output markdown from the CLI's output directory.

        All sibling CLIs write to: {output_dir}/{stem}/{stem}.md
        """
        stem = sanitize_filename(pdf_path.stem)
        md_path = output_dir / stem / f"{stem}.md"

        if md_path.exists():
            text = md_path.read_text(encoding="utf-8")
            return self._strip_frontmatter(text)

        # Fallback: search for any .md file in output
        for md_file in output_dir.rglob("*.md"):
            text = md_file.read_text(encoding="utf-8")
            return self._strip_frontmatter(text)

        return None

    @staticmethod
    def _strip_frontmatter(text: str) -> str:
        """Remove YAML frontmatter if present."""
        if text.startswith("---"):
            parts = text.split("---", 2)
            if len(parts) >= 3:
                return parts[2].strip()
        return text
