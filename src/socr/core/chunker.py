"""PDF chunking for long documents.

Splits PDFs that exceed a page threshold into smaller segments so each
chunk stays within engine context/output limits.  Uses PyMuPDF (fitz) for
zero-copy page extraction.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


@dataclass
class PDFChunk:
    """Metadata for a single chunk of a split PDF."""

    chunk_num: int
    start_page: int  # 1-indexed
    end_page: int  # inclusive
    path: Path  # path to the chunk PDF file
    page_count: int


class PDFChunker:
    """Split long PDFs into fixed-size page chunks.

    Parameters
    ----------
    max_pages_per_chunk : int
        Maximum pages in each chunk file (default 20).
    threshold : int
        Only chunk PDFs with more than this many pages (default 30).
    """

    def __init__(
        self,
        max_pages_per_chunk: int = 20,
        threshold: int = 30,
    ) -> None:
        self.max_pages_per_chunk = max_pages_per_chunk
        self.threshold = threshold

    def needs_chunking(self, pdf_path: Path) -> bool:
        """Check if a PDF exceeds the chunk threshold."""
        with fitz.open(pdf_path) as doc:
            return len(doc) > self.threshold

    def chunk(self, pdf_path: Path, output_dir: Path) -> list[PDFChunk]:
        """Split a PDF into chunks.

        Returns a list of :class:`PDFChunk` with paths to the generated
        chunk files inside *output_dir*.  If the PDF does not exceed the
        threshold, returns a single chunk covering the whole document.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        with fitz.open(pdf_path) as src_doc:
            total_pages = len(src_doc)

            if total_pages <= self.threshold:
                # No splitting needed -- return one chunk pointing at the original
                return [
                    PDFChunk(
                        chunk_num=1,
                        start_page=1,
                        end_page=total_pages,
                        path=pdf_path,
                        page_count=total_pages,
                    )
                ]

            chunks: list[PDFChunk] = []
            chunk_num = 0

            for start_0 in range(0, total_pages, self.max_pages_per_chunk):
                chunk_num += 1
                end_0 = min(start_0 + self.max_pages_per_chunk, total_pages)
                page_range = list(range(start_0, end_0))

                chunk_path = output_dir / f"{pdf_path.stem}_chunk{chunk_num}.pdf"

                # Create a new document with only the selected pages
                chunk_doc = fitz.open()
                chunk_doc.insert_pdf(src_doc, from_page=start_0, to_page=end_0 - 1)
                chunk_doc.save(str(chunk_path))
                chunk_doc.close()

                chunks.append(
                    PDFChunk(
                        chunk_num=chunk_num,
                        start_page=start_0 + 1,  # 1-indexed
                        end_page=end_0,  # inclusive, 1-indexed
                        path=chunk_path,
                        page_count=len(page_range),
                    )
                )

            logger.info(
                f"Split {pdf_path.name} ({total_pages} pages) into "
                f"{len(chunks)} chunks of up to {self.max_pages_per_chunk} pages"
            )
            return chunks
