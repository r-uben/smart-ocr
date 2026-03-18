"""Tests for PDFChunker -- splitting long PDFs into page-range segments."""

from pathlib import Path

import fitz  # PyMuPDF
import pytest

from socr.core.chunker import PDFChunk, PDFChunker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_pdf(path: Path, num_pages: int) -> Path:
    """Create a minimal PDF with *num_pages* blank pages."""
    doc = fitz.open()
    for _ in range(num_pages):
        doc.new_page(width=612, height=792)
    doc.save(str(path))
    doc.close()
    return path


# ---------------------------------------------------------------------------
# needs_chunking
# ---------------------------------------------------------------------------

class TestNeedsChunking:
    def test_small_pdf_does_not_need_chunking(self, tmp_path: Path) -> None:
        pdf = _create_pdf(tmp_path / "small.pdf", num_pages=10)
        chunker = PDFChunker(max_pages_per_chunk=20, threshold=30)
        assert chunker.needs_chunking(pdf) is False

    def test_large_pdf_needs_chunking(self, tmp_path: Path) -> None:
        pdf = _create_pdf(tmp_path / "large.pdf", num_pages=50)
        chunker = PDFChunker(max_pages_per_chunk=20, threshold=30)
        assert chunker.needs_chunking(pdf) is True

    def test_exactly_at_threshold_does_not_need_chunking(self, tmp_path: Path) -> None:
        pdf = _create_pdf(tmp_path / "exact.pdf", num_pages=30)
        chunker = PDFChunker(max_pages_per_chunk=20, threshold=30)
        assert chunker.needs_chunking(pdf) is False

    def test_one_above_threshold_needs_chunking(self, tmp_path: Path) -> None:
        pdf = _create_pdf(tmp_path / "above.pdf", num_pages=31)
        chunker = PDFChunker(max_pages_per_chunk=20, threshold=30)
        assert chunker.needs_chunking(pdf) is True

    def test_single_page_does_not_need_chunking(self, tmp_path: Path) -> None:
        pdf = _create_pdf(tmp_path / "single.pdf", num_pages=1)
        chunker = PDFChunker(max_pages_per_chunk=20, threshold=30)
        assert chunker.needs_chunking(pdf) is False


# ---------------------------------------------------------------------------
# chunk
# ---------------------------------------------------------------------------

class TestChunk:
    def test_below_threshold_returns_single_chunk_pointing_at_original(
        self, tmp_path: Path
    ) -> None:
        pdf = _create_pdf(tmp_path / "small.pdf", num_pages=10)
        chunker = PDFChunker(max_pages_per_chunk=20, threshold=30)

        chunks = chunker.chunk(pdf, tmp_path / "chunks")

        assert len(chunks) == 1
        assert chunks[0].path == pdf  # points at original, not a copy
        assert chunks[0].start_page == 1
        assert chunks[0].end_page == 10
        assert chunks[0].page_count == 10
        assert chunks[0].chunk_num == 1

    def test_chunk_produces_correct_number_of_chunks(self, tmp_path: Path) -> None:
        pdf = _create_pdf(tmp_path / "long.pdf", num_pages=50)
        chunker = PDFChunker(max_pages_per_chunk=20, threshold=30)

        chunks = chunker.chunk(pdf, tmp_path / "chunks")

        # 50 pages / 20 per chunk = 3 chunks (20 + 20 + 10)
        assert len(chunks) == 3

    def test_chunk_page_ranges_are_correct(self, tmp_path: Path) -> None:
        pdf = _create_pdf(tmp_path / "long.pdf", num_pages=50)
        chunker = PDFChunker(max_pages_per_chunk=20, threshold=30)

        chunks = chunker.chunk(pdf, tmp_path / "chunks")

        assert chunks[0].start_page == 1
        assert chunks[0].end_page == 20
        assert chunks[0].page_count == 20

        assert chunks[1].start_page == 21
        assert chunks[1].end_page == 40
        assert chunks[1].page_count == 20

        assert chunks[2].start_page == 41
        assert chunks[2].end_page == 50
        assert chunks[2].page_count == 10

    def test_chunk_pdfs_have_correct_page_count(self, tmp_path: Path) -> None:
        pdf = _create_pdf(tmp_path / "long.pdf", num_pages=50)
        chunker = PDFChunker(max_pages_per_chunk=20, threshold=30)

        chunks = chunker.chunk(pdf, tmp_path / "chunks")

        for chunk in chunks:
            assert chunk.path.exists()
            with fitz.open(chunk.path) as doc:
                assert len(doc) == chunk.page_count

    def test_chunk_exactly_divisible(self, tmp_path: Path) -> None:
        """60 pages / 20 per chunk = exactly 3 chunks, no remainder."""
        pdf = _create_pdf(tmp_path / "exact.pdf", num_pages=60)
        chunker = PDFChunker(max_pages_per_chunk=20, threshold=30)

        chunks = chunker.chunk(pdf, tmp_path / "chunks")

        assert len(chunks) == 3
        for chunk in chunks:
            assert chunk.page_count == 20

    def test_chunk_just_above_threshold(self, tmp_path: Path) -> None:
        """31 pages with threshold=30 -> 2 chunks (20 + 11)."""
        pdf = _create_pdf(tmp_path / "above.pdf", num_pages=31)
        chunker = PDFChunker(max_pages_per_chunk=20, threshold=30)

        chunks = chunker.chunk(pdf, tmp_path / "chunks")

        assert len(chunks) == 2
        assert chunks[0].page_count == 20
        assert chunks[1].page_count == 11

    def test_custom_chunk_size(self, tmp_path: Path) -> None:
        """Verify custom chunk sizes work correctly."""
        pdf = _create_pdf(tmp_path / "long.pdf", num_pages=40)
        chunker = PDFChunker(max_pages_per_chunk=15, threshold=20)

        chunks = chunker.chunk(pdf, tmp_path / "chunks")

        # 40 pages / 15 per chunk = 3 chunks (15 + 15 + 10)
        assert len(chunks) == 3
        assert chunks[0].page_count == 15
        assert chunks[1].page_count == 15
        assert chunks[2].page_count == 10

    def test_chunk_creates_output_directory(self, tmp_path: Path) -> None:
        pdf = _create_pdf(tmp_path / "long.pdf", num_pages=40)
        chunker = PDFChunker(max_pages_per_chunk=20, threshold=30)

        out_dir = tmp_path / "nested" / "chunks"
        assert not out_dir.exists()

        chunks = chunker.chunk(pdf, out_dir)

        assert out_dir.exists()
        assert len(chunks) == 2

    def test_chunk_nums_are_sequential(self, tmp_path: Path) -> None:
        pdf = _create_pdf(tmp_path / "long.pdf", num_pages=80)
        chunker = PDFChunker(max_pages_per_chunk=20, threshold=30)

        chunks = chunker.chunk(pdf, tmp_path / "chunks")

        for i, chunk in enumerate(chunks, start=1):
            assert chunk.chunk_num == i

    def test_all_pages_covered(self, tmp_path: Path) -> None:
        """Verify that chunks cover all pages with no gaps or overlaps."""
        pdf = _create_pdf(tmp_path / "long.pdf", num_pages=73)
        chunker = PDFChunker(max_pages_per_chunk=20, threshold=30)

        chunks = chunker.chunk(pdf, tmp_path / "chunks")

        total = sum(c.page_count for c in chunks)
        assert total == 73

        # No gaps
        for i in range(len(chunks) - 1):
            assert chunks[i].end_page + 1 == chunks[i + 1].start_page

        # Boundaries correct
        assert chunks[0].start_page == 1
        assert chunks[-1].end_page == 73
