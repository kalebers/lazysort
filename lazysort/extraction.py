from __future__ import annotations

from pathlib import Path

SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md"}


class UnsupportedFileError(ValueError):
    """Raised when a file is outside the supported document set."""


def extract_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="replace")
    if suffix == ".pdf":
        try:
            import fitz
        except ImportError as exc:
            raise RuntimeError("PDF support requires PyMuPDF (fitz)") from exc
        with fitz.open(path) as document:
            return "\n".join(page.get_text() for page in document)
    raise UnsupportedFileError(f"Unsupported file type: {suffix or 'none'}")


def supported_files(directory: Path) -> list[Path]:
    return sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )
