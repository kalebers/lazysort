from __future__ import annotations

import re
from pathlib import Path

from .models import Classification, PlanItem


def sanitize_filename(name: str, fallback: str = "untitled") -> str:
    raw_name = name.replace("\\", "/").strip()
    suffix = Path(raw_name).suffix.lower()
    stem = raw_name[:-len(suffix)] if suffix else raw_name
    stem = re.sub(r"[\x00-\x1f/\\:]", "-", stem)
    stem = re.sub(r"\s+", " ", stem).strip(" .-")
    stem = stem or fallback
    return f"{stem}{suffix}"


def unique_destination(directory: Path, filename: str, reserved: set[Path] | None = None) -> Path:
    reserved = reserved or set()
    candidate = directory / filename
    stem, suffix = candidate.stem, candidate.suffix
    counter = 2
    while candidate.exists() or candidate in reserved:
        candidate = directory / f"{stem} ({counter}){suffix}"
        counter += 1
    return candidate


def create_plan(directory: Path, source: Path, classification: Classification) -> PlanItem:
    category = sanitize_filename(classification.suggested_category)
    filename = sanitize_filename(classification.suggested_filename, source.stem)
    destination_dir = directory / Path(category).stem
    destination = unique_destination(destination_dir, filename)
    return PlanItem(str(source), str(destination), classification)
