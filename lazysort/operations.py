from __future__ import annotations

import os
import shutil
from pathlib import Path

from .persistence import HistoryStore


class SafetyError(RuntimeError):
    pass


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def apply_plan(
    root: Path,
    moves: list[tuple[Path, Path]],
    history: HistoryStore,
    dry_run: bool = True,
) -> list[tuple[Path, Path]]:
    root = root.resolve()
    checked: list[tuple[Path, Path]] = []
    destinations: set[Path] = set()
    for source, destination in moves:
        source, destination = source.resolve(), destination.resolve()
        if not _inside(source, root) or not _inside(destination, root):
            raise SafetyError("Source and destination must remain inside the selected directory")
        if not source.is_file():
            raise SafetyError(f"Source is not a file: {source}")
        if destination.exists() or destination in destinations:
            raise SafetyError(f"Destination already exists: {destination}")
        destinations.add(destination)
        checked.append((source, destination))
    if dry_run:
        return checked
    completed: list[tuple[Path, Path]] = []
    try:
        for source, destination in checked:
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(source, destination)
            history.record(source, destination)
            completed.append((source, destination))
    except OSError:
        for source, destination in reversed(completed):
            if destination.exists() and not source.exists():
                os.replace(destination, source)
        raise
    return completed


def undo_operation(operation_id: int, source: Path, destination: Path, history: HistoryStore) -> None:
    if source.exists() or not destination.exists():
        raise SafetyError("Undo would overwrite an existing file or destination is missing")
    source.parent.mkdir(parents=True, exist_ok=True)
    os.replace(destination, source)
    history.mark_undone(operation_id)
