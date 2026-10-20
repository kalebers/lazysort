from pathlib import Path

import pytest

from lazysort.inference import InferenceError, UnavailableBackend, parse_classification
from lazysort.models import PlanItem
from lazysort.operations import SafetyError, apply_plan, undo_operation
from lazysort.persistence import HistoryStore
from lazysort.planning import sanitize_filename, unique_destination
from lazysort.extraction import UnsupportedFileError, extract_text


VALID = {
    "document_type": "invoice",
    "suggested_filename": "Acme invoice.pdf",
    "suggested_category": "Finance",
    "short_summary": "An invoice from Acme.",
    "confidence": 0.91,
    "reasoning": "The heading and amount indicate an invoice.",
}


def test_valid_model_response():
    result = parse_classification(VALID)
    assert result.document_type == "invoice"
    assert result.confidence == 0.91


def test_low_confidence_result_is_marked_for_review():
    item = PlanItem("source.txt", "destination.txt", parse_classification({**VALID, "confidence": 0.2}))
    assert item.is_low_confidence


@pytest.mark.parametrize("response", ["not json", "{}", '{"confidence": 2}'])
def test_malformed_model_response(response):
    with pytest.raises(InferenceError):
        parse_classification(response)


def test_filename_sanitization():
    assert sanitize_filename("../secret:/report?.txt") == "secret--report?.txt"
    assert sanitize_filename("   ") == "untitled"


def test_duplicate_filenames(tmp_path: Path):
    destination = tmp_path / "report.txt"
    destination.touch()
    assert unique_destination(tmp_path, "report.txt").name == "report (2).txt"


def test_unsupported_file_type(tmp_path: Path):
    path = tmp_path / "image.png"
    path.write_bytes(b"data")
    with pytest.raises(UnsupportedFileError):
        extract_text(path)


def test_inference_failure_is_explicit():
    with pytest.raises(InferenceError):
        UnavailableBackend().classify("private document text")


def test_dry_run_does_not_move(tmp_path: Path):
    source = tmp_path / "source.txt"
    destination = tmp_path / "folder" / "destination.txt"
    source.write_text("hello")
    history = HistoryStore(tmp_path / "history.sqlite3")
    result = apply_plan(tmp_path, [(source, destination)], history, dry_run=True)
    assert result == [(source.resolve(), destination.resolve())]
    assert source.exists()
    assert not destination.exists()
    assert history.recent() == []


def test_path_traversal_is_rejected(tmp_path: Path):
    source = tmp_path / "source.txt"
    source.write_text("hello")
    history = HistoryStore(tmp_path / "history.sqlite3")
    with pytest.raises(SafetyError):
        apply_plan(tmp_path, [(source, tmp_path / ".." / "outside.txt")], history, dry_run=True)


def test_move_and_undo(tmp_path: Path):
    source = tmp_path / "source.txt"
    destination = tmp_path / "folder" / "destination.txt"
    source.write_text("hello")
    history = HistoryStore(tmp_path / "history.sqlite3")
    apply_plan(tmp_path, [(source, destination)], history, dry_run=False)
    operation_id = history.recent()[0][0]
    assert destination.exists() and not source.exists()
    undo_operation(operation_id, source, destination, history)
    assert source.read_text() == "hello"
    assert not destination.exists()
    assert history.recent()[0][4] is not None


def test_undo_does_not_overwrite(tmp_path: Path):
    source = tmp_path / "source.txt"
    destination = tmp_path / "destination.txt"
    source.write_text("old")
    destination.write_text("new")
    history = HistoryStore(tmp_path / "history.sqlite3")
    with pytest.raises(SafetyError):
        undo_operation(1, source, destination, history)
