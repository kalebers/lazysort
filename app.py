from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from lazysort.extraction import UnsupportedFileError, extract_text, supported_files
from lazysort.inference import InferenceBackend, InferenceError, LlamaCppBackend, UnavailableBackend
from lazysort.models import PlanItem
from lazysort.operations import SafetyError, apply_plan, undo_operation
from lazysort.persistence import HistoryStore
from lazysort.planning import create_plan


@st.cache_resource(show_spinner="Loading local AI model...")
def load_backend(model_path: str | None) -> tuple[InferenceBackend, str | None]:
    if not model_path:
        return UnavailableBackend(), None
    try:
        return LlamaCppBackend(model_path), None
    except InferenceError as exc:
        return UnavailableBackend(), str(exc)


st.set_page_config(page_title="LazySort", page_icon="🗂️", layout="wide")
st.title("🗂️ LazySort")
st.caption("A local-first, reviewable file organizer. Nothing is uploaded.")
st.info("Supported documents: PDF, Markdown, and plain text. LazySort organizes existing files; it does not create new PDF documents.")

root_value = st.text_input("Folder to organize", value=str(Path.home() / "Downloads"))
root = Path(root_value).expanduser()
if not root.is_dir():
    st.error(f"Folder not found: {root}. Enter an existing folder path.")
    st.stop()

database = root / ".lazysort" / "history.sqlite3"
if "history" not in st.session_state:
    st.session_state.history = None

st.caption(f"Selected folder: {root}")
st.caption("LazySort creates its hidden .lazysort folder only when you scan or approve a plan.")

default_model = Path(__file__).parent / "models" / "qwen2.5-0.5b-instruct-q4_k_m.gguf"
model_path = os.getenv("LAZYSORT_MODEL_PATH") or (
    str(default_model) if default_model.exists() else None
)
if model_path:
    backend, model_error = load_backend(model_path)
    if model_error:
        st.warning(f"Local model unavailable; using manual review: {model_error}")
else:
    backend, model_error = load_backend(None)

if "plan" not in st.session_state:
    st.session_state.plan = []
if "scan_message" not in st.session_state:
    st.session_state.scan_message = None

if st.button("Scan folder", type="primary"):
    if not root.is_dir():
        st.error("Select an existing directory.")
    else:
        try:
            history = HistoryStore(database)
            st.session_state.history = history
        except OSError as exc:
            st.error(f"Cannot use this folder for LazySort history: {exc}")
            st.info("Choose a folder you can read and write, then try again.")
            st.stop()

        items: list[PlanItem] = []
        files = supported_files(root)
        if not files:
            st.session_state.plan = []
            st.session_state.scan_message = "No supported documents found. Add PDF, Markdown, or plain-text files to this folder."
            st.info(st.session_state.scan_message)
            st.stop()

        with st.status(f"Scanning {len(files)} supported document(s)...", expanded=True) as status:
            for source in files:
                st.write(f"Reading {source.name}")
                try:
                    text = extract_text(source)
                    item = PlanItem(str(source), str(source), None, f"Text extracted ({len(text)} chars)")
                    try:
                        classification = backend.classify(text)
                    except InferenceError as exc:
                        item.error = str(exc)
                        items.append(item)
                    else:
                        items.append(create_plan(root, source, classification))
                except (OSError, RuntimeError, UnsupportedFileError) as exc:
                    items.append(PlanItem(str(source), str(source), None, str(exc)))
            status.update(label="Scan complete", state="complete")
        st.session_state.plan = items
        st.session_state.scan_message = (
            f"Found {len(items)} supported document(s)."
            if items
            else "No supported documents found. Add PDF, Markdown, or plain-text files to this folder."
        )

plan = st.session_state.plan
if st.session_state.scan_message:
    st.info(st.session_state.scan_message)
if plan:
    st.subheader("Organization plan")
    for index, item in enumerate(plan):
        columns = st.columns([2, 2, 1, 3])
        columns[0].write(Path(item.source).name)
        item.destination = columns[1].text_input(
            "Destination", value=item.destination, key=f"destination-{index}"
        )
        columns[2].write(f"{item.confidence:.0%}" if item.confidence is not None else "—")
        if item.error:
            columns[3].warning(item.error)
        elif item.is_low_confidence:
            columns[3].warning("Low confidence — review carefully")
        else:
            columns[3].write("Ready")

    if st.button("Dry-run preview"):
        moves = [(Path(item.source), Path(item.destination)) for item in plan if item.source != item.destination]
        try:
            preview = apply_plan(root, moves, history, dry_run=True)
            st.write(f"{len(preview)} move(s) would be applied. Originals are not changed.")
            st.dataframe([{"from": str(source), "to": str(destination)} for source, destination in preview])
        except (OSError, SafetyError) as exc:
            st.error(str(exc))

    if st.button("Approve and organize", type="primary"):
        moves = [(Path(item.source), Path(item.destination)) for item in plan if item.source != item.destination]
        try:
            completed = apply_plan(root, moves, history, dry_run=False)
            st.success(f"Organized {len(completed)} file(s).")
        except (OSError, SafetyError) as exc:
            st.error(str(exc))

with st.expander("Operation history", expanded=True):
    history_for_view = st.session_state.get("history") or HistoryStore(database)
    operations = history_for_view.recent()
    if not operations:
        st.caption("No approved operations yet. Run a scan, review the plan, and approve a move to create history.")
    for operation_id, source, destination, created_at, undone_at in operations:
        st.write(f"{operation_id}: {source} → {destination} ({created_at})")
        if not undone_at and st.button(f"Undo {operation_id}", key=f"undo-{operation_id}"):
            try:
                undo_operation(operation_id, Path(source), Path(destination), history_for_view)
                st.success("Operation undone.")
                st.rerun()
            except (OSError, SafetyError) as exc:
                st.error(str(exc))
