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


st.set_page_config(page_title="LazySort", page_icon="🗂️", layout="wide")
st.title("🗂️ LazySort")
st.caption("A local-first, reviewable file organizer. Nothing is uploaded.")

root_value = st.text_input("Folder to organize", value=str(Path.home() / "Downloads"))
root = Path(root_value).expanduser()
database = root / ".lazysort" / "history.sqlite3"
history = HistoryStore(database)
default_model = Path(__file__).parent / "models" / "qwen2.5-0.5b-instruct-q4_k_m.gguf"
model_path = os.getenv("LAZYSORT_MODEL_PATH") or (
    str(default_model) if default_model.exists() else None
)
if model_path:
    try:
        backend: InferenceBackend = LlamaCppBackend(model_path)
    except InferenceError as exc:
        st.warning(f"Local model unavailable; using manual review: {exc}")
        backend = UnavailableBackend()
else:
    backend = UnavailableBackend()

if "plan" not in st.session_state:
    st.session_state.plan = []

if st.button("Scan folder", type="primary"):
    if not root.is_dir():
        st.error("Select an existing directory.")
    else:
        items: list[PlanItem] = []
        for source in supported_files(root):
            try:
                text = extract_text(source)
                items.append(PlanItem(str(source), str(source), None, f"Text extracted ({len(text)} chars)"))
                try:
                    classification = backend.classify(text)
                except InferenceError as exc:
                    items[-1].error = str(exc)
                else:
                    items[-1] = create_plan(root, source, classification)
            except (RuntimeError, UnsupportedFileError) as exc:
                items.append(PlanItem(str(source), str(source), None, str(exc)))
        st.session_state.plan = items

plan = st.session_state.plan
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

with st.expander("Operation history"):
    for operation_id, source, destination, created_at, undone_at in history.recent():
        st.write(f"{operation_id}: {source} → {destination} ({created_at})")
        if not undone_at and st.button(f"Undo {operation_id}", key=f"undo-{operation_id}"):
            try:
                undo_operation(operation_id, Path(source), Path(destination), history)
                st.success("Operation undone.")
                st.rerun()
            except (OSError, SafetyError) as exc:
                st.error(str(exc))
