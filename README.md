# LazySort

LazySort is a privacy-first, local-first file organizer for macOS and other desktop
platforms. It extracts text from supported documents, optionally asks a compact local
model for a structured suggestion, and presents a reviewable plan before moving anything.

## Why local open-source AI?

Documents often contain personal, financial, or confidential information. Local inference
keeps that content on the user's machine and avoids paid API keys, usage telemetry, and
network dependency. LazySort deliberately treats model output as an untrusted suggestion:
responses are schema-validated, low-confidence results remain reviewable, and filesystem
changes require explicit approval.

The MVP is designed for an 8 GB Intel Mac: one document at a time, small prompts, short
generation limits, CPU-only llama.cpp configuration, and no automatic model downloads.

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

PDF extraction uses PyMuPDF. Text and Markdown are supported without extra tooling.
Folder monitoring is intentionally not included in the default flow; if added later,
`watchdog` should remain optional and disabled by default.

### Included local model

LazySort uses **Qwen2.5-0.5B-Instruct Q4_K_M GGUF**, an Apache-2.0 open-weight model
quantized for CPU inference. It is approximately 0.5 GB and is a practical starting point
for an 8 GB Intel MacBook Pro. The model is downloaded into `models/` and is ignored by
Git; model files are not source code and should not be committed.

```bash
mkdir -p models
curl -L --fail --retry 3 \
  -o models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf
```

`requirements.txt` installs `llama-cpp-python` for CPU inference. On older macOS systems,
installation may compile from source and can take a few minutes. If the runtime tries to
initialize Metal and fails, reinstall it with Metal disabled:

```bash
CMAKE_ARGS="-DGGML_METAL=OFF" python -m pip install --force-reinstall --no-cache-dir llama-cpp-python==0.2.90
```

LazySort sets
`n_gpu_layers=0`, uses a 2,048-token context, processes one document at a time, and
generates at most 256 tokens. It never downloads a model automatically.

The UI auto-detects the downloaded file. To use another compatible GGUF model, set
`LAZYSORT_MODEL_PATH=/absolute/path/to/model.gguf` before starting Streamlit. Check the
model license, chat template, quantization, context size, and llama.cpp compatibility
before use. If the runtime or model cannot load, LazySort remains usable in manual mode.

## Usage and safety

1. Enter a folder path and click **Scan folder**.
2. Review or edit each destination in the organization plan.
3. Use **Dry-run preview** to inspect moves without changing files.
4. Click **Approve and organize** only after reviewing the plan.
5. Use **Operation history** to undo a completed move.

LazySort never overwrites an existing destination, deletes originals as part of normal
organization, or allows source/destination paths outside the selected folder. It stores
history in `.lazysort/history.sqlite3` inside the selected folder. Do not point it at a
folder whose `.lazysort` directory should be managed by another application.

## Demo

Use the included `sample_documents/` folder for a safe recording. Show the scan, edit a
destination, run the dry-run preview, approve the move, and undo it from operation history.
For a model-enabled demo, configure a small local GGUF backend and visibly include a
low-confidence result so the review workflow is clear.

## Development

```bash
pytest -q
```

The test suite uses temporary directories and synthetic documents only.
