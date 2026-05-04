#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"

"${PYTHON_BIN}" -V
"${PYTHON_BIN}" -m pip --version
"${PYTHON_BIN}" -m pip install --upgrade pip
"${PYTHON_BIN}" -m pip install -r requirements.txt
"${PYTHON_BIN}" -m pip check
"${PYTHON_BIN}" -c "import streamlit; print(f'streamlit={streamlit.__version__}')"
