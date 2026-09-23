#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

if [[ -x "$PROJECT_ROOT/.venv/bin/streamlit" ]]; then
  STREAMLIT="$PROJECT_ROOT/.venv/bin/streamlit"
elif command -v streamlit >/dev/null 2>&1; then
  STREAMLIT="$(command -v streamlit)"
else
  printf 'Streamlit is not installed. Install dependencies first with:\n' >&2
  printf '  python3 -m pip install -r requirements.txt\n' >&2
  exit 1
fi

exec "$STREAMLIT" run app.py "$@"