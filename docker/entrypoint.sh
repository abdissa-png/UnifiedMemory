#!/usr/bin/env bash
set -euo pipefail

exec PYTHONPATH=/app/src uvicorn unified_memory.api.app:app --host 0.0.0.0 --port 8000 "$@"
