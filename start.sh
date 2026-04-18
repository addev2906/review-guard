#!/usr/bin/env bash
# ── ReviewGuard — Start Backend Server ───────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
BACKEND_DIR="$SCRIPT_DIR/backend"
ML_DIR="$SCRIPT_DIR/ML Project 2026"

# Check model exists
if [ ! -f "$ML_DIR/models/fake_review_detector.joblib" ]; then
  echo "⚠  Model not found. Training the model first…"
  source "$VENV_DIR/bin/activate"
  pip install -q -r "$ML_DIR/requirements.txt"
  python "$ML_DIR/train_model.py"
fi

# Install backend deps if needed
source "$VENV_DIR/bin/activate"
pip install -q -r "$BACKEND_DIR/requirements.txt"

echo ""
echo "  ╔═══════════════════════════════════════════╗"
echo "  ║     ReviewGuard Backend Server            ║"
echo "  ║     http://localhost:8000                  ║"
echo "  ║     Docs: http://localhost:8000/docs       ║"
echo "  ╚═══════════════════════════════════════════╝"
echo ""

cd "$BACKEND_DIR"
python -m uvicorn server:app --host 0.0.0.0 --port "${PORT:-8000}" --reload
