#!/usr/bin/env bash
# One command entry point. Runs the pipeline end to end and writes to out/.
#
#   ./run.sh          run every step implemented so far
#   ./run.sh test     run the test suite only
#   ./run.sh web      launch the browser based decision review on port 8000
#
# Override the dataset location with YLOOKUP_DATA_DIR if it is not in data/.

set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"

if [ "${1:-}" = "test" ]; then
    exec "$PYTHON" -m pytest
fi

if [ "${1:-}" = "web" ]; then
    exec env PYTHONPATH="$PWD/src" "$PYTHON" -m uvicorn web.app:app --host 0.0.0.0 --port 8000
fi

echo "Installing dependencies..."
"$PYTHON" -m pip install -q -r requirements.txt 2>/dev/null \
  || "$PYTHON" -m pip install -q --break-system-packages -r requirements.txt

echo
"$PYTHON" -m pytest || { echo "Tests failed; stopping before the pipeline runs."; exit 1; }

echo
"$PYTHON" scripts/step1_inventory.py

echo
"$PYTHON" scripts/step2_mappings.py

echo
"$PYTHON" scripts/step3_decisions.py

echo
echo "Reports written to out/"
ls -1 out/
