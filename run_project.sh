#!/bin/bash
# Force the execution context to the project root
cd "$(dirname "$0")"

echo "🚀 Starting Multi-City Air Pollution Dashboard..."
# Run the app specifying the explicit path
if [ -x ".venv/bin/python" ]; then
	PYTHON_CMD=".venv/bin/python"
else
	PYTHON_CMD="python3"
fi

"$PYTHON_CMD" -m streamlit run frontend/app_dashboard.py
