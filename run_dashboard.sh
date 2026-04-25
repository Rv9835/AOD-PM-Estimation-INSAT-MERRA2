#!/bin/bash

# Simple script to run the Air Quality Dashboard with Streamlit

cd "$(dirname "$0")"
source .venv/bin/activate

echo "════════════════════════════════════════════════════════════"
echo "🌍 Air Quality Prediction Dashboard"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "📋 Make sure the FastAPI server is running in another terminal:"
echo "   cd \"$(pwd)\""
echo "   source .venv/bin/activate"
echo "   python backend/scripts/run_server.py"
echo ""
echo "🚀 Starting Streamlit dashboard..."
echo "📱 Dashboard will open at: http://localhost:8501"
echo ""
echo "════════════════════════════════════════════════════════════"
echo ""

streamlit run frontend/app_dashboard.py
