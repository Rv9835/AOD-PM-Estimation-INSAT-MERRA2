#!/bin/bash

# Simple script to run the Air Quality Dashboard with Streamlit

cd "/Users/chaman/Desktop/HP challenge"
source .venv/bin/activate

echo "════════════════════════════════════════════════════════════"
echo "🌍 Air Quality Prediction Dashboard"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "📋 Make sure the FastAPI server is running in another terminal:"
echo "   cd \"/Users/chaman/Desktop/HP challenge\""
echo "   source .venv/bin/activate"
echo "   python scripts/run_server.py"
echo ""
echo "🚀 Starting Streamlit dashboard..."
echo "📱 Dashboard will open at: http://localhost:8501"
echo ""
echo "════════════════════════════════════════════════════════════"
echo ""

streamlit run app_dashboard.py
