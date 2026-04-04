#!/bin/bash


echo ""
echo "═══════════════════════════════════════"
echo "  Hybrid Privacy AI — Starting Up"
echo "═══════════════════════════════════════"
echo ""

cd backend

if [ ! -f "../models/loan_model.joblib" ]; then
  echo "[1/2] Training model (first time setup)..."
  python model.py
  echo ""
fi

echo "[2/2] Starting FastAPI server on http://localhost:8000"
echo ""
echo "  API docs:  http://localhost:8000/docs"
echo "  Frontend:  open frontend/index.html in your browser"
echo ""
echo "Press Ctrl+C to stop."
echo ""

uvicorn main:app --reload --port 8000
