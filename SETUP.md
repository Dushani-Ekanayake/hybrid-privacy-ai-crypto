# Setup Guide

Follow these steps to get the project running locally in VS Code.

---

## Prerequisites

- Python 3.9 or 3.10 (TenSEAL doesn't support 3.11+ yet)
- pip
- VS Code

---

## Step 1 — Open the project

Open VS Code, then:
```
File → Open Folder → select hybrid-privacy-ai/
```

---

## Step 2 — Create a virtual environment

Open the VS Code terminal (`Ctrl + ~`) and run:

```bash
python -m venv venv
```

Activate it:

- **Windows:** `venv\Scripts\activate`
- **Mac/Linux:** `source venv/bin/activate`

VS Code should auto-detect the venv. If prompted, select it as your Python interpreter.

---

## Step 3 — Install dependencies

```bash
cd backend
pip install -r requirements.txt
```

> Note: TenSEAL may take a couple minutes to install — it includes native C++ extensions.

---

## Step 4 — Train the model

```bash
python model.py
```

This generates:
- `data/sample_loan_data.csv` — 500 synthetic loan records
- `models/loan_model.joblib` — trained logistic regression
- `models/scaler.joblib` — feature scaler

---

## Step 5 — Start the server

**Option A — VS Code launch config:**
Press `F5` and select "Start Backend Server"

**Option B — Terminal:**
```bash
uvicorn main:app --reload --port 8000
```

**Option C — From project root:**
```bash
bash run.sh
```

---

## Step 6 — Open the demo UI

Open `frontend/index.html` in your browser.
(Just double-click the file — no server needed for the frontend.)

---

## Step 7 — Explore the API docs

FastAPI generates interactive docs automatically:

```
http://localhost:8000/docs
```

---

## Step 8 — Run the notebook

Open `notebooks/demo.ipynb` in VS Code (Jupyter extension required).
Run all cells for the full walkthrough including visualizations.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `tenseal` install fails | Make sure you're using Python 3.9 or 3.10 |
| `ModuleNotFoundError` | Activate your venv first |
| Server won't start | Check nothing else is on port 8000 |
| Frontend shows "Could not connect" | Make sure the backend is running |
| Model accuracy low | Re-run `python model.py` to retrain |

---

## Project layout recap

```
hybrid-privacy-ai/
├── .vscode/           # VS Code settings, launch configs
├── backend/
│   ├── main.py        # FastAPI server (start here)
│   ├── model.py       # ML model training + inference
│   ├── encryption.py  # TenSEAL HE wrapper
│   ├── privacy.py     # Differential privacy (Laplace)
│   └── requirements.txt
├── frontend/
│   └── index.html     # Demo 
├── notebooks/
│   └── demo.ipynb     # Jupyter walkthrough
├── data/              # Generated data goes here
├── models/            # Saved model files go here
├── api_tests.http     # REST Client test file
├── run.sh             # One-command start
└── SETUP.md           # This file
```
