# Hybrid Privacy-Preserving AI System

A research project combining **Homomorphic Encryption** and **Differential Privacy** to enable secure AI inference on sensitive data — without exposing it to the cloud.

---

## What This Does

We built a system that lets users run AI predictions on sensitive data (medical, financial, student records) while ensuring:

1. **The server never sees raw data** — Homomorphic Encryption encrypts inputs before they leave the user's machine.
2. **Outputs can't be reverse-engineered** — Differential Privacy adds calibrated noise to predictions, blocking inference attacks.

---

## Project Structure

```
hybrid-privacy-ai/
├── backend/
│   ├── main.py              # FastAPI server
│   ├── model.py             # ML model (TensorFlow/sklearn)
│   ├── encryption.py        # TenSEAL HE wrapper
│   ├── privacy.py           # Differential privacy noise
│   └── requirements.txt
├── frontend/
│   └── index.html           # Simple demo UI
├── notebooks/
│   └── demo.ipynb           # Full walkthrough notebook
├── data/
│   └── sample_loan_data.csv
├── models/
│   └── (saved model files go here)
├── run.sh                   # One-command startup
└── README.md
```

---

## Quick Start

### 1. Install dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Train the model (first time only)

```bash
cd backend
python model.py
```

### 3. Start the server

```bash
cd backend
uvicorn main:app --reload --port 8000
```

Or from root:

```bash
bash run.sh
```

### 4. Open the demo

Open `frontend/index.html` in your browser.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| POST | `/predict/plain` | Prediction without privacy (baseline) |
| POST | `/predict/private` | Prediction with differential privacy |
| POST | `/predict/encrypted` | Full HE + DP pipeline |
| GET | `/epsilon` | Current privacy budget |

---

## Privacy Parameters

- **Epsilon (ε)**: Controls privacy strength. Lower = more private. Default: `1.0`
- **Sensitivity**: Maximum output change from one record. Default: `1.0`
- **HE Poly Modulus Degree**: `4096` (balances speed vs. precision)

---

## Team

Built as a research project exploring practical deployment of privacy-preserving machine learning.
