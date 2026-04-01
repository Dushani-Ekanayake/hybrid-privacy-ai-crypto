

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import numpy as np
import time

from model import load_model, predict_plain, get_model_weights
from privacy import DifferentialPrivacy
from encryption import (
    create_context,
    encrypt_vector,
    decrypt_vector,
    deserialize_encrypted,
    he_dot_product,
    he_add_bias,
    sigmoid_approx,
)


app = FastAPI(
    title="Hybrid Privacy AI",
    description="Secure AI inference using Homomorphic Encryption + Differential Privacy",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model on startup
model, scaler = load_model()
weights, bias = get_model_weights(model, scaler)

# Privacy engine
dp = DifferentialPrivacy(epsilon=1.0, sensitivity=1.0)

# HE context (server-side, for running operations on encrypted data)
he_context = create_context()


class PlainInput(BaseModel):
    income: float = Field(..., example=50000)
    credit_score: float = Field(..., example=720)
    debt: float = Field(..., example=10000)
    employment_years: float = Field(..., example=5)


class EncryptedInput(BaseModel):
    ciphertext: str = Field(..., description="Base64-encoded TenSEAL CKKS vector (scaled features)")


class PredictionResult(BaseModel):
    probability: float
    decision: str
    privacy_applied: bool
    noise_added: Optional[float] = None
    latency_ms: float


def scale_features(features: list) -> list:
    """Apply the same scaling the model was trained with."""
    import numpy as np
    x = np.array(features).reshape(1, -1)
    return scaler.transform(x)[0].tolist()


def make_decision(prob: float) -> str:
    return "Approved ✓" if prob >= 0.5 else "Rejected ✗"


@app.get("/")
def root():
    return {
        "status": "running",
        "service": "Hybrid Privacy AI",
        "endpoints": ["/predict/plain", "/predict/private", "/predict/encrypted", "/epsilon"],
    }


@app.post("/predict/plain", response_model=PredictionResult)
def predict_plain_endpoint(data: PlainInput):
    """
    Baseline prediction with no privacy protection.
    The server sees everything — income, credit score, debt, years.
    Use this to compare against the private endpoints.
    """
    t0 = time.time()
    features = [data.income, data.credit_score, data.debt, data.employment_years]
    prob = predict_plain(model, scaler, features)
    latency = (time.time() - t0) * 1000

    return PredictionResult(
        probability=round(prob, 4),
        decision=make_decision(prob),
        privacy_applied=False,
        latency_ms=round(latency, 2),
    )


@app.post("/predict/private", response_model=PredictionResult)
def predict_private_endpoint(data: PlainInput):
    """
    Prediction with Differential Privacy applied to the output.
    Server still sees input values, but the returned probability
    has calibrated Laplace noise added.
    """
    t0 = time.time()
    features = [data.income, data.credit_score, data.debt, data.employment_years]
    raw_prob = predict_plain(model, scaler, features)
    noisy_prob = dp.privatize(raw_prob)
    noise = noisy_prob - raw_prob
    latency = (time.time() - t0) * 1000

    return PredictionResult(
        probability=round(noisy_prob, 4),
        decision=make_decision(noisy_prob),
        privacy_applied=True,
        noise_added=round(noise, 6),
        latency_ms=round(latency, 2),
    )


@app.post("/predict/encrypted", response_model=PredictionResult)
def predict_encrypted_endpoint(data: EncryptedInput):
  
    t0 = time.time()

    try:
        # Deserialize the encrypted vector sent by user
        import base64
        import tenseal as ts

        raw = base64.b64decode(data.ciphertext.encode())
        enc_vec = ts.lazy_ckks_vector_from(raw)
        enc_vec.link_context(he_context)

        # Perform HE dot product: encrypted_features · weights
        enc_result = enc_vec.dot(weights)

        # Decrypt only the logit (a single number — not the raw inputs)
        logit = decrypt_vector(enc_result)[0] + bias

        # Apply sigmoid to get probability
        raw_prob = sigmoid_approx(logit)

        # Apply differential privacy noise
        noisy_prob = dp.privatize(raw_prob)
        noise = noisy_prob - raw_prob

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Decryption/HE error: {str(e)}")

    latency = (time.time() - t0) * 1000

    return PredictionResult(
        probability=round(noisy_prob, 4),
        decision=make_decision(noisy_prob),
        privacy_applied=True,
        noise_added=round(noise, 6),
        latency_ms=round(latency, 2),
    )


@app.get("/epsilon")
def get_epsilon():
    """Return current privacy budget settings."""
    return dp.info()
