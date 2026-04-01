

import numpy as np
import pandas as pd
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/loan_model.joblib")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "../models/scaler.joblib")
DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/sample_loan_data.csv")


def generate_sample_data(n=500, save=True):
    """Generate synthetic loan approval data for demo purposes."""
    np.random.seed(42)

    income = np.random.normal(50000, 15000, n).clip(15000, 120000)
    credit_score = np.random.normal(680, 80, n).clip(300, 850)
    debt = np.random.normal(12000, 6000, n).clip(0, 60000)
    employment_years = np.random.exponential(5, n).clip(0, 30)

    # Simple approval rule with noise
    score = (
        (income / 100000) * 0.4
        + ((credit_score - 300) / 550) * 0.35
        - (debt / 60000) * 0.2
        + (employment_years / 30) * 0.05
    )
    noise = np.random.normal(0, 0.05, n)
    approved = (score + noise > 0.45).astype(int)

    df = pd.DataFrame({
        "income": income,
        "credit_score": credit_score,
        "debt": debt,
        "employment_years": employment_years,
        "approved": approved,
    })

    if save:
        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        df.to_csv(DATA_PATH, index=False)
        print(f"[data] Saved {n} records to {DATA_PATH}")

    return df


def train_model():
    """Train and save the logistic regression model."""
    # Load or generate data
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        print(f"[data] Loaded {len(df)} records from {DATA_PATH}")
    else:
        print("[data] No data found, generating synthetic dataset...")
        df = generate_sample_data()

    features = ["income", "credit_score", "debt", "employment_years"]
    X = df[features].values
    y = df["approved"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n[model] Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=["Rejected", "Approved"]))

    # Save
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"[model] Saved to {MODEL_PATH}")
    print(f"[model] Scaler saved to {SCALER_PATH}")

    return model, scaler


def load_model():
    """Load trained model and scaler."""
    if not os.path.exists(MODEL_PATH):
        print("[model] No saved model found, training now...")
        return train_model()
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


def predict_plain(model, scaler, features: list) -> float:
    """Standard prediction (no privacy). Returns approval probability."""
    x = np.array(features).reshape(1, -1)
    x_scaled = scaler.transform(x)
    prob = model.predict_proba(x_scaled)[0][1]
    return float(prob)


def get_model_weights(model, scaler):
    """
    Extract weights and bias from trained LR model.
    Used for performing inference manually on encrypted data.
    """
    # Weights in scaled space
    weights = model.coef_[0].tolist()
    bias = float(model.intercept_[0])
    return weights, bias


if __name__ == "__main__":
    train_model()
