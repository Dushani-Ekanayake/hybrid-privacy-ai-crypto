"""
generate_data.py — Run this to regenerate sample_loan_data.csv

Usage:
    python generate_data.py
    python generate_data.py --n 1000
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../backend"))
from model import generate_sample_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=500, help="Number of records")
    args = parser.parse_args()

    df = generate_sample_data(n=args.n, save=True)
    print(f"\nApproval rate: {df['approved'].mean():.1%}")
    print(df.describe().round(1))
