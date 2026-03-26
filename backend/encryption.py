"""
encryption.py — Homomorphic Encryption wrapper using TenSEAL

We use the CKKS scheme (supports real numbers, not just integers).
The key idea: the user encrypts their data locally, sends ciphertext
to the server, the server runs computations, and returns an encrypted result.
The server never sees the raw values at any point.

TenSEAL wraps Microsoft SEAL under the hood.
"""

import tenseal as ts
import numpy as np
import base64


def create_context() -> ts.Context:
    """
    Create a TenSEAL CKKS context.
    
    poly_modulus_degree: controls capacity (higher = more operations but slower).
    coeff_mod_bit_sizes: precision of the encryption.
    global_scale: how precisely real numbers are encoded.
    """
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=4096,
        coeff_mod_bit_sizes=[40, 20, 40],
    )
    context.generate_galois_keys()
    context.global_scale = 2 ** 20
    return context


def encrypt_vector(context: ts.Context, values: list) -> ts.CKKSVector:
    """Encrypt a list of floats into a CKKS ciphertext vector."""
    return ts.ckks_vector(context, values)


def decrypt_vector(encrypted: ts.CKKSVector) -> list:
    """Decrypt a CKKS vector back to floats."""
    return encrypted.decrypt()


def serialize_encrypted(encrypted: ts.CKKSVector) -> str:
    """Serialize ciphertext to base64 string for API transport."""
    raw = encrypted.serialize()
    return base64.b64encode(raw).decode("utf-8")


def deserialize_encrypted(context: ts.Context, data: str) -> ts.CKKSVector:
    """Deserialize ciphertext from base64 string."""
    raw = base64.b64decode(data.encode("utf-8"))
    return ts.lazy_ckks_vector_from(raw)


def he_dot_product(encrypted_vec: ts.CKKSVector, weights: list) -> ts.CKKSVector:
    """
    Perform a dot product on encrypted data with plaintext weights.
    This is the core HE operation used by our linear model layer.
    """
    return encrypted_vec.dot(weights)


def he_add_bias(encrypted_result: ts.CKKSVector, bias: float) -> ts.CKKSVector:
    """Add plaintext bias to encrypted result."""
    return encrypted_result + [bias]


def sigmoid_approx(x: float) -> float:
    """
    Polynomial approximation of sigmoid for encrypted domain.
    Used after decryption when full HE sigmoid isn't available.
    Real HE sigmoid requires degree-7+ polynomials — overkill for this demo.
    """
    return 1.0 / (1.0 + np.exp(-x))
