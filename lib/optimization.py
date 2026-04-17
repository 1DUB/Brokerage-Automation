"""
Minimum-variance portfolio optimization.

Used by KDA to weight selected risky assets according to a covariance matrix.
The goal is to find weights w that minimize w' Σ w subject to:
  - Sum of weights = 1 (fully invested)
  - All weights >= 0 (long-only)

This replicates the behavior of R's portfolio.optim() function used in
Kipnis' original 2019 R implementation.
"""

import numpy as np
from scipy.optimize import minimize
from typing import Optional


def min_variance_weights(cov_matrix: np.ndarray) -> np.ndarray:
    """
    Compute long-only minimum-variance portfolio weights.
    
    Args:
        cov_matrix: NxN covariance matrix. Should be symmetric, positive 
                    semi-definite. May be slightly non-PSD due to numerical
                    issues; we'll regularize if needed.
    
    Returns:
        1D numpy array of N weights summing to 1, all >= 0.
    
    Raises:
        ValueError if cov_matrix is invalid (not square, etc.).
    """
    cov = np.asarray(cov_matrix, dtype=float)
    
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError(f"Expected square covariance matrix, got shape {cov.shape}")
    
    n = cov.shape[0]
    
    # Edge cases
    if n == 0:
        return np.array([])
    if n == 1:
        return np.array([1.0])
    
    # Regularize: ensure PSD by adding a tiny diagonal if needed.
    # This is a standard trick to make numerical optimization stable.
    try:
        eigenvalues = np.linalg.eigvalsh(cov)
        min_eig = eigenvalues.min()
    except np.linalg.LinAlgError:
        # Eigenvalue solver failed — covariance matrix is highly degenerate.
        # Force-regularize aggressively.
        min_eig = -1.0
    
    if min_eig < 1e-8:
        # Add enough to make smallest eigenvalue safely positive
        cov = cov + np.eye(n) * (abs(min_eig) + 1e-6)
    
    # Objective: minimize w' Σ w (portfolio variance)
    def objective(w):
        return float(w @ cov @ w)
    
    def gradient(w):
        return 2.0 * cov @ w
    
    # Constraints: weights sum to 1
    constraints = [{
        "type": "eq",
        "fun": lambda w: np.sum(w) - 1.0,
        "jac": lambda w: np.ones(n),
    }]
    
    # Bounds: 0 <= w_i <= 1 (long-only)
    bounds = [(0.0, 1.0) for _ in range(n)]
    
    # Initial guess: equal weights
    x0 = np.ones(n) / n
    
    result = minimize(
        objective,
        x0,
        jac=gradient,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-10, "maxiter": 200},
    )
    
    if not result.success:
        # Fall back to equal weights if optimization fails — better than crashing
        import logging
        logging.getLogger(__name__).warning(
            f"Min-variance optimization failed: {result.message}. "
            f"Falling back to equal weights."
        )
        return np.ones(n) / n
    
    weights = result.x
    
    # Clean up tiny negative weights from numerical noise
    weights = np.maximum(weights, 0.0)
    
    # Renormalize to sum to exactly 1
    total = weights.sum()
    if total > 0:
        weights = weights / total
    else:
        weights = np.ones(n) / n
    
    return weights


def build_covariance(
    correlation_matrix: np.ndarray,
    volatilities: np.ndarray,
) -> np.ndarray:
    """
    Build a covariance matrix from a correlation matrix and per-asset 
    volatilities.
    
    Cov[i,j] = Vol[i] * Vol[j] * Corr[i,j]
    
    This is the standard decomposition used in KDA per Kipnis (2019).
    
    Args:
        correlation_matrix: NxN correlation matrix.
        volatilities: 1D array of N volatility values.
    
    Returns:
        NxN covariance matrix.
    """
    corr = np.asarray(correlation_matrix, dtype=float)
    vols = np.asarray(volatilities, dtype=float)
    
    if corr.shape != (len(vols), len(vols)):
        raise ValueError(
            f"Shape mismatch: correlation {corr.shape} vs volatility ({len(vols)},)"
        )
    
    # Outer product gives Vol[i] * Vol[j] for all i,j
    vol_outer = np.outer(vols, vols)
    
    return vol_outer * corr
