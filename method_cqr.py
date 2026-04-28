"""
Conformalized Quantile Regression (CQR) — Romano, Patterson, Candès (2019).

Split conformal prediction using Chronos's sample quantiles as the base
quantile estimator.

The nonconformity score is asymmetric:
    s(x, y) = max( q_lo(x) - y,  y - q_hi(x) )

where q_lo and q_hi are the α/2 and 1-α/2 empirical quantiles of Chronos
samples. Geometrically, s > 0 means y is outside [q_lo, q_hi]; s ≤ 0 means
y is inside.

At test time the interval is inflated by the (1-α)(1+1/N)-quantile of
calibration scores Q̂:
    C(x) = [ q_lo(x) - Q̂,  q_hi(x) + Q̂ ]

This gives a finite-sample marginal coverage guarantee:
    P( y ∈ C(x) ) ≥ 1 - α

Key differences from method_naive.py:
  - Calibration set is used to compute the correction Q̂.
  - Q̂ is a single scalar applied symmetrically to both interval edges
    (but the base interval itself is already asymmetric from Chronos samples).
  - Has a formal finite-sample coverage guarantee (unlike naive).

Key differences from method_pid.py:
  - Offline / split conformal — Q̂ is fixed after calibration, not adaptive.
  - Coverage guarantee is marginal (over random draw of cal+test), not sequential.
  - Does not require sequential observations; calibration and test are independent.

Usage:
    python method_cqr.py
"""

import numpy as np
import torch
from chronos import BaseChronosPipeline

from covid_dataset import load_hosp_wide, split_wide, to_tensor_list, make_all_windows

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_ID    = "amazon/chronos-t5-small"
CONTEXT_LEN = 60
PRED_LEN    = 14
STRIDE      = 7
NUM_SAMPLES = 100
BATCH_SIZE  = 32
ALPHAS      = [0.1, 0.2, 0.3]


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def get_quantiles(
    pipeline: BaseChronosPipeline,
    contexts: torch.Tensor,
    pred_len: int,
    alpha: float,
    num_samples: int = NUM_SAMPLES,
    batch_size: int = BATCH_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run Chronos and return the α/2 and 1-α/2 sample quantiles.

    Returns
    -------
    q_lo : (N, pred_len)
    q_hi : (N, pred_len)
    """
    q_lo_all, q_hi_all = [], []

    for start in range(0, contexts.shape[0], batch_size):
        batch = contexts[start : start + batch_size]
        samples = pipeline.predict(
            batch, prediction_length=pred_len, num_samples=num_samples
        )  # (B, num_samples, pred_len)

        q_lo_all.append(torch.quantile(samples, alpha / 2,     dim=1).numpy())
        q_hi_all.append(torch.quantile(samples, 1 - alpha / 2, dim=1).numpy())

    return np.concatenate(q_lo_all), np.concatenate(q_hi_all)


def cqr_scores(
    q_lo: np.ndarray,
    q_hi: np.ndarray,
    futures: np.ndarray,
) -> np.ndarray:
    """
    Compute CQR nonconformity scores.

    s = max( q_lo - y,  y - q_hi )

    Positive score  → y is outside [q_lo, q_hi]  (missed)
    Negative score  → y is inside  [q_lo, q_hi]  (covered)

    We take the max across the pred_len horizon so each window gets one
    scalar score: the worst violation across all forecast steps.

    Parameters
    ----------
    q_lo, q_hi : (N, pred_len)
    futures     : (N, pred_len)

    Returns
    -------
    scores : (N,)  one score per calibration window
    """
    per_step = np.maximum(q_lo - futures, futures - q_hi)  # (N, pred_len)
    return per_step.max(axis=1)                             # (N,)  worst step


def cqr_quantile(scores: np.ndarray, alpha: float) -> float:
    """
    Compute the conformal correction Q̂: the (1-α)(1+1/N) quantile of
    calibration scores, clipped to [scores.min(), ∞).

    This is the finite-sample-valid adjustment from Theorem 1 of Romano et al.
    """
    N = len(scores)
    level = np.ceil((1 - alpha) * (N + 1)) / N
    level = float(np.clip(level, 0.0, 1.0))
    return float(np.quantile(scores, level))


def apply_correction(
    q_lo: np.ndarray,
    q_hi: np.ndarray,
    Q_hat: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Inflate the base quantile interval by the conformal correction Q̂.

        lo = q_lo - Q̂
        hi = q_hi + Q̂

    Parameters
    ----------
    q_lo, q_hi : (N, pred_len)
    Q_hat      : scalar correction

    Returns
    -------
    lo, hi : (N, pred_len)
    """
    return q_lo - Q_hat, q_hi + Q_hat


def compute_metrics(
    lo: np.ndarray,
    hi: np.ndarray,
    futures: np.ndarray,
) -> dict:
    """
    Marginal coverage and interval width, both per-step and pooled.
    """
    covered = (futures >= lo) & (futures <= hi)
    return {
        "coverage":           covered.mean(),
        "width":              (hi - lo).mean(),
        "coverage_per_step":  covered.mean(axis=0),
        "width_per_step":     (hi - lo).mean(axis=0),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading data ...")
    wide = load_hosp_wide()
    train_wide, cal_wide, test_wide = split_wide(wide)

    cal_tensors  = to_tensor_list(cal_wide)
    test_tensors = to_tensor_list(test_wide)

    cal_ctx,  cal_fut  = make_all_windows(cal_tensors,  CONTEXT_LEN, PRED_LEN, stride=STRIDE)
    test_ctx, test_fut = make_all_windows(test_tensors, CONTEXT_LEN, PRED_LEN, stride=STRIDE)

    print(f"Calibration windows : {cal_ctx.shape[0]}")
    print(f"Test windows        : {test_ctx.shape[0]}")

    print(f"\nLoading Chronos model: {MODEL_ID} ...")
    pipeline = BaseChronosPipeline.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype=torch.float32,
    )

    print("\n--- CQR (Romano et al. 2019) ---")
    print(f"{'Alpha':>6}  {'Nominal':>8}  {'Coverage':>10}  {'Width':>10}  {'Q̂':>8}")
    print("-" * 50)

    for alpha in ALPHAS:
        # Step 1: get base quantile intervals on calibration set
        cal_lo, cal_hi = get_quantiles(pipeline, cal_ctx, PRED_LEN, alpha)

        # Step 2: compute nonconformity scores on calibration
        scores = cqr_scores(cal_lo, cal_hi, cal_fut.numpy())

        # Step 3: compute the conformal correction Q̂
        Q_hat = cqr_quantile(scores, alpha)

        # Step 4: get base quantile intervals on test set
        test_lo, test_hi = get_quantiles(pipeline, test_ctx, PRED_LEN, alpha)

        # Step 5: inflate test intervals by Q̂
        lo, hi = apply_correction(test_lo, test_hi, Q_hat)

        metrics = compute_metrics(lo, hi, test_fut.numpy())

        print(
            f"{alpha:>6.2f}  {1-alpha:>8.0%}  "
            f"{metrics['coverage']:>10.3f}  "
            f"{metrics['width']:>10.2f}  "
            f"{Q_hat:>8.2f}"
        )


if __name__ == "__main__":
    main()
