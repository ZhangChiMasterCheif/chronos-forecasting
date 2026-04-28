"""
Naive Chronos Sampling — empirical quantile intervals from sample paths.

This is the baseline method from the Chronos paper: generate N sample
trajectories, then take the α/2 and 1-α/2 empirical quantiles as the
prediction interval. No calibration, no conformal correction.

The key limitation: there is no coverage guarantee. If Chronos's
predictive distribution is miscalibrated (too narrow or too wide),
the empirical coverage will drift away from the nominal level.

Usage:
    python method_naive.py
"""

import torch
import numpy as np
from chronos import BaseChronosPipeline

from covid_dataset import load_hosp_wide, split_wide, to_tensor_list, make_all_windows

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_ID    = "amazon/chronos-t5-small"
CONTEXT_LEN = 60    # days of history fed as input
PRED_LEN    = 14    # days ahead to forecast
STRIDE      = 7     # days between consecutive windows (weekly)
NUM_SAMPLES = 100   # number of Chronos sample paths per prediction
BATCH_SIZE  = 32    # contexts processed at once (tune to your GPU memory)
ALPHAS      = [0.1, 0.2, 0.3]   # target miscoverage rates → 90%, 80%, 70% intervals


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def predict_intervals(
    pipeline: BaseChronosPipeline,
    contexts: torch.Tensor,
    pred_len: int,
    alpha: float,
    num_samples: int = NUM_SAMPLES,
    batch_size: int = BATCH_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run Chronos on a batch of context windows and return empirical quantile
    prediction intervals.

    Parameters
    ----------
    pipeline  : loaded ChronosPipeline
    contexts  : (N, context_len) float tensor — one row per prediction task
    pred_len  : number of steps to forecast
    alpha     : miscoverage rate (e.g. 0.1 → 90% interval)
    num_samples: number of sample paths Chronos generates per context
    batch_size : how many contexts to push through the model at once

    Returns
    -------
    lo : (N, pred_len) lower bound of prediction interval
    hi : (N, pred_len) upper bound of prediction interval
    """
    N = contexts.shape[0]
    lo_all, hi_all = [], []

    for start in range(0, N, batch_size):
        batch = contexts[start : start + batch_size]   # (B, context_len)

        # pipeline.predict returns (B, num_samples, pred_len) in real-value space
        samples = pipeline.predict(
            batch,
            prediction_length=pred_len,
            num_samples=num_samples,
        )  # (B, num_samples, pred_len)

        # empirical quantiles across the sample dimension
        lo = torch.quantile(samples, alpha / 2,       dim=1).numpy()  # (B, pred_len)
        hi = torch.quantile(samples, 1 - alpha / 2,   dim=1).numpy()  # (B, pred_len)

        lo_all.append(lo)
        hi_all.append(hi)

    return np.concatenate(lo_all), np.concatenate(hi_all)


def compute_metrics(
    lo: np.ndarray,
    hi: np.ndarray,
    futures: np.ndarray,
) -> dict:
    """
    Compute coverage and interval width.

    Parameters
    ----------
    lo      : (N, pred_len) lower bounds
    hi      : (N, pred_len) upper bounds
    futures : (N, pred_len) ground truth values

    Returns
    -------
    dict with:
      coverage      — fraction of true values inside [lo, hi] (scalar, pooled over N and pred_len)
      width         — mean interval width hi - lo (scalar)
      coverage_per_step — (pred_len,) coverage at each forecast horizon
      width_per_step    — (pred_len,) mean width at each forecast horizon
    """
    covered = (futures >= lo) & (futures <= hi)   # (N, pred_len) bool

    return {
        "coverage":           covered.mean(),
        "width":              (hi - lo).mean(),
        "coverage_per_step":  covered.mean(axis=0),   # (pred_len,)
        "width_per_step":     (hi - lo).mean(axis=0), # (pred_len,)
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading data ...")
    wide = load_hosp_wide()
    _, _, test_wide = split_wide(wide)

    # Use the full training history as context prefix for each test window.
    # We concatenate train+cal so the model sees as much history as possible,
    # then slide windows over the test period only.
    _, cal_wide, test_wide = split_wide(wide)

    test_tensors = to_tensor_list(test_wide)
    test_ctx, test_fut = make_all_windows(test_tensors, CONTEXT_LEN, PRED_LEN, stride=STRIDE)
    print(f"Test windows: {test_ctx.shape[0]}  (contexts {tuple(test_ctx.shape)}, futures {tuple(test_fut.shape)})")

    print(f"\nLoading Chronos model: {MODEL_ID} ...")
    pipeline = BaseChronosPipeline.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype=torch.float32,
    )

    print("\n--- Naive Chronos Sampling ---")
    print(f"{'Alpha':>6}  {'Nominal':>8}  {'Coverage':>10}  {'Width':>10}")
    print("-" * 40)

    results = {}
    for alpha in ALPHAS:
        lo, hi = predict_intervals(pipeline, test_ctx, PRED_LEN, alpha)
        metrics = compute_metrics(lo, hi, test_fut.numpy())

        print(
            f"{alpha:>6.2f}  {1-alpha:>8.0%}  "
            f"{metrics['coverage']:>10.3f}  "
            f"{metrics['width']:>10.2f}"
        )
        results[alpha] = {"lo": lo, "hi": hi, "metrics": metrics}

    return results


if __name__ == "__main__":
    main()
