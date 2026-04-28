"""
Conformal PID Control — Angelopoulos, Candès, Tibshirani (2023).
https://arxiv.org/abs/2307.16895

Online adaptive conformal prediction. A PI controller adjusts the effective
miscoverage rate α_t at every step based on running coverage error:

    err_t  = α - 1{y_t ∉ Ĉ_t}          # +α if covered, α-1 if not
    α_{t+1} = clamp(α_t + K_I * err_t, ε, 1-ε)

Intuition:
  - If we keep covering → err_t = α > 0 → α_t grows → intervals narrow.
  - If we miss       → err_t = α - 1 < 0 → α_t shrinks → intervals widen.
  - At steady state the coverage equals the nominal rate (1 - α).

This is fully model-agnostic: Chronos is just the base forecaster providing
a predictive distribution. The PID controller operates on coverage errors only.

Key differences from method_naive.py:
  - α_t is NOT fixed; it adapts online after every non-overlapping window.
  - Calibration set is used only to warm-start α_0 (a stable starting point).
  - Has a formal asymptotic coverage guarantee under mild mixing conditions.

Usage:
    python method_pid.py
"""

import numpy as np
import torch
from chronos import BaseChronosPipeline

from covid_dataset import (
    load_hosp_wide,
    split_wide,
    to_tensor_list,
    make_all_windows,
    CONTEXT_LEN,
    PRED_LEN,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_ID    = "amazon/chronos-t5-small"
CONTEXT_LEN = 60
PRED_LEN    = 14
NUM_SAMPLES = 100
BATCH_SIZE  = 32
ALPHAS      = [0.1, 0.2, 0.3]

# PI controller gains (from the paper; K_P=0 is pure integral / "I control")
K_I = 0.005
K_P = 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_samples_batched(
    pipeline: BaseChronosPipeline,
    contexts: torch.Tensor,
    pred_len: int,
    num_samples: int,
    batch_size: int,
) -> np.ndarray:
    """
    Run Chronos on all contexts and return raw samples.

    Returns
    -------
    samples : (N, num_samples, pred_len)
    """
    all_samples = []
    for start in range(0, contexts.shape[0], batch_size):
        batch = contexts[start : start + batch_size]
        s = pipeline.predict(batch, prediction_length=pred_len, num_samples=num_samples)
        all_samples.append(s.numpy())
    return np.concatenate(all_samples, axis=0)   # (N, num_samples, pred_len)


def samples_to_interval(samples: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert samples to prediction interval given a (possibly adapted) alpha.

    Parameters
    ----------
    samples : (N, num_samples, pred_len)  or  (num_samples, pred_len) for single window
    alpha   : scalar effective miscoverage rate at this step

    Returns
    -------
    lo, hi : each (N, pred_len) or (pred_len,)
    """
    lo = np.quantile(samples, alpha / 2,       axis=-2)
    hi = np.quantile(samples, 1 - alpha / 2,   axis=-2)
    return lo, hi


# ---------------------------------------------------------------------------
# Warm-start: initialise α_0 from calibration residuals
# ---------------------------------------------------------------------------

def warmstart_alpha(
    pipeline: BaseChronosPipeline,
    cal_tensors: list,
    train_wide,
    alpha: float,
) -> float:
    """
    Estimate a good starting α from the calibration set.

    We predict each calibration window (context drawn from *training* history
    + calibration prefix), observe the calibration future, and find the α_0
    such that the empirical coverage on calibration is exactly (1 - α).

    Returns
    -------
    alpha_0 : float  (same as alpha if calibration is skipped or errors occur)
    """
    cal_ctx, cal_fut = make_all_windows(cal_tensors, CONTEXT_LEN, PRED_LEN, stride=PRED_LEN)
    if cal_ctx.shape[0] == 0:
        return alpha

    samples = get_samples_batched(pipeline, cal_ctx, PRED_LEN, NUM_SAMPLES, BATCH_SIZE)
    # (N, num_samples, pred_len)

    # use median as point forecast; compute max absolute error across horizon
    median = np.median(samples, axis=1)              # (N, pred_len)
    errors = np.abs(cal_fut.numpy() - median).max(axis=1)  # (N,)  worst-case error per window

    # α_0 is the empirical miscoverage that gives (1-α) coverage on cal set
    # i.e. find q such that proportion of errors below q equals (1-alpha)
    # equivalently, the (1-alpha)-quantile of errors corresponds to q
    # and the effective alpha_0 is such that quantile(samples, alpha_0/2) achieves this
    # Simple approximation: use the same alpha (calibration effect is implicit in q_0 = quantile of errors)
    # For PID we just need a sensible starting q; we return the original alpha and
    # let the caller initialise q_0 from the error quantile.
    q0 = float(np.quantile(errors, 1 - alpha))
    return alpha, q0


# ---------------------------------------------------------------------------
# Online conformal PID over the test period (per state)
# ---------------------------------------------------------------------------

def run_pid_one_state(
    pipeline: BaseChronosPipeline,
    full_series: torch.Tensor,
    test_start_idx: int,
    alpha: float,
    q0: float,
    k_i: float = K_I,
    k_p: float = K_P,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run online conformal PI control for a single state time series.

    We process non-overlapping windows of length pred_len through the test
    period.  After each window we observe the truth, compute the max-horizon
    absolute error, and update the interval half-width q_t.

    The PI update on q (equivalent to updating alpha):
        err_t    = alpha - 1{max_error_t > q_t}     # negative if missed
        q_{t+1}  = q_t  -  k_i * err_t              # decrease q if covered, increase if not

    Note: operating on q directly (the half-width) is equivalent to operating
    on alpha_t in the paper when samples are fixed; we choose q-space because
    it's simpler to reason about interval widths.

    Parameters
    ----------
    full_series   : 1D tensor, the full time series for this state
    test_start_idx: index into full_series where the test period begins
    alpha         : nominal miscoverage rate
    q0            : initial half-width (warm-started from calibration)

    Returns
    -------
    lo_all  : (N_windows, pred_len)
    hi_all  : (N_windows, pred_len)
    covered : (N_windows,) bool  — whether all pred_len steps were inside the interval
    """
    T = len(full_series)
    lo_all, hi_all, covered_all = [], [], []

    q = q0          # current interval half-width
    t = test_start_idx

    while t + CONTEXT_LEN + PRED_LEN <= T:
        context = full_series[t : t + CONTEXT_LEN].unsqueeze(0)   # (1, context_len)
        future  = full_series[t + CONTEXT_LEN : t + CONTEXT_LEN + PRED_LEN].numpy()

        # Chronos samples → median point forecast
        samples = pipeline.predict(context, prediction_length=PRED_LEN, num_samples=NUM_SAMPLES)
        # samples: (1, num_samples, pred_len)
        median = np.median(samples[0].numpy(), axis=0)  # (pred_len,)

        # symmetric interval around the median
        lo = median - q
        hi = median + q

        lo_all.append(lo)
        hi_all.append(hi)

        # coverage check: all pred_len steps inside the interval
        covered = bool(np.all((future >= lo) & (future <= hi)))
        covered_all.append(covered)

        # PI update
        # err_t = alpha - 1{not covered}
        # if covered:   err_t = alpha      → q decreases (tighten)
        # if not covered: err_t = alpha-1  → q increases (loosen)
        err = alpha - (0 if covered else 1)
        q   = float(np.clip(q - k_i * err, a_min=0.0, a_max=None))

        t += PRED_LEN   # non-overlapping windows

    if not lo_all:
        return np.empty((0, PRED_LEN)), np.empty((0, PRED_LEN)), np.array([], dtype=bool)

    return np.stack(lo_all), np.stack(hi_all), np.array(covered_all)


# ---------------------------------------------------------------------------
# Run across all states
# ---------------------------------------------------------------------------

def run_pid(
    pipeline: BaseChronosPipeline,
    wide,
    train_wide,
    cal_wide,
    test_wide,
    alpha: float,
) -> dict:
    """
    Run conformal PID over all states and return aggregated metrics.
    """
    # Warm-start q0 from calibration set (use cal tensors with training prefix)
    cal_tensors = to_tensor_list(cal_wide)
    _, q0 = warmstart_alpha(pipeline, cal_tensors, train_wide, alpha)
    print(f"  α={alpha:.2f}  warm-start q0={q0:.2f}")

    # Figure out where the test period starts in the full (train+cal+test) series
    n_train_cal = len(train_wide) + len(cal_wide)

    all_lo, all_hi, all_covered = [], [], []

    states = wide.columns.tolist()
    for state in states:
        series = torch.tensor(wide[state].values.astype("float32"))

        lo, hi, covered = run_pid_one_state(
            pipeline, series, n_train_cal, alpha, q0
        )

        if len(covered) == 0:
            continue

        all_lo.append(lo)
        all_hi.append(hi)
        all_covered.append(covered)

    all_lo      = np.concatenate(all_lo)       # (N_total, pred_len)
    all_hi      = np.concatenate(all_hi)
    all_covered = np.concatenate(all_covered)  # (N_total,)

    return {
        "lo":       all_lo,
        "hi":       all_hi,
        "covered":  all_covered,
        "metrics": {
            "joint_coverage": all_covered.mean(),        # all horizons jointly inside
            "width":          (all_hi - all_lo).mean(),
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading data ...")
    wide = load_hosp_wide()
    train_wide, cal_wide, test_wide = split_wide(wide)

    print(f"\nLoading Chronos model: {MODEL_ID} ...")
    pipeline = BaseChronosPipeline.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype=torch.float32,
    )

    print("\n--- Conformal PID (Angelopoulos et al. 2023) ---")
    print(f"{'Alpha':>6}  {'Nominal':>8}  {'Coverage':>10}  {'Width':>10}")
    print("-" * 40)

    for alpha in ALPHAS:
        result = run_pid(pipeline, wide, train_wide, cal_wide, test_wide, alpha)
        m = result["metrics"]
        print(
            f"{alpha:>6.2f}  {1-alpha:>8.0%}  "
            f"{m['joint_coverage']:>10.3f}  "
            f"{m['width']:>10.2f}"
        )


if __name__ == "__main__":
    main()
