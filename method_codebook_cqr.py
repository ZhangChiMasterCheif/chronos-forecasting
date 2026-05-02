"""
Codebook-Distance Conformalized Quantile Regression (Codebook-CQR).

Same conformal procedure as method_cqr.py, but the nonconformity score is
measured in *codebook bin-index space* rather than in real-value space.

Score
-----
    s = max over horizon of  max( bin_lo − bin_true,  bin_true − bin_hi )

where bin_lo, bin_hi are the α/2 and 1−α/2 quantiles of *sample bin IDs*,
obtained by re-tokenizing Chronos samples through the same MeanScaleUniformBins
tokenizer that the model itself uses internally.

Why bin-space?
  - Bounded — score is at most ≤ vocab size, intervals always finite.
  - Scale-invariant — Chronos already mean-abs-normalizes values into the
    codebook, so the metric is comparable across series of different scales.
  - Coarsened — robust to small numerical noise in residuals.
  - Aligned with the model — distance in bin space is what the LM actually
    "sees", since it operates on token IDs end-to-end.

Coverage guarantee: same finite-sample marginal coverage as residual CQR
(Romano et al. 2019), since the procedure is identical except for the metric.

Usage:
    python method_codebook_cqr.py
"""

import numpy as np
import torch
from chronos import BaseChronosPipeline

from covid_dataset import load_hosp_wide, split_wide, to_tensor_list, make_all_windows
from method_cqr import cqr_quantile, compute_metrics

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
# Tokenization helpers
# ---------------------------------------------------------------------------

def get_sample_bins(
    pipeline: BaseChronosPipeline,
    contexts: torch.Tensor,
    pred_len: int,
    alpha: float,
    num_samples: int = NUM_SAMPLES,
    batch_size: int = BATCH_SIZE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run Chronos and return α/2 and 1−α/2 quantiles in *bin-index space*.

    Procedure:
      1. tokenizer.context_input_transform(context) → scale (per series)
      2. pipeline.predict(context) → samples in real-value space (B, NS, T)
      3. tokenizer._input_transform(samples, scale) → bin IDs (B, NS, T)
      4. quantile across the sample dim → bin_lo, bin_hi (B, T)

    Returns
    -------
    bin_lo, bin_hi : (N, pred_len) float arrays of bin IDs
    scale          : (N,)         per-series scale used for tokenization
    """
    tokenizer = pipeline.tokenizer

    bin_lo_all, bin_hi_all, scale_all = [], [], []

    for start in range(0, contexts.shape[0], batch_size):
        batch = contexts[start : start + batch_size]

        # 1. per-series scale from the context tokenization
        _, _, scale = tokenizer.context_input_transform(batch)        # (B,)

        # 2. samples in real-value space
        samples = pipeline.predict(
            batch, prediction_length=pred_len, num_samples=num_samples
        )                                                              # (B, NS, T)

        # 3. re-tokenize each sample with the same scale
        B, NS, T = samples.shape
        samples_flat = samples.reshape(B * NS, T)                       # (B*NS, T)
        scale_rep    = scale.unsqueeze(1).expand(-1, NS).reshape(-1)    # (B*NS,)

        sample_bins, _, _ = tokenizer._input_transform(samples_flat, scale=scale_rep)
        sample_bins = sample_bins.reshape(B, NS, T).float()             # (B, NS, T)

        # 4. quantile across samples
        bin_lo = torch.quantile(sample_bins, alpha / 2,     dim=1)      # (B, T)
        bin_hi = torch.quantile(sample_bins, 1 - alpha / 2, dim=1)      # (B, T)

        bin_lo_all.append(bin_lo.numpy())
        bin_hi_all.append(bin_hi.numpy())
        scale_all.append(scale.numpy())

    return (
        np.concatenate(bin_lo_all),
        np.concatenate(bin_hi_all),
        np.concatenate(scale_all),
    )


def get_true_bins(
    pipeline: BaseChronosPipeline,
    futures: torch.Tensor,
    scale: np.ndarray,
) -> np.ndarray:
    """
    Tokenize the true futures using the per-window scale from the corresponding
    context. Returns integer bin IDs of shape (N, pred_len).
    """
    futures_t = torch.as_tensor(futures, dtype=torch.float32)
    scale_t   = torch.as_tensor(scale,   dtype=torch.float32)
    bin_ids, _, _ = pipeline.tokenizer._input_transform(futures_t, scale=scale_t)
    return bin_ids.numpy()


# ---------------------------------------------------------------------------
# Conformal procedure in bin-space
# ---------------------------------------------------------------------------

def codebook_scores(
    bin_lo: np.ndarray,
    bin_hi: np.ndarray,
    true_bins: np.ndarray,
) -> np.ndarray:
    """
    CQR-style score in bin-index space:
        s_i = max_t  max( bin_lo[i,t] − bin_true[i,t],  bin_true[i,t] − bin_hi[i,t] )

    Positive → at least one horizon step had the true bin outside [bin_lo, bin_hi].
    Negative → every horizon step had the true bin inside.
    Returns (N,) one score per window.
    """
    per_step = np.maximum(bin_lo - true_bins, true_bins - bin_hi)
    return per_step.max(axis=1)


def apply_correction(
    bin_lo: np.ndarray,
    bin_hi: np.ndarray,
    Q_hat: float,
    n_tokens: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Inflate bin interval by Q̂ and clamp to the valid bin range [0, n_tokens-1].

    Boundedness of the codebook means the interval is always finite — even
    without clamping, the maximum sensible bin is n_tokens − 1.
    """
    lo = np.clip(bin_lo - Q_hat, 0, n_tokens - 1)
    hi = np.clip(bin_hi + Q_hat, 0, n_tokens - 1)
    return lo, hi


def bins_to_values(
    pipeline: BaseChronosPipeline,
    bin_lo: np.ndarray,
    bin_hi: np.ndarray,
    scale: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert bin-index intervals back to real-value intervals via the codebook
    centers and the per-series scale.

    Uses tokenizer.output_transform, which expects (B, num_samples, T) and
    decrements bin IDs by (n_special_tokens + 1) before indexing into centers.
    We unsqueeze a dummy num_samples=1 dim and squeeze it back.
    """
    tokenizer = pipeline.tokenizer
    bin_lo_t  = torch.as_tensor(bin_lo).unsqueeze(1).long()         # (N, 1, T)
    bin_hi_t  = torch.as_tensor(bin_hi).unsqueeze(1).long()         # (N, 1, T)
    scale_t   = torch.as_tensor(scale, dtype=torch.float32)         # (N,)

    value_lo = tokenizer.output_transform(bin_lo_t, scale_t).squeeze(1).numpy()
    value_hi = tokenizer.output_transform(bin_hi_t, scale_t).squeeze(1).numpy()
    return value_lo, value_hi


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading data ...")
    wide = load_hosp_wide()
    _, cal_wide, test_wide = split_wide(wide)

    cal_tensors  = to_tensor_list(cal_wide)
    test_tensors = to_tensor_list(test_wide)
    cal_ctx,  cal_fut  = make_all_windows(cal_tensors,  CONTEXT_LEN, PRED_LEN, stride=STRIDE)
    test_ctx, test_fut = make_all_windows(test_tensors, CONTEXT_LEN, PRED_LEN, stride=STRIDE)

    print(f"Calibration windows : {cal_ctx.shape[0]}")
    print(f"Test windows        : {test_ctx.shape[0]}")

    print(f"\nLoading Chronos model: {MODEL_ID} ...")
    pipeline = BaseChronosPipeline.from_pretrained(
        MODEL_ID, device_map="auto", torch_dtype=torch.float32,
    )

    n_tokens = pipeline.tokenizer.config.n_tokens
    print(f"Codebook size: {n_tokens} bins  ({n_tokens - pipeline.tokenizer.config.n_special_tokens - 1} usable)")

    print("\n--- Codebook-CQR ---")
    print(f"{'Alpha':>6}  {'Nominal':>8}  {'Coverage':>10}  {'Width':>10}  {'Q̂(bins)':>10}")
    print("-" * 50)

    for alpha in ALPHAS:
        # Calibration: bin-space quantiles + true bins
        cal_bin_lo, cal_bin_hi, cal_scale = get_sample_bins(pipeline, cal_ctx, PRED_LEN, alpha)
        cal_true_bins                      = get_true_bins(pipeline, cal_fut, cal_scale)

        scores = codebook_scores(cal_bin_lo, cal_bin_hi, cal_true_bins)
        Q_hat  = cqr_quantile(scores, alpha)

        # Test: same bin-space, then convert back to values
        test_bin_lo, test_bin_hi, test_scale = get_sample_bins(pipeline, test_ctx, PRED_LEN, alpha)
        lo_bins, hi_bins = apply_correction(test_bin_lo, test_bin_hi, Q_hat, n_tokens)
        lo, hi           = bins_to_values(pipeline, lo_bins, hi_bins, test_scale)

        metrics = compute_metrics(lo, hi, test_fut.numpy())

        print(
            f"{alpha:>6.2f}  {1-alpha:>8.0%}  "
            f"{metrics['coverage']:>10.3f}  "
            f"{metrics['width']:>10.2f}  "
            f"{Q_hat:>10.1f}"
        )


if __name__ == "__main__":
    main()
