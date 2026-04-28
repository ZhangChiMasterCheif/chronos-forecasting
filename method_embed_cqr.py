"""
Embedding-Weighted CQR — locally adaptive conformal prediction using
Chronos encoder embeddings.

Motivation
----------
Standard CQR (method_cqr.py) applies a single scalar correction Q̂ uniformly
to every test point. But not all test contexts are equally hard: a context
during a stable endemic plateau is easier to forecast than one during a surge.

This method makes Q̂ test-specific by weighting calibration scores by how
similar each calibration context is to the test context *in Chronos's learned
embedding space*.  Calibration windows that "look like" the test window
contribute more to Q̂.

Algorithm
---------
1. Embed all calibration contexts: e_i = mean_pool(Encoder(x_i))  →  (N_cal, d)
2. Embed all test contexts:        e_j = mean_pool(Encoder(x_j))  →  (N_test, d)
3. For each test point j:
     w_i = exp(−γ · ‖e_j − e_i‖²)          RBF kernel weights
     Q̂_j = weighted_quantile(scores, w, α)  test-specific correction
     C_j  = [q_lo_j − Q̂_j,  q_hi_j + Q̂_j]
4. Evaluate coverage and width on C_j vs true futures.

The bandwidth γ is set via the median heuristic:
     γ = 1 / (2 · median(pairwise distances among calibration embeddings)²)

Theoretical grounding
---------------------
Weighted conformal prediction gives valid coverage under covariate shift when
the weights are proportional to the likelihood ratio dP_test/dP_cal.
Here we approximate that ratio with a kernel — a standard non-parametric choice.
Reference: Tibshirani et al. (2019) "Conformal Prediction Under Covariate Shift".

Note: encoder embeddings are only available for seq2seq (T5-based) Chronos
models, not the causal (GPT2) variant. Use chronos-t5-* checkpoints.

Usage:
    python method_embed_cqr.py
"""

import numpy as np
import torch
from chronos import BaseChronosPipeline

from covid_dataset import load_hosp_wide, split_wide, to_tensor_list, make_all_windows
from method_cqr import get_quantiles, cqr_scores, apply_correction, compute_metrics

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
# Embedding
# ---------------------------------------------------------------------------

def get_embeddings(
    pipeline: BaseChronosPipeline,
    contexts: torch.Tensor,
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    """
    Run the Chronos T5 encoder on all context windows and return mean-pooled
    embeddings.

    pipeline.embed(context) returns:
      embeddings : (B, context_len, d_model)  — one vector per time step
      tokenizer_state : the scale used for normalization (not needed here)

    We mean-pool across the time dimension to get one fixed-size vector per
    context window: (B, d_model).

    Parameters
    ----------
    contexts : (N, context_len)

    Returns
    -------
    embeddings : (N, d_model)  float32 numpy array
    """
    all_embs = []
    for start in range(0, contexts.shape[0], batch_size):
        batch = contexts[start : start + batch_size]
        # embed() returns (embeddings, tokenizer_state)
        embs, _ = pipeline.embed(batch)          # (B, context_len, d_model)
        pooled  = embs.mean(dim=1)               # (B, d_model)  mean over time
        all_embs.append(pooled.numpy())
    return np.concatenate(all_embs, axis=0)      # (N, d_model)


# ---------------------------------------------------------------------------
# Kernel and weighted quantile
# ---------------------------------------------------------------------------

def median_heuristic_gamma(cal_embs: np.ndarray) -> float:
    """
    Set the RBF bandwidth via the median pairwise distance heuristic.

    Subsample at most 500 calibration embeddings to keep computation O(500²).
    γ = 1 / (2 · median_distance²)
    """
    idx = np.random.choice(len(cal_embs), size=min(500, len(cal_embs)), replace=False)
    sub = cal_embs[idx]                          # (M, d)
    # pairwise squared distances
    diff = sub[:, None, :] - sub[None, :, :]     # (M, M, d)
    sq_dists = (diff ** 2).sum(axis=-1)          # (M, M)
    median_sq = float(np.median(sq_dists[sq_dists > 0]))
    if median_sq == 0:
        return 1.0
    return 1.0 / (2.0 * median_sq)


def rbf_weights(
    test_emb: np.ndarray,
    cal_embs: np.ndarray,
    gamma: float,
) -> np.ndarray:
    """
    Compute RBF kernel weights between one test embedding and all calibration
    embeddings.

    w_i = exp(−γ · ‖e_test − e_i‖²)

    Parameters
    ----------
    test_emb : (d,)
    cal_embs : (N, d)
    gamma    : bandwidth

    Returns
    -------
    weights : (N,)  positive, NOT normalized (normalization happens in weighted_quantile)
    """
    diff    = cal_embs - test_emb[None, :]       # (N, d)
    sq_dist = (diff ** 2).sum(axis=1)            # (N,)
    return np.exp(-gamma * sq_dist)


def weighted_quantile(
    scores: np.ndarray,
    weights: np.ndarray,
    alpha: float,
) -> float:
    """
    Weighted (1-α)-quantile of scores.

    Equivalent to: find the smallest score s* such that the total weight of
    calibration points with score ≤ s* is at least (1-α).

    We also add a "ghost" point at +∞ with weight 1/(N+1) following
    Tibshirani et al. (2019) for finite-sample validity.

    Parameters
    ----------
    scores  : (N,)
    weights : (N,)  positive, any scale
    alpha   : miscoverage rate

    Returns
    -------
    Q_hat : float
    """
    N = len(scores)
    # normalize weights, add ghost point at +inf
    w = weights / weights.sum()
    ghost_w = 1.0 / (N + 1)
    w = w * (N / (N + 1))   # rescale so ghost makes total = 1

    # sort calibration scores ascending
    order        = np.argsort(scores)
    sorted_s     = scores[order]
    sorted_w     = w[order]
    cum_w        = np.cumsum(sorted_w)

    # find smallest score where cumulative weight ≥ (1-alpha)
    # ghost point handles the case where (1-alpha) > cum_w[-1]
    target = 1.0 - alpha
    idx = np.searchsorted(cum_w, target)
    if idx >= N:
        return float("inf")   # ghost point: interval covers everything
    return float(sorted_s[idx])


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
        MODEL_ID,
        device_map="auto",
        torch_dtype=torch.float32,
    )

    # --- Compute embeddings (done once, shared across all alpha levels) ---
    print("\nComputing calibration embeddings ...")
    cal_embs = get_embeddings(pipeline, cal_ctx)    # (N_cal, d_model)

    print("Computing test embeddings ...")
    test_embs = get_embeddings(pipeline, test_ctx)  # (N_test, d_model)

    print(f"Embedding dim: {cal_embs.shape[1]}")

    gamma = median_heuristic_gamma(cal_embs)
    print(f"RBF bandwidth γ = {gamma:.6f}")

    print("\n--- Embedding-Weighted CQR ---")
    print(f"{'Alpha':>6}  {'Nominal':>8}  {'Coverage':>10}  {'Width':>10}  {'Q̂ mean':>10}")
    print("-" * 55)

    for alpha in ALPHAS:
        # Step 1: base quantile intervals on calibration set
        cal_lo, cal_hi = get_quantiles(pipeline, cal_ctx, PRED_LEN, alpha)

        # Step 2: nonconformity scores on calibration
        scores = cqr_scores(cal_lo, cal_hi, cal_fut.numpy())   # (N_cal,)

        # Step 3: base quantile intervals on test set
        test_lo, test_hi = get_quantiles(pipeline, test_ctx, PRED_LEN, alpha)

        # Step 4: for each test point, compute its own Q̂ using similarity weights
        N_test  = test_ctx.shape[0]
        lo_all  = np.empty_like(test_lo)
        hi_all  = np.empty_like(test_hi)
        q_hats  = np.empty(N_test)

        for j in range(N_test):
            w      = rbf_weights(test_embs[j], cal_embs, gamma)   # (N_cal,)
            Q_hat  = weighted_quantile(scores, w, alpha)
            lo_all[j], hi_all[j] = apply_correction(
                test_lo[j:j+1], test_hi[j:j+1], Q_hat
            )
            q_hats[j] = Q_hat

        metrics = compute_metrics(lo_all, hi_all, test_fut.numpy())

        print(
            f"{alpha:>6.2f}  {1-alpha:>8.0%}  "
            f"{metrics['coverage']:>10.3f}  "
            f"{metrics['width']:>10.2f}  "
            f"{q_hats.mean():>10.2f}"
        )


if __name__ == "__main__":
    main()
