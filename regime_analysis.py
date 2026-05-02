"""
Regime stratification (Experiment E2).

For a single α level, classify each test window by its recent dynamics
(falling / stable / rising) and report coverage and width per regime per method.

The thesis this supports:

  Plain CQR applies the same scalar correction Q̂ to every test point.
  Structure-aware methods (Codebook-CQR, Embed-CQR) adapt the effective
  correction based on what the model "knows" about the test context.
  We expect them to be tighter in *stable* regimes (where Chronos is
  confident and right) and to maintain coverage in *rising* / *falling*
  regimes (where surges and reversals make uncertainty larger).

Output
------
  regime_data.npz          — cached per-window (lo, hi, futures, regimes)
                              so reruns skip the expensive model passes.
  regime_table.csv         — coverage and width per (method, regime).
  regime_analysis.png      — coverage and width bar charts per regime.

Usage
-----
    python regime_analysis.py
    python regime_analysis.py --reset   # ignore cache and recompute
"""

import argparse
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from chronos import BaseChronosPipeline

from covid_dataset import load_hosp_wide, split_wide, to_tensor_list, make_all_windows
import method_naive
import method_cqr
import method_codebook_cqr
import method_embed_cqr
import evaluate

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ALPHA       = 0.10
MODEL_ID    = "amazon/chronos-t5-small"
CONTEXT_LEN = 60
PRED_LEN    = 14
STRIDE      = 7
NUM_SAMPLES = 100
BATCH_SIZE  = 32

# Slope threshold: normalized daily change. 0.02 = 2% per day.
SLOPE_THRESH = 0.02
LOOKBACK     = 14

CACHE_PATH = "regime_data.npz"
TABLE_PATH = "regime_table.csv"
PLOT_PATH  = "regime_analysis.png"

REGIMES_ORDER = ["falling", "stable", "rising"]


# ---------------------------------------------------------------------------
# Regime tagger
# ---------------------------------------------------------------------------

def tag_regimes(contexts: np.ndarray, lookback: int = LOOKBACK,
                thresh: float = SLOPE_THRESH) -> np.ndarray:
    """
    Classify each context window by the slope of its last `lookback` days,
    normalized by the recent mean (so the threshold is in % per day).

      rising   :  slope / mean >  thresh
      falling  :  slope / mean < -thresh
      stable   :  |slope / mean| ≤ thresh

    Parameters
    ----------
    contexts : (N, context_len)
    lookback : how many recent days to fit the slope on
    thresh   : normalized slope threshold (units: per day)

    Returns
    -------
    regimes : (N,) string array
    """
    N = contexts.shape[0]
    slopes = np.zeros(N)

    for i in range(N):
        recent = contexts[i, -lookback:]
        valid  = ~np.isnan(recent)
        if valid.sum() < 3:
            continue
        x = np.arange(lookback)[valid].astype(float)
        y = recent[valid]
        m = y.mean()
        if m > 0:
            slopes[i] = np.polyfit(x, y, 1)[0] / m

    return np.where(slopes < -thresh, "falling",
            np.where(slopes >  thresh, "rising", "stable"))


# ---------------------------------------------------------------------------
# Stratified metrics
# ---------------------------------------------------------------------------

def stratified_metrics(lo, hi, futures, regimes) -> list[dict]:
    rows = []
    for reg in REGIMES_ORDER:
        mask = (regimes == reg)
        if mask.sum() == 0:
            rows.append({"regime": reg, "n": 0,
                         "coverage": np.nan, "width": np.nan})
            continue
        covered = (futures[mask] >= lo[mask]) & (futures[mask] <= hi[mask])
        rows.append({
            "regime":   reg,
            "n":        int(mask.sum()),
            "coverage": float(covered.mean()),
            "width":    float((hi[mask] - lo[mask]).mean()),
        })
    return rows


# ---------------------------------------------------------------------------
# Data + model
# ---------------------------------------------------------------------------

def run_all_methods(pipeline, cal_ctx, cal_fut, test_ctx, test_fut,
                    cal_embs, test_embs):
    """
    Run all four methods at the configured ALPHA. Returns dict
    method_name -> (lo, hi).
    """
    out = {}

    print("  Naive ...")
    lo, hi, _ = evaluate.run_naive(pipeline, test_ctx, test_fut, ALPHA)
    out["Naive"] = (lo, hi)

    print("  CQR ...")
    lo, hi, _ = evaluate.run_cqr(pipeline, cal_ctx, cal_fut, test_ctx, test_fut, ALPHA)
    out["CQR"] = (lo, hi)

    print("  Codebook-CQR ...")
    lo, hi, _ = evaluate.run_codebook_cqr(pipeline, cal_ctx, cal_fut, test_ctx, test_fut, ALPHA)
    out["Codebook-CQR"] = (lo, hi)

    print("  Embed-CQR ...")
    lo, hi, _ = evaluate.run_embed_cqr(
        pipeline, cal_ctx, cal_fut, test_ctx, test_fut,
        cal_embs, test_embs, ALPHA
    )
    out["Embed-CQR"] = (lo, hi)

    return out


def load_or_run(reset: bool):
    """Load cached per-window arrays, or recompute if cache missing or --reset."""
    if not reset and os.path.exists(CACHE_PATH):
        print(f"Loading cache: {CACHE_PATH}")
        data = np.load(CACHE_PATH, allow_pickle=True)
        results = {}
        for k in data.files:
            if k.endswith("_lo"):
                m = k[:-3]
                results[m] = (data[f"{m}_lo"], data[f"{m}_hi"])
        return results, data["futures"], data["regimes"]

    print("Loading data ...")
    wide = load_hosp_wide()
    _, cal_wide, test_wide = split_wide(wide)

    cal_tensors  = to_tensor_list(cal_wide)
    test_tensors = to_tensor_list(test_wide)
    cal_ctx,  cal_fut  = make_all_windows(cal_tensors,  CONTEXT_LEN, PRED_LEN, stride=STRIDE)
    test_ctx, test_fut = make_all_windows(test_tensors, CONTEXT_LEN, PRED_LEN, stride=STRIDE)
    print(f"Test windows: {test_ctx.shape[0]}")

    regimes = tag_regimes(test_ctx.numpy())
    counts  = pd.Series(regimes).value_counts().to_dict()
    print(f"Regime counts: {counts}")

    print(f"\nLoading Chronos: {MODEL_ID}")
    pipeline = BaseChronosPipeline.from_pretrained(
        MODEL_ID, device_map="auto", torch_dtype=torch.float32,
    )

    print("Computing embeddings ...")
    cal_embs  = method_embed_cqr.get_embeddings(pipeline, cal_ctx)
    test_embs = method_embed_cqr.get_embeddings(pipeline, test_ctx)

    print("Running methods ...")
    results = run_all_methods(pipeline, cal_ctx, cal_fut, test_ctx, test_fut,
                              cal_embs, test_embs)
    fut_np = test_fut.numpy()

    save_kwargs = {"futures": fut_np, "regimes": regimes}
    for m, (lo, hi) in results.items():
        save_kwargs[f"{m}_lo"] = lo
        save_kwargs[f"{m}_hi"] = hi
    np.savez(CACHE_PATH, **save_kwargs)
    print(f"Cached to {CACHE_PATH}")

    return results, fut_np, regimes


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_regime_analysis(df: pd.DataFrame):
    methods = df["method"].unique().tolist()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    x       = np.arange(len(REGIMES_ORDER))
    n_meth  = len(methods)
    bar_w   = 0.8 / n_meth

    colors = {
        "Naive":        "steelblue",
        "CQR":          "darkorange",
        "Codebook-CQR": "crimson",
        "Embed-CQR":    "seagreen",
    }

    for i, m in enumerate(methods):
        sub = df[df["method"] == m].set_index("regime").reindex(REGIMES_ORDER)
        offset = (i - (n_meth - 1) / 2) * bar_w
        axes[0].bar(x + offset, sub["coverage"], bar_w,
                    label=m, color=colors.get(m, None))
        axes[1].bar(x + offset, sub["width"],    bar_w,
                    label=m, color=colors.get(m, None))

    axes[0].axhline(1 - ALPHA, color="k", ls="--", lw=1,
                    label=f"Nominal {1-ALPHA:.0%}")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(REGIMES_ORDER)
    axes[0].set_ylabel("Empirical coverage")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title(f"Coverage by regime (α = {ALPHA})")
    axes[0].legend(loc="lower right", fontsize=9)

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(REGIMES_ORDER)
    axes[1].set_ylabel("Mean interval width (admissions)")
    axes[1].set_title(f"Width by regime (α = {ALPHA})")
    axes[1].legend(loc="upper left", fontsize=9)

    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=150)
    plt.close(fig)
    print(f"Saved {PLOT_PATH}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(reset: bool = False):
    results, futures, regimes = load_or_run(reset=reset)

    rows = []
    for method, (lo, hi) in results.items():
        for stat in stratified_metrics(lo, hi, futures, regimes):
            rows.append({"method": method, **stat})

    df = pd.DataFrame(rows)
    df.to_csv(TABLE_PATH, index=False)

    print("\n=== Stratified results ===")
    print(df.to_string(index=False))

    plot_regime_analysis(df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true",
                        help="Ignore cache and recompute everything")
    args = parser.parse_args()
    main(reset=args.reset)
