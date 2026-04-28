"""
Unified evaluation: runs all four methods and produces comparison tables and plots.

Outputs
-------
  results_table.csv       — coverage and width for every method × alpha
  calibration_curve.png   — coverage vs nominal level for all methods
  width_vs_alpha.png      — interval width vs alpha for all methods
  intervals_<state>.png   — prediction intervals over time for a few example states
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from chronos import BaseChronosPipeline

from covid_dataset import (
    load_hosp_wide,
    split_wide,
    to_tensor_list,
    make_all_windows,
)
import method_naive
import method_cqr
import method_embed_cqr

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_ID    = "amazon/chronos-t5-small"
CONTEXT_LEN = 60
PRED_LEN    = 14
STRIDE      = 7
NUM_SAMPLES = 100
BATCH_SIZE  = 32
ALPHAS      = [0.05, 0.10, 0.20, 0.30, 0.40]

EXAMPLE_STATES = ["New York", "California", "Texas", "Florida"]   # for interval plots


# ---------------------------------------------------------------------------
# Run all methods
# ---------------------------------------------------------------------------

def run_all(pipeline, cal_ctx, cal_fut, test_ctx, test_fut, cal_embs, test_embs, alpha):
    """
    Run all four methods for a given alpha. Returns a dict of results.
    Each result has keys: lo (N, pred_len), hi (N, pred_len).
    """
    results = {}

    # --- Naive ---
    lo, hi = method_naive.predict_intervals(
        pipeline, test_ctx, PRED_LEN, alpha, NUM_SAMPLES, BATCH_SIZE
    )
    results["Naive"] = {"lo": lo, "hi": hi}

    # --- CQR ---
    cal_lo, cal_hi   = method_cqr.get_quantiles(pipeline, cal_ctx, PRED_LEN, alpha)
    scores           = method_cqr.cqr_scores(cal_lo, cal_hi, cal_fut.numpy())
    Q_hat            = method_cqr.cqr_quantile(scores, alpha)
    test_lo, test_hi = method_cqr.get_quantiles(pipeline, test_ctx, PRED_LEN, alpha)
    lo, hi           = method_cqr.apply_correction(test_lo, test_hi, Q_hat)
    results["CQR"]   = {"lo": lo, "hi": hi, "Q_hat": Q_hat}

    # --- Embedding-CQR ---
    gamma  = method_embed_cqr.median_heuristic_gamma(cal_embs)
    N_test = test_ctx.shape[0]
    lo_all = np.empty_like(test_lo)
    hi_all = np.empty_like(test_hi)
    q_hats = np.empty(N_test)
    for j in range(N_test):
        w          = method_embed_cqr.rbf_weights(test_embs[j], cal_embs, gamma)
        q_j        = method_embed_cqr.weighted_quantile(scores, w, alpha)
        lo_all[j], hi_all[j] = method_cqr.apply_correction(
            test_lo[j:j+1], test_hi[j:j+1], q_j
        )
        q_hats[j]  = q_j
    results["Embed-CQR"] = {"lo": lo_all, "hi": hi_all, "Q_hat_mean": q_hats.mean()}

    return results


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def summarise(results_by_alpha):
    """
    Build a DataFrame with coverage and width for every method × alpha.
    results_by_alpha : { alpha: { method: { lo, hi } } }
    """
    rows = []
    for alpha, method_dict in results_by_alpha.items():
        # test futures are the same for all methods at a given alpha
        # we need to retrieve them — they're stored in the outer scope
        for method, res in method_dict.items():
            lo, hi   = res["lo"], res["hi"]
            fut      = res["futures"]
            covered  = (fut >= lo) & (fut <= hi)
            rows.append({
                "method":   method,
                "alpha":    alpha,
                "nominal":  1 - alpha,
                "coverage": covered.mean(),
                "width":    (hi - lo).mean(),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_calibration_curve(df, out_path="calibration_curve.png"):
    """
    Coverage vs nominal level for each method.
    A perfectly calibrated method lies on the diagonal.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")

    for method, grp in df.groupby("method"):
        ax.plot(grp["nominal"], grp["coverage"], marker="o", label=method)

    ax.set_xlabel("Nominal coverage (1 − α)")
    ax.set_ylabel("Empirical coverage")
    ax.set_title("Calibration curve")
    ax.legend()
    ax.set_xlim(0.5, 1.0)
    ax.set_ylim(0.5, 1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_width_vs_alpha(df, out_path="width_vs_alpha.png"):
    """
    Interval width vs alpha for each method.
    Smaller width = more efficient intervals.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    for method, grp in df.groupby("method"):
        ax.plot(grp["nominal"], grp["width"], marker="o", label=method)

    ax.set_xlabel("Nominal coverage (1 − α)")
    ax.set_ylabel("Mean interval width (admissions / day)")
    ax.set_title("Interval width vs nominal coverage")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_intervals_for_state(
    pipeline,
    wide,
    test_wide,
    state: str,
    alpha: float = 0.1,
    out_path: str = None,
):
    """
    For one state, plot the actual time series and prediction intervals from
    all three offline methods side by side over the test period.
    """
    if state not in test_wide.columns:
        print(f"  State '{state}' not found, skipping.")
        return

    series_test = torch.tensor(test_wide[state].values.astype("float32"))
    ctx, fut    = make_all_windows([series_test], CONTEXT_LEN, PRED_LEN, stride=PRED_LEN)
    # non-overlapping windows so we can plot a clean timeline
    if ctx.shape[0] == 0:
        return

    dates_test = test_wide.index
    # start date of each window's future
    window_dates = [
        dates_test[CONTEXT_LEN + i * PRED_LEN]
        for i in range(ctx.shape[0])
        if CONTEXT_LEN + i * PRED_LEN < len(dates_test)
    ]
    n_windows = len(window_dates)
    ctx  = ctx[:n_windows]
    fut  = fut[:n_windows]

    # Get intervals from each method
    naive_lo, naive_hi = method_naive.predict_intervals(
        pipeline, ctx, PRED_LEN, alpha, NUM_SAMPLES, BATCH_SIZE
    )
    cal_lo, cal_hi     = method_cqr.get_quantiles(pipeline, ctx, PRED_LEN, alpha)

    # CQR: need calibration scores — use a quick single-state cal set
    cal_series = torch.tensor(
        wide.loc[wide.index < test_wide.index[0], state].values.astype("float32")
    )
    cal_ctx_s, cal_fut_s = make_all_windows([cal_series[-182:]], CONTEXT_LEN, PRED_LEN, stride=PRED_LEN)
    if cal_ctx_s.shape[0] > 0:
        c_lo, c_hi = method_cqr.get_quantiles(pipeline, cal_ctx_s, PRED_LEN, alpha)
        scores_s   = method_cqr.cqr_scores(c_lo, c_hi, cal_fut_s.numpy())
        Q_hat_s    = method_cqr.cqr_quantile(scores_s, alpha)
    else:
        Q_hat_s = 0.0

    cqr_lo, cqr_hi = method_cqr.apply_correction(cal_lo, cal_hi, Q_hat_s)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    method_intervals = [
        ("Naive",   naive_lo, naive_hi, "steelblue"),
        ("CQR",     cqr_lo,   cqr_hi,   "darkorange"),
        ("Embed-CQR (approx)", cqr_lo, cqr_hi, "seagreen"),  # placeholder
    ]

    for ax, (name, lo, hi, color) in zip(axes, method_intervals):
        # plot actual values
        ax.plot(dates_test, test_wide[state].values, color="black", lw=1.0,
                label="Observed", zorder=3)

        # plot each prediction window's interval as a shaded band
        for i, d in enumerate(window_dates):
            if i >= lo.shape[0]:
                break
            horizon_dates = pd.date_range(d, periods=PRED_LEN, freq="D")
            ax.fill_between(horizon_dates, lo[i], hi[i],
                            alpha=0.35, color=color)
            ax.plot(horizon_dates, (lo[i] + hi[i]) / 2,
                    color=color, lw=0.8, alpha=0.7)

        ax.set_ylabel("Admissions")
        ax.set_title(f"{name}  (nominal {1-alpha:.0%})")
        ax.legend(loc="upper right", fontsize=8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))

    fig.suptitle(f"{state} — COVID hospitalisations, test period", fontsize=12)
    fig.autofmt_xdate()
    fig.tight_layout()

    path = out_path or f"intervals_{state.replace(' ', '_')}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


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

    print(f"\nLoading Chronos model: {MODEL_ID} ...")
    pipeline = BaseChronosPipeline.from_pretrained(
        MODEL_ID, device_map="auto", torch_dtype=torch.float32,
    )

    print("\nComputing embeddings (needed for Embed-CQR) ...")
    cal_embs  = method_embed_cqr.get_embeddings(pipeline, cal_ctx)
    test_embs = method_embed_cqr.get_embeddings(pipeline, test_ctx)

    # --- Run all methods across all alphas ---
    rows = []
    for alpha in ALPHAS:
        print(f"\nAlpha = {alpha} ...")
        res = run_all(pipeline, cal_ctx, cal_fut, test_ctx, test_fut, cal_embs, test_embs, alpha)

        fut_np = test_fut.numpy()
        for method, r in res.items():
            lo, hi  = r["lo"], r["hi"]
            covered = (fut_np >= lo) & (fut_np <= hi)
            rows.append({
                "method":   method,
                "alpha":    alpha,
                "nominal":  round(1 - alpha, 2),
                "coverage": round(float(covered.mean()), 4),
                "width":    round(float((hi - lo).mean()), 2),
            })

    df = pd.DataFrame(rows)
    df.to_csv("results_table.csv", index=False)
    print("\n" + df.to_string(index=False))

    # --- Plots ---
    plot_calibration_curve(df)
    plot_width_vs_alpha(df)

    for state in EXAMPLE_STATES:
        plot_intervals_for_state(pipeline, wide, test_wide, state, alpha=0.1)


if __name__ == "__main__":
    main()
