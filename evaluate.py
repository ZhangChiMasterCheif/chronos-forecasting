"""
Unified evaluation: runs all methods and produces comparison tables and plots.

Results are saved incrementally to results_table.csv after each (method, alpha)
pair completes. Re-running skips any combination already in the CSV.

Outputs
-------
  results_table.csv       — coverage and width for every method × alpha
  calibration_curve.png   — coverage vs nominal level for all methods
  width_vs_alpha.png      — interval width vs alpha for all methods
  intervals_<state>.png   — prediction intervals over time for example states

Usage
-----
    python evaluate.py              # run everything, skip already-done
    python evaluate.py --plots-only # skip all experiments, just redo plots
"""

import argparse
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from chronos import BaseChronosPipeline

from covid_dataset import load_hosp_wide, split_wide, to_tensor_list, make_all_windows
import method_naive
import method_cqr
import method_embed_cqr
import method_pid

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

EXAMPLE_STATES = ["New York", "California", "Texas", "Florida"]
RESULTS_CSV    = "results_table.csv"

ALL_METHODS = ["Naive", "CQR", "Embed-CQR", "PID"]


# ---------------------------------------------------------------------------
# Incremental CSV helpers
# ---------------------------------------------------------------------------

def load_existing_results() -> pd.DataFrame:
    if os.path.exists(RESULTS_CSV):
        return pd.read_csv(RESULTS_CSV)
    return pd.DataFrame(columns=["method", "alpha", "nominal", "coverage", "width"])


def already_done(df_existing: pd.DataFrame, method: str, alpha: float) -> bool:
    if df_existing.empty:
        return False
    return ((df_existing["method"] == method) & (df_existing["alpha"] == alpha)).any()


def append_result(method: str, alpha: float, coverage: float, width: float):
    row = pd.DataFrame([{
        "method":   method,
        "alpha":    alpha,
        "nominal":  round(1 - alpha, 2),
        "coverage": round(coverage, 4),
        "width":    round(width, 2),
    }])
    write_header = not os.path.exists(RESULTS_CSV)
    row.to_csv(RESULTS_CSV, mode="a", header=write_header, index=False)


def metrics_from_arrays(lo: np.ndarray, hi: np.ndarray, futures: np.ndarray) -> dict:
    covered = (futures >= lo) & (futures <= hi)
    return {
        "coverage":           float(covered.mean()),
        "width":              float((hi - lo).mean()),
        "coverage_per_step":  covered.mean(axis=0),
        "width_per_step":     (hi - lo).mean(axis=0),
    }


# ---------------------------------------------------------------------------
# Run one (method, alpha) and return (lo, hi, futures_np)
# ---------------------------------------------------------------------------

def run_naive(pipeline, test_ctx, test_fut, alpha):
    lo, hi = method_naive.predict_intervals(
        pipeline, test_ctx, PRED_LEN, alpha, NUM_SAMPLES, BATCH_SIZE
    )
    return lo, hi, test_fut.numpy()


def run_cqr(pipeline, cal_ctx, cal_fut, test_ctx, test_fut, alpha):
    cal_lo, cal_hi   = method_cqr.get_quantiles(pipeline, cal_ctx, PRED_LEN, alpha)
    scores           = method_cqr.cqr_scores(cal_lo, cal_hi, cal_fut.numpy())
    Q_hat            = method_cqr.cqr_quantile(scores, alpha)
    test_lo, test_hi = method_cqr.get_quantiles(pipeline, test_ctx, PRED_LEN, alpha)
    lo, hi           = method_cqr.apply_correction(test_lo, test_hi, Q_hat)
    return lo, hi, test_fut.numpy()


def run_embed_cqr(pipeline, cal_ctx, cal_fut, test_ctx, test_fut,
                  cal_embs, test_embs, alpha):
    gamma            = method_embed_cqr.median_heuristic_gamma(cal_embs)
    cal_lo, cal_hi   = method_cqr.get_quantiles(pipeline, cal_ctx, PRED_LEN, alpha)
    scores           = method_cqr.cqr_scores(cal_lo, cal_hi, cal_fut.numpy())
    test_lo, test_hi = method_cqr.get_quantiles(pipeline, test_ctx, PRED_LEN, alpha)

    N_test = test_ctx.shape[0]
    lo_all = np.empty_like(test_lo)
    hi_all = np.empty_like(test_hi)

    for j in range(N_test):
        w         = method_embed_cqr.rbf_weights(test_embs[j], cal_embs, gamma)
        Q_hat_j   = method_embed_cqr.weighted_quantile(scores, w, alpha)
        lo_all[j], hi_all[j] = method_cqr.apply_correction(
            test_lo[j:j+1], test_hi[j:j+1], Q_hat_j
        )

    return lo_all, hi_all, test_fut.numpy()


def run_pid_method(pipeline, wide, train_wide, cal_wide, alpha):
    result = method_pid.run_pid(pipeline, wide, train_wide, cal_wide,
                                test_wide=None, alpha=alpha)
    # run_pid returns lo (N, pred_len) and hi (N, pred_len) over all states/windows
    # futures are embedded inside run_pid; we need to reconstruct them for consistency
    # Instead, use joint_coverage and width directly from run_pid's metrics
    return result["lo"], result["hi"], result.get("futures")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_calibration_curve(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0.5, 1.0], [0.5, 1.0], "k--", lw=1, label="Perfect calibration")

    for method in ALL_METHODS:
        grp = df[df["method"] == method].sort_values("nominal")
        if grp.empty:
            continue
        ax.plot(grp["nominal"], grp["coverage"], marker="o", label=method)

    ax.set_xlabel("Nominal coverage (1 − α)")
    ax.set_ylabel("Empirical coverage")
    ax.set_title("Calibration curve — all methods")
    ax.legend()
    ax.set_xlim(0.55, 1.02)
    ax.set_ylim(0.55, 1.02)
    fig.tight_layout()
    fig.savefig("calibration_curve.png", dpi=150)
    plt.close(fig)
    print("Saved calibration_curve.png")


def plot_width_vs_alpha(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6, 4))

    for method in ALL_METHODS:
        grp = df[df["method"] == method].sort_values("nominal")
        if grp.empty:
            continue
        ax.plot(grp["nominal"], grp["width"], marker="o", label=method)

    ax.set_xlabel("Nominal coverage (1 − α)")
    ax.set_ylabel("Mean interval width (admissions)")
    ax.set_title("Interval width vs nominal coverage")
    ax.legend()
    fig.tight_layout()
    fig.savefig("width_vs_alpha.png", dpi=150)
    plt.close(fig)
    print("Saved width_vs_alpha.png")


def plot_intervals_for_state(
    pipeline, wide, test_wide, state: str,
    cal_ctx_all, cal_fut_all, cal_embs_all,
    alpha: float = 0.10,
):
    """
    Parameters
    ----------
    cal_ctx_all  : (N_cal, context_len)  — full 50-state calibration contexts
    cal_fut_all  : (N_cal, pred_len)     — full 50-state calibration futures
    cal_embs_all : (N_cal, d_model)      — pre-computed 50-state embeddings
    """
    if state not in test_wide.columns:
        print(f"  '{state}' not found, skipping.")
        return

    series_test = torch.tensor(test_wide[state].values.astype("float32"))
    ctx, fut    = make_all_windows([series_test], CONTEXT_LEN, PRED_LEN, stride=PRED_LEN)
    if ctx.shape[0] == 0:
        return

    dates_test = test_wide.index
    n_windows  = ctx.shape[0]

    window_starts = [
        dates_test[CONTEXT_LEN + i * PRED_LEN]
        for i in range(n_windows)
        if CONTEXT_LEN + i * PRED_LEN < len(dates_test)
    ]
    n_windows = len(window_starts)
    ctx    = ctx[:n_windows]
    fut    = fut[:n_windows]
    fut_np = fut.numpy()

    # shared calibration scores from the full 50-state cal set
    cal_lo_all, cal_hi_all = method_cqr.get_quantiles(
        pipeline, cal_ctx_all, PRED_LEN, alpha
    )
    scores_all = method_cqr.cqr_scores(cal_lo_all, cal_hi_all, cal_fut_all.numpy())
    Q_hat      = method_cqr.cqr_quantile(scores_all, alpha)
    gamma      = method_embed_cqr.median_heuristic_gamma(cal_embs_all)

    naive_lo, naive_hi = method_naive.predict_intervals(pipeline, ctx, PRED_LEN, alpha)
    base_lo, base_hi   = method_cqr.get_quantiles(pipeline, ctx, PRED_LEN, alpha)
    cqr_lo, cqr_hi     = method_cqr.apply_correction(base_lo, base_hi, Q_hat)

    # Embed-CQR: weight 50-state cal scores by similarity to each test window
    test_embs = method_embed_cqr.get_embeddings(pipeline, ctx)
    ecqr_lo   = np.empty_like(base_lo)
    ecqr_hi   = np.empty_like(base_hi)
    for j in range(ctx.shape[0]):
        w          = method_embed_cqr.rbf_weights(test_embs[j], cal_embs_all, gamma)
        Q_hat_j    = method_embed_cqr.weighted_quantile(scores_all, w, alpha)
        ecqr_lo[j], ecqr_hi[j] = method_cqr.apply_correction(
            base_lo[j:j+1], base_hi[j:j+1], Q_hat_j
        )

    # PID: run online for this single state, warm-started with the 50-state Q_hat
    full_series    = torch.tensor(wide[state].values.astype("float32"))
    test_start_idx = len(wide) - len(test_wide)
    pid_lo, pid_hi, _ = method_pid.run_pid_one_state(
        pipeline, full_series, test_start_idx, alpha, q0=Q_hat
    )

    method_intervals = [
        ("Naive",      naive_lo, naive_hi, "steelblue"),
        ("CQR",        cqr_lo,   cqr_hi,   "darkorange"),
        ("Embed-CQR",  ecqr_lo,  ecqr_hi,  "seagreen"),
        ("PID",        pid_lo,   pid_hi,   "mediumpurple"),
    ]

    fig, axes = plt.subplots(len(method_intervals), 1,
                             figsize=(13, 4 * len(method_intervals)), sharex=True)
    if len(method_intervals) == 1:
        axes = [axes]

    obs = test_wide[state].values

    for ax, (name, lo, hi, color) in zip(axes, method_intervals):
        ax.plot(dates_test, obs, color="black", lw=1.0, label="Observed", zorder=3)

        for i, d in enumerate(window_starts):
            if i >= lo.shape[0]:
                break
            h_dates = pd.date_range(d, periods=PRED_LEN, freq="D")
            ax.fill_between(h_dates, lo[i], hi[i], alpha=0.35, color=color)
            ax.plot(h_dates, (lo[i] + hi[i]) / 2, color=color, lw=0.8, alpha=0.7)

        cov = float(((fut_np >= lo[:n_windows]) & (fut_np <= hi[:n_windows])).mean())
        ax.set_ylabel("Admissions")
        ax.set_title(f"{name}  (nominal {1-alpha:.0%}, empirical {cov:.1%})")
        ax.legend(loc="upper right", fontsize=8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))

    fig.suptitle(f"{state} — COVID hospitalisations (test period)", fontsize=12)
    fig.autofmt_xdate()
    fig.tight_layout()

    out = f"intervals_{state.replace(' ', '_')}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(plots_only=False):
    df_existing = load_existing_results()

    # Data and pipeline are always needed: experiments need them, and so do
    # the per-state interval plots (which call the model directly).
    print("Loading data ...")
    wide = load_hosp_wide()
    train_wide, cal_wide, test_wide = split_wide(wide)

    cal_tensors  = to_tensor_list(cal_wide)
    test_tensors = to_tensor_list(test_wide)
    cal_ctx,  cal_fut  = make_all_windows(cal_tensors,  CONTEXT_LEN, PRED_LEN, stride=STRIDE)
    test_ctx, test_fut = make_all_windows(test_tensors, CONTEXT_LEN, PRED_LEN, stride=STRIDE)

    print(f"\nLoading Chronos model: {MODEL_ID} ...")
    pipeline = BaseChronosPipeline.from_pretrained(
        MODEL_ID, device_map="auto", torch_dtype=torch.float32,
    )

    if not plots_only:

        # Embeddings only needed for Embed-CQR
        need_embed = any(
            not already_done(df_existing, "Embed-CQR", a) for a in ALPHAS
        )
        if need_embed:
            print("\nComputing calibration embeddings ...")
            cal_embs  = method_embed_cqr.get_embeddings(pipeline, cal_ctx)
            print("Computing test embeddings ...")
            test_embs = method_embed_cqr.get_embeddings(pipeline, test_ctx)
        else:
            cal_embs = test_embs = None

        for alpha in ALPHAS:
            for method in ALL_METHODS:
                if already_done(df_existing, method, alpha):
                    print(f"  [skip] {method}  α={alpha}")
                    continue

                print(f"  [run]  {method}  α={alpha} ...")

                if method == "Naive":
                    lo, hi, fut_np = run_naive(pipeline, test_ctx, test_fut, alpha)

                elif method == "CQR":
                    lo, hi, fut_np = run_cqr(pipeline, cal_ctx, cal_fut,
                                             test_ctx, test_fut, alpha)

                elif method == "Embed-CQR":
                    lo, hi, fut_np = run_embed_cqr(
                        pipeline, cal_ctx, cal_fut, test_ctx, test_fut,
                        cal_embs, test_embs, alpha
                    )

                elif method == "PID":
                    # PID needs the full time series, not pre-windowed tensors
                    result = method_pid.run_pid(
                        pipeline, wide, train_wide, cal_wide,
                        test_wide=test_wide, alpha=alpha
                    )
                    lo, hi = result["lo"], result["hi"]
                    # reconstruct futures from test windows to match shape
                    # PID strides by pred_len internally; use non-overlapping windows
                    test_tensors_nonoverlap = to_tensor_list(test_wide)
                    _, fut_pid = make_all_windows(
                        test_tensors_nonoverlap, CONTEXT_LEN, PRED_LEN, stride=PRED_LEN
                    )
                    fut_np = fut_pid.numpy()
                    # trim to match PID output length
                    n = min(lo.shape[0], fut_np.shape[0])
                    lo, hi, fut_np = lo[:n], hi[:n], fut_np[:n]

                m = metrics_from_arrays(lo, hi, fut_np)
                append_result(method, alpha, m["coverage"], m["width"])
                print(f"         coverage={m['coverage']:.3f}  width={m['width']:.2f}")

    # --- Plots (always regenerated) ---
    df = pd.read_csv(RESULTS_CSV)
    print("\n" + df.sort_values(["method", "alpha"]).to_string(index=False))

    plot_calibration_curve(df)
    plot_width_vs_alpha(df)

    # Embeddings needed for per-state Embed-CQR plots — compute if not already done
    if "cal_embs" not in dir():
        print("\nComputing calibration embeddings for plots ...")
        cal_embs  = method_embed_cqr.get_embeddings(pipeline, cal_ctx)
    for state in EXAMPLE_STATES:
        plot_intervals_for_state(
            pipeline, wide, test_wide, state,
            cal_ctx_all=cal_ctx, cal_fut_all=cal_fut, cal_embs_all=cal_embs,
            alpha=0.10,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plots-only", action="store_true",
                        help="Skip experiments; regenerate plots from saved CSV")
    args = parser.parse_args()
    main(plots_only=args.plots_only)
