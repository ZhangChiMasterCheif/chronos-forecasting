"""
Robustness experiment: re-run the Chronos COVID benchmark on a 2021 split.

The main evaluation in evaluate.py calibrates on Jan-Jun 2022 and tests on
Jul 2022 onwards, a period where the pandemic dynamics had largely settled
into an endemic plateau. As a robustness check we shift everything one year
earlier:

  calibration : Jan-Jun 2021  (post-winter-2020 decline, vaccine ramp-up)
  test        : Jul-Dec 2021  (Delta wave, large state-level variation)

This re-uses the existing dataset loader, method implementations, and runners
from evaluate.py. The only thing that changes is which dates form the cal and
test partitions, and where the outputs are written:

  results_table_2021.csv
  calibration_curve_2021.png
  width_vs_alpha_2021.png

Usage
-----
    python evaluate_2021.py              # run missing combinations, then plot
    python evaluate_2021.py --plots-only # skip experiments, regenerate plots
    python evaluate_2021.py --reset      # wipe CSV first, run everything
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

from covid_dataset import load_hosp_wide, to_tensor_list, make_all_windows
import method_embed_cqr
import method_pid
import evaluate as ev   # reuse run_naive, run_cqr, run_codebook_cqr, run_embed_cqr, helpers

# ---------------------------------------------------------------------------
# Config (mirrors evaluate.py, but with 2021 splits and 2021-suffixed outputs)
# ---------------------------------------------------------------------------

MODEL_ID    = ev.MODEL_ID
CONTEXT_LEN = ev.CONTEXT_LEN
PRED_LEN    = ev.PRED_LEN
STRIDE      = ev.STRIDE
NUM_SAMPLES = ev.NUM_SAMPLES
BATCH_SIZE  = ev.BATCH_SIZE
ALPHAS      = ev.ALPHAS
ALL_METHODS = ev.ALL_METHODS

# 2021 split (one year earlier than the main experiment)
TRAIN_END_2021  = "2020-12-31"
CAL_START_2021  = "2021-01-01"
CAL_END_2021    = "2021-06-30"
TEST_START_2021 = "2021-07-01"
TEST_END_2021   = "2021-12-31"

RESULTS_CSV = "results_table_2021.csv"


# ---------------------------------------------------------------------------
# CSV bookkeeping — same shape as evaluate.py but with the 2021-suffixed file
# ---------------------------------------------------------------------------

def load_existing() -> pd.DataFrame:
    if os.path.exists(RESULTS_CSV):
        return pd.read_csv(RESULTS_CSV)
    return pd.DataFrame(columns=["method", "alpha", "nominal", "coverage", "width"])


def already_done(df, method, alpha):
    if df.empty:
        return False
    return ((df["method"] == method) & (df["alpha"] == alpha)).any()


def append_result(method, alpha, coverage, width):
    row = pd.DataFrame([{
        "method":   method,
        "alpha":    alpha,
        "nominal":  round(1 - alpha, 2),
        "coverage": round(coverage, 4),
        "width":    round(width, 2),
    }])
    write_header = not os.path.exists(RESULTS_CSV)
    row.to_csv(RESULTS_CSV, mode="a", header=write_header, index=False)


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------

def split_wide_2021(wide):
    """Same shape as covid_dataset.split_wide but one year earlier."""
    train = wide.loc[:TRAIN_END_2021]
    cal   = wide.loc[CAL_START_2021:CAL_END_2021]
    test  = wide.loc[TEST_START_2021:TEST_END_2021]
    return train, cal, test


# ---------------------------------------------------------------------------
# Plotting (same logic as evaluate.py, different output filenames)
# ---------------------------------------------------------------------------

def plot_calibration_curve(df):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0.5, 1.0], [0.5, 1.0], "k--", lw=1, label="Perfect calibration")
    for method in ALL_METHODS:
        grp = df[df["method"] == method].sort_values("nominal")
        if grp.empty:
            continue
        ax.plot(grp["nominal"], grp["coverage"], marker="o", label=method)
    ax.set_xlabel("Nominal coverage (1 − α)")
    ax.set_ylabel("Empirical coverage")
    ax.set_title("Calibration curve — 2021 cal/test split")
    ax.legend()
    ax.set_xlim(0.55, 1.02)
    ax.set_ylim(0.55, 1.02)
    fig.tight_layout()
    fig.savefig("calibration_curve_2021.png", dpi=150)
    plt.close(fig)
    print("Saved calibration_curve_2021.png")


def plot_width_vs_alpha(df):
    fig, ax = plt.subplots(figsize=(6, 4))
    for method in ALL_METHODS:
        grp = df[df["method"] == method].sort_values("nominal")
        if grp.empty:
            continue
        ax.plot(grp["nominal"], grp["width"], marker="o", label=method)
    ax.set_xlabel("Nominal coverage (1 − α)")
    ax.set_ylabel("Mean interval width (admissions)")
    ax.set_title("Interval width vs nominal coverage — 2021 cal/test split")
    ax.legend()
    fig.tight_layout()
    fig.savefig("width_vs_alpha_2021.png", dpi=150)
    plt.close(fig)
    print("Saved width_vs_alpha_2021.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(plots_only: bool, reset: bool):
    if reset and os.path.exists(RESULTS_CSV):
        os.remove(RESULTS_CSV)
        print(f"Removed {RESULTS_CSV}")

    print("Loading data ...")
    wide = load_hosp_wide()
    train_wide, cal_wide, test_wide = split_wide_2021(wide)
    print(f"  train : {train_wide.index[0].date()} → {train_wide.index[-1].date()}  ({len(train_wide)} days)")
    print(f"  cal   : {cal_wide.index[0].date()} → {cal_wide.index[-1].date()}    ({len(cal_wide)} days)")
    print(f"  test  : {test_wide.index[0].date()} → {test_wide.index[-1].date()}  ({len(test_wide)} days)")

    cal_tensors  = to_tensor_list(cal_wide)
    test_tensors = to_tensor_list(test_wide)
    cal_ctx,  cal_fut  = make_all_windows(cal_tensors,  CONTEXT_LEN, PRED_LEN, stride=STRIDE)
    test_ctx, test_fut = make_all_windows(test_tensors, CONTEXT_LEN, PRED_LEN, stride=STRIDE)
    print(f"  cal windows : {cal_ctx.shape[0]}")
    print(f"  test windows: {test_ctx.shape[0]}")

    df_existing = load_existing()
    needed = [(m, a) for m in ALL_METHODS for a in ALPHAS
              if not already_done(df_existing, m, a)]

    if plots_only or not needed:
        if not needed:
            print("All combinations already done — only regenerating plots")
        pipeline = None
        cal_embs = test_embs = None
    else:
        print(f"\nLoading Chronos: {MODEL_ID}")
        pipeline = BaseChronosPipeline.from_pretrained(
            MODEL_ID, device_map="auto", torch_dtype=torch.float32,
        )

        need_embed = any(m == "Embed-CQR" for (m, _) in needed)
        if need_embed:
            print("Computing calibration embeddings ...")
            cal_embs  = method_embed_cqr.get_embeddings(pipeline, cal_ctx)
            print("Computing test embeddings ...")
            test_embs = method_embed_cqr.get_embeddings(pipeline, test_ctx)
        else:
            cal_embs = test_embs = None

        for (method, alpha) in needed:
            print(f"\n  [{method}, α={alpha}]")

            if method == "Naive":
                lo, hi, fut_np = ev.run_naive(pipeline, test_ctx, test_fut, alpha)

            elif method == "CQR":
                lo, hi, fut_np = ev.run_cqr(pipeline, cal_ctx, cal_fut,
                                            test_ctx, test_fut, alpha)

            elif method == "Codebook-CQR":
                lo, hi, fut_np = ev.run_codebook_cqr(
                    pipeline, cal_ctx, cal_fut, test_ctx, test_fut, alpha
                )

            elif method == "Embed-CQR":
                lo, hi, fut_np = ev.run_embed_cqr(
                    pipeline, cal_ctx, cal_fut, test_ctx, test_fut,
                    cal_embs, test_embs, alpha
                )

            elif method == "PID":
                # PID processes the full series, indexing into it by length.
                # We pass it the 2021-split DataFrames so its
                # test_start_idx = len(train_wide) + len(cal_wide) lands at
                # the first day of our 2021 test partition.
                result = method_pid.run_pid(
                    pipeline, wide, train_wide, cal_wide,
                    test_wide=test_wide, alpha=alpha
                )
                lo, hi = result["lo"], result["hi"]
                # PID strides by PRED_LEN internally; build matching futures.
                test_tensors_no  = to_tensor_list(test_wide)
                _, fut_pid       = make_all_windows(
                    test_tensors_no, CONTEXT_LEN, PRED_LEN, stride=PRED_LEN
                )
                fut_np = fut_pid.numpy()
                n = min(lo.shape[0], fut_np.shape[0])
                lo, hi, fut_np = lo[:n], hi[:n], fut_np[:n]

            else:
                raise ValueError(method)

            m = ev.metrics_from_arrays(lo, hi, fut_np)
            append_result(method, alpha, m["coverage"], m["width"])
            print(f"     coverage={m['coverage']:.3f}  width={m['width']:.2f}")

    df = pd.read_csv(RESULTS_CSV)
    print("\n" + df.sort_values(["method", "alpha"]).to_string(index=False))
    plot_calibration_curve(df)
    plot_width_vs_alpha(df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plots-only", action="store_true",
                        help="Skip experiments; regenerate plots from saved CSV")
    parser.add_argument("--reset", action="store_true",
                        help="Delete results_table_2021.csv before running")
    args = parser.parse_args()
    main(plots_only=args.plots_only, reset=args.reset)
