"""
Quantile-head forecasters as baselines, uncalibrated and conformalised.

Answers r3Md Q4 and the metareview's point that "stronger available baselines
with dedicated quantile heads (TimesFM 2.5, Chronos 2.0) are not compared
against, despite being public at submission time."

Models
------
  amazon/chronos-t5-small    the submission's model: sampling, discrete codebook
  amazon/chronos-bolt-base   direct quantile regression head, no sampling
  amazon/chronos-2           direct quantile regression head, no sampling

TimesFM 2.5 lives outside this repo and needs the `timesfm` package; the same
protocol applies and a hook is left in `MODELS` for it.

What is measured, per model and per alpha
-----------------------------------------
  1. UNCALIBRATED  -- the model's own predicted alpha/2 and 1-alpha/2 quantiles,
     used exactly as shipped. This is the number that decides whether the
     submission's framing survives: if the quantile heads are already
     well calibrated, then "TS-LLM intervals are miscalibrated" is a statement
     about sampling-based decoders specifically, not about the field.
  2. CONFORMALISED -- split CQR on top of those same quantile heads. CQR was
     designed exactly for this: a quantile regressor plus a calibration set. It
     is the natural, strongest version of the baseline.

A scope limitation worth stating plainly in the rebuttal
--------------------------------------------------------
Chronos-Bolt, Chronos-2 and TimesFM 2.5 do not tokenize values into a discrete
vocabulary -- they regress quantiles directly. The proposed codebook score is
therefore INAPPLICABLE to them. If quantile heads are where the field is going,
that bounds the reach of the method, and the paper should say so rather than
leave the reviewer to infer it. The scale-normalised CQR score, by contrast,
applies to any forecaster, which is another reason to foreground it.

Usage
-----
    python quantile_head_baselines.py
    python quantile_head_baselines.py --models amazon/chronos-bolt-base
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
from chronos import BaseChronosPipeline

import conformal_lib as cl
from covid_dataset import load_hosp_wide, split_wide, to_tensor_list, make_all_windows

CONTEXT_LEN = 60
PRED_LEN = 14
STRIDE = 7
BATCH_SIZE = 32
ALPHAS = [0.05, 0.10, 0.20, 0.30, 0.40]
OUT_CSV = "rebuttal_quantile_heads.csv"

MODELS = [
    "amazon/chronos-t5-small",
    "amazon/chronos-bolt-base",
    "amazon/chronos-2",
    # "google/timesfm-2.5-200m-pytorch",  # needs the separate `timesfm` package
]


@torch.no_grad()
def predict_quantile_bounds(pipeline, contexts, alpha):
    """
    Ask the model directly for its alpha/2 and 1-alpha/2 quantiles.

    Uses the unified `BaseChronosPipeline.predict_quantiles` API, which every
    Chronos variant implements; Chronos-2 returns a list of per-item tensors
    while the T5 and Bolt pipelines return a single batched tensor, so we
    normalise both to a (N, H) pair.
    """
    levels = [alpha / 2.0, 1.0 - alpha / 2.0]
    lo_all, hi_all = [], []

    for start in range(0, contexts.shape[0], BATCH_SIZE):
        batch = contexts[start : start + BATCH_SIZE]
        q, _ = pipeline.predict_quantiles(
            batch, prediction_length=PRED_LEN, quantile_levels=levels
        )
        if isinstance(q, (list, tuple)):
            q = torch.stack([qi if qi.ndim == 2 else qi.squeeze(0) for qi in q])
        q = q.float().cpu().numpy()          # (B, H, 2)
        lo_all.append(q[..., 0])
        hi_all.append(q[..., 1])

    return np.concatenate(lo_all), np.concatenate(hi_all)


def cqr_on_quantile_head(cal_lo, cal_hi, cal_fut, test_lo, test_hi, alpha):
    """Split CQR using the model's own quantile head as the base estimator."""
    scores = np.maximum(cal_lo - cal_fut, cal_fut - cal_hi).max(axis=1)
    q = cl.conformal_quantile(scores, alpha)
    return test_lo - q, test_hi + q


def cqr_perstep_on_quantile_head(cal_lo, cal_hi, cal_fut, test_lo, test_hi, alpha):
    """Per-horizon calibration, so the target matches marginal per-step reporting."""
    per = np.maximum(cal_lo - cal_fut, cal_fut - cal_hi)
    lo = np.empty_like(test_lo)
    hi = np.empty_like(test_hi)
    for h in range(per.shape[1]):
        q_h = cl.conformal_quantile(per[:, h], alpha)
        lo[:, h] = test_lo[:, h] - q_h
        hi[:, h] = test_hi[:, h] + q_h
    return lo, hi


def main(models):
    print("Loading COVID data ...")
    wide = load_hosp_wide()
    _, cal_wide, test_wide = split_wide(wide)
    cal_ctx, cal_fut = make_all_windows(
        to_tensor_list(cal_wide), CONTEXT_LEN, PRED_LEN, stride=STRIDE
    )
    test_ctx, test_fut = make_all_windows(
        to_tensor_list(test_wide), CONTEXT_LEN, PRED_LEN, stride=STRIDE
    )
    cal_f = cal_fut.numpy()
    test_f = test_fut.numpy()
    print(f"  cal={len(cal_f)} test={len(test_f)}")

    rows = []
    for model_id in models:
        print(f"\n=== {model_id} ===")
        try:
            pipe = BaseChronosPipeline.from_pretrained(
                model_id, device_map="auto", torch_dtype=torch.float32
            )
        except Exception as e:                      # noqa: BLE001
            print(f"  could not load ({type(e).__name__}: {e}); skipping")
            continue

        for alpha in ALPHAS:
            try:
                c_lo, c_hi = predict_quantile_bounds(pipe, cal_ctx, alpha)
                t_lo, t_hi = predict_quantile_bounds(pipe, test_ctx, alpha)
            except Exception as e:                  # noqa: BLE001
                print(f"  alpha={alpha}: predict_quantiles failed "
                      f"({type(e).__name__}: {e}); skipping")
                continue

            variants = {
                "uncalibrated (model quantile head)": (t_lo, t_hi),
                "+ CQR (joint)": cqr_on_quantile_head(
                    c_lo, c_hi, cal_f, t_lo, t_hi, alpha),
                "+ CQR (per-step)": cqr_perstep_on_quantile_head(
                    c_lo, c_hi, cal_f, t_lo, t_hi, alpha),
            }
            for name, (lo, hi) in variants.items():
                m = cl.evaluate_intervals(lo, hi, test_f, alpha, nonneg=True)
                rows.append(dict(
                    model=model_id, variant=name, alpha=alpha,
                    nominal=round(1 - alpha, 2),
                    cov_perstep=m["coverage_perstep"],
                    cov_joint=m["coverage_joint"],
                    width=m["width"], winkler=m["winkler"],
                    frac_lo_neg=m["frac_lo_negative"],
                ))
                print(f"  a={alpha:.2f} {name:<36} "
                      f"per-step {m['coverage_perstep']:.3f}  "
                      f"joint {m['coverage_joint']:.3f}  "
                      f"width {m['width']:.1f}")

        del pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not rows:
        print("\nNo results -- all models failed to load.")
        return

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_CSV}")

    print("\n" + "=" * 82)
    print("Per-step coverage, uncalibrated quantile heads "
          "(the number that decides the framing)")
    print("=" * 82)
    unc = df[df.variant.str.startswith("uncalibrated")]
    print(unc.pivot_table(index="model", columns="nominal", values="cov_perstep")
             .to_string(float_format=lambda v: f"{v:.3f}"))
    print("\nJoint coverage, same rows:")
    print(unc.pivot_table(index="model", columns="nominal", values="cov_joint")
             .to_string(float_format=lambda v: f"{v:.3f}"))
    print("\nIf these sit close to the nominal level, the miscalibration claim is\n"
          "specifically about sampling-based decoding, not about TSFMs generally,\n"
          "and the paper's framing needs narrowing to match.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=MODELS)
    a = p.parse_args()
    main(a.models)
