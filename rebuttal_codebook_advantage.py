"""
When, if ever, does codebook-distance CQR beat residual CQR?

This targets the single strongest objection in the metareview ("the method does
not clearly outperform standard CQR") and r3Md's Q3 ("give a practical example
where Codebook-CQR gives an advantage over CQR, beyond the theoretical
boundedness property").

The submission's own tables say it does not win on average width, and that is
not going to change. So the honest move is to identify the operating regime the
boundedness and scale-invariance properties were supposed to buy, and test
whether they actually buy it. Two protocols, both derived from the properties
claimed in Section 4.5, both runnable from the cached samples with no extra
inference:

E1  CROSS-SCALE TRANSFER (tests the scale-invariance claim)
    Calibrate on the 25 lowest-volume states, test on the 25 highest-volume
    states, and vice versa. Residual CQR's Q-hat is an absolute number of
    admissions, so it should transfer badly across a magnitude shift. The
    codebook score is computed after Chronos divides out the per-window scale,
    so it should transfer. This is an ordinary deployment situation -- calibrate
    on the series you have, deploy on a new series of different magnitude -- not
    an adversarial construction.

E2  CALIBRATION-SET CONTAMINATION (tests the boundedness claim)
    Corrupt a fraction of calibration targets with a large multiplicative
    outlier, as a stand-in for a data-entry error or an unflagged regime break.

    We expected the bounded codebook score to degrade more gracefully than the
    unbounded residual score. In our own runs it does NOT, and the rebuttal
    should not claim otherwise. Two things happen:

      * While the contamination rate stays well below alpha, every method is
        essentially unharmed -- the conformal quantile simply does not reach the
        corrupted tail. Both score families widen by only a few percent.
      * Once the contamination rate approaches alpha, the quantile lands on the
        corrupted region and every method breaks. The SCALE-RELATIVE scores
        (codebook and scale-normalised CQR) break harder, because a 20x outlier
        on a small-magnitude window is an enormous relative error but only a
        modest absolute one.

    Boundedness does not rescue this: with ~4000 bins spanning +/-15 sigma, a
    20x outlier sits comfortably inside the codebook, so the score never
    saturates. Saturation would only bite for corruptions large enough to leave
    the representable range entirely.

    We keep this experiment in because it is the honest test of a property we
    claimed, and it came out negative. Report it as a limitation of the
    scale-relative score, not as evidence for it.

IMPORTANT -- the honest control
-------------------------------
Both properties are also obtainable by simply normalising the residual before
calibrating: divide the CQR score by the same per-window scale Chronos uses.
That is a two-line change to standard CQR and needs no tokenizer at all. We
include it as `CQR (scale-normalised)`. If it matches codebook-CQR, then the
tokenizer is not doing the work and we should say so plainly in the paper
rather than let a reviewer discover it. Reporting this either way is the point.

Usage
-----
    python rebuttal_codebook_advantage.py            # needs cache/ from cache_samples.py
"""

import argparse
import os

import numpy as np
import pandas as pd

import conformal_lib as cl
import codebook_repair as cr

CACHE_DIR = "cache"
ALPHAS = [0.10, 0.20]
OUT_PREFIX = "rebuttal_codebook_advantage"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_split(split, seed=0):
    path = os.path.join(CACHE_DIR, f"chronos_{split}_seed{seed}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} missing -- run `python cache_samples.py --seeds {seed}` first"
        )
    d = np.load(path)
    return {k: d[k] for k in d.files}


def build_grid(d):
    return cr.BinGrid.from_chronos_arrays(
        centers=d["centers"], boundaries=_chronos_boundaries(d["centers"]),
        n_special=int(d["n_special"]),
    )


def _chronos_boundaries(centers):
    """Rebuild MeanScaleUniformBins.boundaries from its centers."""
    c = np.asarray(centers, dtype=np.float64)
    return np.concatenate(([-1e20], (c[1:] + c[:-1]) / 2.0, [1e20]))


def subset(d, mask):
    out = dict(d)
    for k in ("samples", "sample_bins", "scale", "true_bins", "futures", "state_idx"):
        if k in d:
            out[k] = d[k][mask]
    return out


# ---------------------------------------------------------------------------
# Methods under comparison
# ---------------------------------------------------------------------------

def cqr_scale_normalised(cal, test, alpha):
    """
    Split CQR whose score is divided by the per-window Chronos scale.

    score = max_h max(q_lo - y, y - q_hi) / sigma

    This is the cheap baseline that gets scale invariance without any
    tokenizer. Q-hat is dimensionless and is multiplied back by each test
    window's own sigma at inference time.
    """
    cal_lo, cal_hi = cl.sample_quantiles(cal["samples"], alpha)
    s = np.maximum(cal_lo - cal["futures"], cal["futures"] - cal_hi).max(axis=1)
    s = s / cal["scale"]
    q = cl.conformal_quantile(s, alpha)

    t_lo, t_hi = cl.sample_quantiles(test["samples"], alpha)
    adj = q * test["scale"][:, None]
    return t_lo - adj, t_hi + adj


def all_methods(cal, test, grid, alpha):
    out = {}
    out["CQR"] = cl.cqr_joint(cal["samples"], cal["futures"], test["samples"], alpha)
    out["CQR (scale-normalised)"] = cqr_scale_normalised(cal, test, alpha)

    for ep, label in [("center", "Codebook-CQR (center, submitted)"),
                      ("phi", "Codebook-CQR (phi, repaired)")]:
        lo, hi, _ = cr.codebook_cqr(
            cal["samples"], cal["futures"], cal["scale"],
            test["samples"], test["scale"], grid, alpha, endpoint=ep,
        )
        out[label] = (lo, hi)
    return out


def score_row(name, lo, hi, fut, alpha, **extra):
    m = cl.evaluate_intervals(lo, hi, fut, alpha, nonneg=True)
    return dict(
        method=name, alpha=alpha, nominal=round(1 - alpha, 2),
        cov_joint=m["coverage_joint"], cov_perstep=m["coverage_perstep"],
        width=m["width"], width_nonneg=m["width_nonneg"],
        winkler=m["winkler"], **extra,
    )


# ---------------------------------------------------------------------------
# E0 : conditional coverage across the scale range
# ---------------------------------------------------------------------------

def experiment_scale_conditional(cal, test, grid, n_strata=10):
    """
    Coverage and width stratified by the magnitude of the target series.

    This is the most important table in this file, and it needs no new
    inference at all -- it is a re-reading of the results already in the paper.

    The COVID test pool spans a ~700x range in daily admissions. Residual CQR
    calibrates a SINGLE absolute Q-hat in admissions and applies it to every
    window, so it necessarily over-covers small states and under-covers large
    ones; its valid *marginal* coverage is an average over that trade. The
    codebook score is computed after the per-window scale is divided out, so its
    correction is proportional to each window's own magnitude.

    On the submitted results this shows up sharply: pooled marginal coverage
    hides the fact that CQR fails on exactly the high-volume series that matter
    operationally. Reporting mean absolute width across a scale-heterogeneous
    pool -- which is what the tables do, and what the metareview's "Codebook-CQR
    is substantially wider" is based on -- is not a like-for-like comparison,
    because the two methods are distributing their width very differently.

    We therefore also report width normalised by series level, which is the
    scale-free efficiency metric.
    """
    level = np.maximum(np.median(test["futures"], axis=1), 1.0)
    edges = np.quantile(level, np.linspace(0, 1, n_strata + 1))[1:-1]
    stratum = np.clip(np.searchsorted(edges, level), 0, n_strata - 1)

    rows = []
    for alpha in ALPHAS:
        for name, (lo, hi) in all_methods(cal, test, grid, alpha).items():
            cov = (test["futures"] >= lo) & (test["futures"] <= hi)
            joint = cov.all(axis=1)
            width = (hi - lo).mean(axis=1)
            for k in range(n_strata):
                m = stratum == k
                if not m.any():
                    continue
                rows.append(dict(
                    method=name, alpha=alpha, nominal=round(1 - alpha, 2),
                    stratum=k, median_level=float(np.median(level[m])),
                    n=int(m.sum()),
                    cov_joint=float(joint[m].mean()),
                    cov_perstep=float(cov[m].mean()),
                    width=float(width[m].mean()),
                    width_over_level=float((width[m] / level[m]).mean()),
                ))
    return pd.DataFrame(rows)


def report_scale_conditional(df):
    print("\n" + "=" * 78)
    print("[E0] conditional coverage across the series-magnitude range")
    print("=" * 78)
    for alpha in ALPHAS:
        sub = df[df.alpha == alpha]
        print(f"\nalpha = {alpha} (nominal joint coverage {1 - alpha:.2f})")
        lv = sub.pivot_table(index="stratum", values="median_level", aggfunc="first")
        print("median series level by stratum: "
              + " ".join(f"{v:.0f}" for v in lv["median_level"]))
        print("\njoint coverage by stratum:")
        print(sub.pivot_table(index="method", columns="stratum", values="cov_joint")
                 .to_string(float_format=lambda v: f"{v:.3f}"))
        print("\nmean width by stratum:")
        print(sub.pivot_table(index="method", columns="stratum", values="width")
                 .to_string(float_format=lambda v: f"{v:.0f}"))

    print("\nScale-free efficiency (mean width / series level), pooled:")
    print(df.groupby(["method", "alpha"])["width_over_level"].mean()
            .unstack().to_string(float_format=lambda v: f"{v:.3f}"))
    print("\nWorst-stratum joint coverage (the operationally relevant number:")
    print("a method whose marginal coverage is valid can still fail on a whole")
    print("magnitude regime):")
    print(df.groupby(["method", "alpha"])["cov_joint"].min()
            .unstack().to_string(float_format=lambda v: f"{v:.3f}"))


# ---------------------------------------------------------------------------
# E1 : cross-scale transfer
# ---------------------------------------------------------------------------

def experiment_cross_scale(cal, test, grid):
    """
    Split states into low- and high-volume halves by median admissions in the
    CALIBRATION period, then calibrate on one half and test on the other.
    """
    vol = {}
    for s in np.unique(cal["state_idx"]):
        m = cal["state_idx"] == s
        vol[s] = float(np.nanmedian(cal["futures"][m]))
    states = np.array(sorted(vol, key=lambda s: vol[s]))
    low, high = states[: len(states) // 2], states[len(states) // 2:]

    lo_med = np.median([vol[s] for s in low])
    hi_med = np.median([vol[s] for s in high])
    print(f"  low-volume states : n={len(low)}, median admissions {lo_med:.1f}")
    print(f"  high-volume states: n={len(high)}, median admissions {hi_med:.1f}")
    print(f"  magnitude ratio   : {hi_med / max(lo_med, 1e-9):.1f}x")

    rows = []
    for direction, cal_states, test_states in [
        ("low -> high", low, high),
        ("high -> low", high, low),
        ("matched (low -> low)", low, low),
        ("matched (high -> high)", high, high),
    ]:
        c = subset(cal, np.isin(cal["state_idx"], cal_states))
        t = subset(test, np.isin(test["state_idx"], test_states))
        for alpha in ALPHAS:
            for name, (lo, hi) in all_methods(c, t, grid, alpha).items():
                rows.append(score_row(name, lo, hi, t["futures"], alpha,
                                      transfer=direction,
                                      n_cal=len(c["futures"]),
                                      n_test=len(t["futures"])))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# E2 : calibration contamination
# ---------------------------------------------------------------------------

def experiment_contamination(cal, test, grid, rates=(0.0, 0.01, 0.02, 0.05, 0.10),
                             factor=20.0, seed=0):
    """
    Multiply the targets of a random `rate` fraction of calibration windows by
    `factor`, leaving the test set untouched, and watch what happens to Q-hat.

    The test set is clean, so any width inflation is pure damage from the
    contaminated calibration set, and coverage above nominal is wasted width
    rather than a benefit.
    """
    rng = np.random.default_rng(seed)
    n = len(cal["futures"])
    rows = []

    for rate in rates:
        c = dict(cal)
        c["futures"] = cal["futures"].copy()
        if rate > 0:
            idx = rng.choice(n, size=int(round(rate * n)), replace=False)
            c["futures"][idx] *= factor

        for alpha in ALPHAS:
            for name, (lo, hi) in all_methods(c, test, grid, alpha).items():
                rows.append(score_row(name, lo, hi, test["futures"], alpha,
                                      contamination=rate, factor=factor))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def report_cross_scale(df):
    print("\n" + "=" * 78)
    print("[E1] cross-scale transfer -- calibrate on one volume regime, test on the other")
    print("=" * 78)
    for alpha in ALPHAS:
        sub = df[df.alpha == alpha]
        print(f"\nalpha = {alpha} (nominal joint coverage {1 - alpha:.2f})")
        piv = sub.pivot_table(index="method", columns="transfer",
                              values=["cov_joint", "width"])
        print(piv.to_string(float_format=lambda v: f"{v:.3f}"))

    print("\nCoverage shortfall under transfer (joint coverage minus nominal;")
    print("negative = the guarantee failed on the shifted regime):")
    sub = df[df.transfer.isin(["low -> high", "high -> low"])].copy()
    sub["shortfall"] = sub["cov_joint"] - sub["nominal"]
    print(sub.pivot_table(index="method", columns="transfer", values="shortfall")
             .to_string(float_format=lambda v: f"{v:+.3f}"))


def report_contamination(df):
    print("\n" + "=" * 78)
    print("[E2] calibration-set contamination -- clean test set throughout")
    print("=" * 78)
    for alpha in ALPHAS:
        sub = df[df.alpha == alpha]
        print(f"\nalpha = {alpha}: mean interval width vs contamination rate")
        piv = sub.pivot_table(index="method", columns="contamination", values="width")
        print(piv.to_string(float_format=lambda v: f"{v:.1f}"))

        base = piv[0.0]
        infl = piv.div(base, axis=0)
        print(f"\nalpha = {alpha}: width relative to the uncontaminated run"
              f" (1.00 = undamaged)")
        print(infl.to_string(float_format=lambda v: f"{v:.2f}"))

    print(
        "\nHow to read E2. Expect little damage while the contamination rate\n"
        "stays below alpha, and a sharp break once it approaches alpha, when the\n"
        "conformal quantile starts landing on corrupted scores. In our runs the\n"
        "scale-relative scores (codebook, scale-normalised CQR) break HARDER\n"
        "than residual CQR at that point, because a large multiplicative outlier\n"
        "on a small-magnitude window is a huge relative error but only a modest\n"
        "absolute one. Boundedness does not help: a 20x outlier still sits well\n"
        "inside a ~4000-bin codebook, so the score never saturates. This is a\n"
        "negative result for the robustness argument and should be reported as\n"
        "one."
    )


def main(seed):
    cal = load_split("cal", seed)
    test = load_split("test", seed)
    grid = build_grid(test)
    print(f"cal={len(cal['futures'])} test={len(test['futures'])} windows, "
          f"codebook L={grid.L}")

    df0 = experiment_scale_conditional(cal, test, grid)
    df0.to_csv(f"{OUT_PREFIX}_scale_conditional.csv", index=False)
    report_scale_conditional(df0)

    print("\nSplitting states by volume ...")
    df1 = experiment_cross_scale(cal, test, grid)
    df1.to_csv(f"{OUT_PREFIX}_cross_scale.csv", index=False)
    report_cross_scale(df1)

    df2 = experiment_contamination(cal, test, grid)
    df2.to_csv(f"{OUT_PREFIX}_contamination.csv", index=False)
    report_contamination(df2)

    print(f"\nWrote {OUT_PREFIX}_cross_scale.csv and {OUT_PREFIX}_contamination.csv")
    print(
        "\nRead the result honestly: if `CQR (scale-normalised)` matches the\n"
        "codebook rows in E1, then scale invariance -- not the tokenizer -- is\n"
        "doing the work, and the paper should say exactly that."
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    main(a.seed)
