"""
Rebuttal analyses for the Chronos / COVID experiment.

Everything here runs off the cached sample paths written by cache_samples.py,
so this script is CPU-only and takes seconds. Run cache_samples.py first.

Analyses
--------
R1  Joint AND per-step coverage for every method, plus the Winkler interval
    score. The submission calibrates for joint coverage but reports per-step,
    which is why every conformal row looks like it overcovers by 8-10 points.
R2  Per-step-calibrated CQR and Bonferroni CQR, so the reported metric and the
    calibration target agree, and the "conformal is very conservative" reading
    goes away.
R3  Seed variability and bootstrap CIs.
R4  Sample-budget sweep K in {5, 10, 25, 50, 100} for naive self-consistency.
    This replaces the six-window anecdote in Section 5.6 with a real curve on
    all test windows, and is the single cheapest way to answer "did you just
    not draw enough samples?".
R5  Codebook clipping diagnostics -- how often the codebook boundary binds and
    how often the target is unreachable. Section 4.5 claims clipping "can only
    expand the intervals"; it truncates them, so this quantifies the exposure.
R6  Non-negativity. COVID admissions cannot be negative, but a large fraction
    of the reported lower bounds are. Re-reports width and Winkler after
    clipping the lower edge at zero.
R7  Per-state coverage spread, as an exchangeability check on the pooled
    50-state design.
R8  PID with a scale-corrected gain vs the submitted absolute gain of 0.005,
    which on this data leaves the controller effectively frozen.

Usage
-----
    python rebuttal_chronos.py --seeds 0 1 2 3 4
"""

import argparse
import os

import numpy as np
import pandas as pd

import conformal_lib as cl
import finite_k as fk

CACHE_DIR = "cache"
ALPHAS = [0.05, 0.10, 0.20, 0.30, 0.40]
OUT_CSV = "rebuttal_chronos_results.csv"


# ---------------------------------------------------------------------------
# Cache loading
# ---------------------------------------------------------------------------

def load_split(split, seed):
    path = os.path.join(CACHE_DIR, f"chronos_{split}_seed{seed}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} missing -- run `python cache_samples.py --seeds {seed}` first"
        )
    return np.load(path)


def make_bins_to_values(centers, n_special, scale):
    """
    Inverse Chronos tokenizer, vectorised over (N, H) bin arrays.

    Mirrors MeanScaleUniformBins.output_transform: subtract the special-token
    offset, clamp into the centers array, look up, multiply by the per-window
    scale.
    """
    def fn(bin_arr):
        idx = np.clip(
            np.asarray(bin_arr).astype(int) - int(n_special) - 1,
            0,
            len(centers) - 1,
        )
        return centers[idx] * scale[:, None]

    return fn


# ---------------------------------------------------------------------------
# Codebook-CQR from cached bins
# ---------------------------------------------------------------------------

def codebook_cqr(cal, test, alpha, fallback=False):
    """
    Codebook-distance CQR. With fallback=False this reproduces the submitted
    method (clip at the codebook edge). With fallback=True the overflow past
    the edge is applied in value space instead, which restores the coverage
    guarantee the submission claims.

    Also returns the clipping diagnostics for R5.
    """
    n_tokens = int(cal["n_tokens"])

    cal_bins = cal["sample_bins"].astype(np.float32)
    cal_lo = np.quantile(cal_bins, alpha / 2, axis=1)
    cal_hi = np.quantile(cal_bins, 1 - alpha / 2, axis=1)
    scores = np.maximum(cal_lo - cal["true_bins"], cal["true_bins"] - cal_hi).max(axis=1)
    q = cl.conformal_quantile(scores, alpha)

    test_bins = test["sample_bins"].astype(np.float32)
    t_lo = np.quantile(test_bins, alpha / 2, axis=1)
    t_hi = np.quantile(test_bins, 1 - alpha / 2, axis=1)

    b2v = make_bins_to_values(test["centers"], test["n_special"], test["scale"])
    diag = cl.clipping_diagnostics(t_lo - q, t_hi + q, test["true_bins"], n_tokens)

    if fallback:
        lo, hi = cl.codebook_cqr_with_fallback(t_lo, t_hi, q, n_tokens, b2v)
    else:
        lo = b2v(np.clip(t_lo - q, 0, n_tokens - 1))
        hi = b2v(np.clip(t_hi + q, 0, n_tokens - 1))
    return lo, hi, diag


# ---------------------------------------------------------------------------
# Method dispatch
# ---------------------------------------------------------------------------

def run_all_methods(cal, test, alpha):
    """Returns {method_name: (lo, hi)} for one alpha, all from cached samples."""
    cal_s, cal_f = cal["samples"], cal["futures"]
    test_s, test_f = test["samples"], test["futures"]

    out = {}
    lo, hi = cl.sample_quantiles(test_s, alpha)
    out["Naive"] = (lo, hi)

    # Finite-K corrected naive: widen the sample-quantile level so that a
    # PERFECTLY calibrated model would attain 1-alpha at this sample budget.
    # Without this row, part of the naive undercoverage we report is just
    # Monte Carlo quantile estimation rather than model miscalibration.
    lo, hi, _ = fk.corrected_interval(test_s, test_s.shape[1], alpha)
    out["Naive (finite-K corrected)"] = (lo, hi)

    out["CQR (joint)"] = cl.cqr_joint(cal_s, cal_f, test_s, alpha)
    out["CQR (per-step)"] = cl.cqr_perstep(cal_s, cal_f, test_s, alpha)
    out["CQR (Bonferroni)"] = cl.cqr_bonferroni(cal_s, cal_f, test_s, alpha)

    lo, hi, _ = codebook_cqr(cal, test, alpha, fallback=False)
    out["Codebook-CQR"] = (lo, hi)
    lo, hi, _ = codebook_cqr(cal, test, alpha, fallback=True)
    out["Codebook-CQR (fallback)"] = (lo, hi)

    out["ACI"] = cl.aci(cal_s, cal_f, test_s, test_f, alpha)
    out["NexCP"] = cl.nexcp(cal_s, cal_f, test_s, alpha)

    lo, hi, _ = cl.pid(cal_s, cal_f, test_s, test_f, alpha, gain_frac=0.0)
    out["PID (submitted gain)"] = (lo, hi)
    lo, hi, _ = cl.pid(cal_s, cal_f, test_s, test_f, alpha, gain_frac=0.05)
    out["PID (scaled gain)"] = (lo, hi)
    return out


# ---------------------------------------------------------------------------
# R1/R2/R6 : main table
# ---------------------------------------------------------------------------

def table_main(cal, test, seed):
    rows = []
    for alpha in ALPHAS:
        for name, (lo, hi) in run_all_methods(cal, test, alpha).items():
            m = cl.evaluate_intervals(lo, hi, test["futures"], alpha, nonneg=True)
            rows.append(
                dict(
                    seed=seed,
                    method=name,
                    alpha=alpha,
                    nominal=round(1 - alpha, 2),
                    oracle_finite_K=round(
                        fk.oracle_coverage(test["samples"].shape[1], alpha), 4),
                    nominal_attainable=(1 - alpha) <= fk.max_order_statistic_coverage(
                        test["samples"].shape[1]),
                    cov_perstep=m["coverage_perstep"],
                    cov_joint=m["coverage_joint"],
                    width=m["width"],
                    winkler=m["winkler"],
                    frac_lo_neg=m["frac_lo_negative"],
                    cov_perstep_nonneg=m["coverage_perstep_nonneg"],
                    width_nonneg=m["width_nonneg"],
                    winkler_nonneg=m["winkler_nonneg"],
                )
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# R4 : sample-budget sweep
# ---------------------------------------------------------------------------

def table_k_sweep(cal, test, ks=(5, 10, 25, 50, 100), alpha=0.10, n_rep=5, seed=0):
    """
    Subsample K of the 100 cached paths per window and recompute naive coverage
    and width.

    If undercoverage were a small-K artifact, coverage would climb toward the
    nominal level as K grows. If it is a location bias, coverage stays flat (or
    falls) while the intervals get narrower. This is the claim Section 5.6
    makes from six windows; here it is measured on every test window.
    """
    rng = np.random.default_rng(seed)
    s = test["samples"]
    fut = test["futures"]
    K_max = s.shape[1]

    rows = []
    for K in ks:
        for r in range(n_rep):
            idx = rng.choice(K_max, size=K, replace=False)
            sub = s[:, idx, :]
            lo, hi = cl.sample_quantiles(sub, alpha)
            m = cl.evaluate_intervals(lo, hi, fut, alpha)
            rows.append(
                dict(
                    K=K,
                    rep=r,
                    cov_perstep=m["coverage_perstep"],
                    cov_joint=m["coverage_joint"],
                    width=m["width"],
                    # What a PERFECTLY calibrated model would attain at this K.
                    # The observed-minus-oracle gap is the part attributable to
                    # the model rather than to Monte Carlo quantile estimation.
                    oracle_finite_K=fk.oracle_coverage(K, alpha),
                    gap_vs_oracle=m["coverage_perstep"] - fk.oracle_coverage(K, alpha),
                )
            )
    df = pd.DataFrame(rows)

    # Median bias vs sample spread, pooled over all windows -- the quantitative
    # version of the Section 5.6 argument.
    med = np.nanmedian(s, axis=1)
    bias = np.abs(fut - med)
    spread = np.nanstd(s, axis=1)
    ratio = float(np.nanmedian(bias / np.maximum(spread, 1e-9)))
    print(f"\n[R4] median |truth - sample median| / sample std = {ratio:.2f}")
    print("     (>1 means location bias dominates sampling spread, so raising")
    print("      K narrows the interval around the same wrong center)")
    return df


# ---------------------------------------------------------------------------
# R9 : endpoint conventions for codebook-CQR
# ---------------------------------------------------------------------------

def table_endpoints(cal, test):
    """
    Compare the four decodings of the inflated endpoint tokens on real data.

    "center" is what the submission does and carries only a token-space
    guarantee; "edge" is valid in value space but unbounded at the extreme
    tokens; "edge_clipped" is finite but valid only under a support assumption;
    "phi" is finite and unconditionally valid. See codebook_repair.py.

    On Chronos the interior half-bin gap is tiny (4094 bins over [-15, 15]), so
    the expectation is that center and phi agree closely and the difference is
    concentrated at the codebook edge. Measuring that, rather than asserting it,
    is the point.
    """
    import codebook_repair as cr

    grid = cr.BinGrid.from_chronos_arrays(
        centers=test["centers"],
        boundaries=np.concatenate(
            ([-1e20],
             (test["centers"][1:].astype(np.float64)
              + test["centers"][:-1].astype(np.float64)) / 2.0,
             [1e20])
        ),
        n_special=int(test["n_special"]),
    )
    sup = cr.support_violation_rate(test["futures"], test["scale"], grid)

    rows = []
    for alpha in ALPHAS:
        for ep in ("center", "edge", "edge_clipped", "phi"):
            lo, hi, info = cr.codebook_cqr(
                cal["samples"], cal["futures"], cal["scale"],
                test["samples"], test["scale"], grid, alpha, endpoint=ep,
            )
            fin = np.isfinite(hi - lo)
            m = cl.evaluate_intervals(
                np.where(fin, lo, -1e30), np.where(fin, hi, 1e30),
                test["futures"], alpha,
            )
            rows.append(dict(
                alpha=alpha, endpoint=ep,
                cov_joint=m["coverage_joint"], cov_perstep=m["coverage_perstep"],
                width_finite_only=float((hi - lo)[fin].mean()) if fin.any() else np.inf,
                frac_infinite=float(1 - fin.mean()),
                Q_hat_bins=info.get("Q_hat_bins", np.nan),
            ))
    return pd.DataFrame(rows), sup


# ---------------------------------------------------------------------------
# R5 : clipping diagnostics
# ---------------------------------------------------------------------------

def table_clipping(cal, test):
    rows = []
    for alpha in ALPHAS:
        _, _, diag = codebook_cqr(cal, test, alpha, fallback=False)
        rows.append(dict(alpha=alpha, **diag))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# R7 : per-state coverage
# ---------------------------------------------------------------------------

def table_per_state(cal, test, alpha=0.10):
    """
    Coverage per state under the pooled calibration. Wide dispersion means the
    pooled marginal guarantee is hiding systematic per-series failures, which
    is the standard objection to pooling windows across 50 series and calling
    them exchangeable.
    """
    methods = run_all_methods(cal, test, alpha)
    sidx = test["state_idx"]
    fut = test["futures"]

    rows = []
    for name, (lo, hi) in methods.items():
        per = []
        for s in np.unique(sidx):
            mask = sidx == s
            cov = (fut[mask] >= lo[mask]) & (fut[mask] <= hi[mask])
            per.append(cov.mean())
        per = np.asarray(per)
        rows.append(
            dict(
                method=name,
                alpha=alpha,
                mean=per.mean(),
                min=per.min(),
                p10=np.percentile(per, 10),
                p90=np.percentile(per, 90),
                max=per.max(),
                n_states_below_nominal=int((per < 1 - alpha).sum()),
                n_states=len(per),
            )
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# R8 : PID controller trace
# ---------------------------------------------------------------------------

def table_pid_trace(cal, test, alpha=0.10):
    """
    Show that the submitted absolute gain of 0.005 leaves q essentially at its
    warm start, so "conformal PID" was in effect a fixed-width baseline.
    """
    rows = []
    for label, gf in [("submitted (0.005 abs)", 0.0), ("scaled (5% of q0)", 0.05)]:
        lo, hi, q = cl.pid(
            cal["samples"], cal["futures"], test["samples"], test["futures"],
            alpha, gain_frac=gf,
        )
        if gf == 0.0:
            # replay the literal submitted update so the trace is faithful
            q = _submitted_pid_trace(cal, test, alpha, k_i=0.005)
        m = cl.evaluate_intervals(lo, hi, test["futures"], alpha)
        rows.append(
            dict(
                variant=label,
                q_start=q[0],
                q_end=q[-1],
                q_min=q.min(),
                q_max=q.max(),
                total_drift=abs(q[-1] - q[0]),
                cov_joint=m["coverage_joint"],
                width=m["width"],
            )
        )
    return pd.DataFrame(rows)


def _submitted_pid_trace(cal, test, alpha, k_i=0.005):
    med_cal = np.nanmedian(cal["samples"], axis=1)
    q = float(np.nanquantile(np.abs(cal["futures"] - med_cal).max(axis=1), 1 - alpha))
    med_test = np.nanmedian(test["samples"], axis=1)
    fut = test["futures"]
    trace = np.empty(len(med_test))
    for t in range(len(med_test)):
        trace[t] = q
        covered = bool(np.all((fut[t] >= med_test[t] - q) & (fut[t] <= med_test[t] + q)))
        q = float(np.clip(q - k_i * (alpha - (0.0 if covered else 1.0)), 0.0, None))
    return trace


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(seeds):
    all_main = []
    for seed in seeds:
        cal = load_split("cal", seed)
        test = load_split("test", seed)
        print(f"\n[seed {seed}] cal={len(cal['futures'])} test={len(test['futures'])}")
        all_main.append(table_main(cal, test, seed))

    df = pd.concat(all_main, ignore_index=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_CSV}")

    print("\n" + "=" * 78)
    print("[R1/R2] joint vs per-step coverage, averaged over seeds")
    print("=" * 78)
    agg = (
        df.groupby(["method", "alpha"])
        .agg(
            cov_perstep=("cov_perstep", "mean"),
            cov_joint=("cov_joint", "mean"),
            cov_joint_sd=("cov_joint", "std"),
            width=("width", "mean"),
            width_sd=("width", "std"),
            winkler=("winkler", "mean"),
        )
        .reset_index()
    )
    print(agg.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print("\n" + "=" * 78)
    print("[R6] after clipping the lower bound at zero (admissions are non-negative)")
    print("=" * 78)
    nn = (
        df[df.alpha == 0.10]
        .groupby("method")[["frac_lo_neg", "width", "width_nonneg",
                            "winkler", "winkler_nonneg"]]
        .mean()
        .reset_index()
    )
    print(nn.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    cal0, test0 = load_split("cal", seeds[0]), load_split("test", seeds[0])

    print("\n" + "=" * 78)
    print("[R4] naive self-consistency vs sample budget K (alpha=0.10)")
    print("=" * 78)
    ks = table_k_sweep(cal0, test0)
    print(
        ks.groupby("K")[["cov_perstep", "oracle_finite_K", "gap_vs_oracle",
                         "cov_joint", "width"]]
        .mean()
        .to_string(float_format=lambda v: f"{v:.4f}")
    )
    print("\n  cov_perstep    = observed naive coverage")
    print("  oracle_finite_K= what a perfectly calibrated model attains at this K")
    print("  gap_vs_oracle  = the part attributable to the MODEL")
    print("  A gap that stays roughly constant as K grows is model miscalibration;")
    print("  a gap that shrinks toward zero was Monte Carlo estimation all along.")
    ks.to_csv("rebuttal_chronos_ksweep.csv", index=False)

    print("\n" + "=" * 78)
    print("[R9] endpoint conventions for codebook-CQR")
    print("=" * 78)
    ep_df, sup = table_endpoints(cal0, test0)
    print(ep_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nsupport (targets outside the representable codebook range):")
    for k, v in sup.items():
        print(f"  {k}: {v:.5f}")
    ep_df.to_csv("rebuttal_chronos_endpoints.csv", index=False)

    print("\n" + "=" * 78)
    print("[R5] codebook clipping diagnostics")
    print("=" * 78)
    cd = table_clipping(cal0, test0)
    print(cd.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    cd.to_csv("rebuttal_chronos_clipping.csv", index=False)

    print("\n" + "=" * 78)
    print("[R7] per-state coverage spread (alpha=0.10)")
    print("=" * 78)
    ps = table_per_state(cal0, test0)
    print(ps.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    ps.to_csv("rebuttal_chronos_per_state.csv", index=False)

    print("\n" + "=" * 78)
    print("[R8] PID controller: submitted gain vs scale-corrected gain")
    print("=" * 78)
    pt = table_pid_trace(cal0, test0)
    print(pt.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    pt.to_csv("rebuttal_chronos_pid.csv", index=False)

    print("\n" + "=" * 78)
    print("[R3] bootstrap CIs on joint coverage (alpha=0.10, seed 0)")
    print("=" * 78)
    for name, (lo, hi) in run_all_methods(cal0, test0, 0.10).items():
        pt_, l, h = cl.bootstrap_ci(lo, hi, test0["futures"], 0.10,
                                    n_boot=500, stat="coverage_joint")
        w, wl, wh = cl.bootstrap_ci(lo, hi, test0["futures"], 0.10,
                                    n_boot=500, stat="width")
        print(f"  {name:<26} joint {pt_:.3f} [{l:.3f}, {h:.3f}]   "
              f"width {w:.1f} [{wl:.1f}, {wh:.1f}]")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    args = p.parse_args()
    main(args.seeds)
