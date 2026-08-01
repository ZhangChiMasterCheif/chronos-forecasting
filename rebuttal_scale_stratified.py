"""
Scale-stratified re-reading of the SUBMITTED Chronos results.

Runs off `regime_data.npz`, which is already on disk. No model, no GPU, no
re-sampling -- this is purely a different cut through the numbers that are
already in the paper. Run it first; it produces the single most useful table
for the rebuttal.

What it shows
-------------
The COVID test pool spans roughly a 700x range in daily admissions, and the two
methods spend their width very differently across that range:

  * Residual CQR calibrates ONE absolute Q-hat, in admissions, and adds it to
    every window regardless of magnitude. It therefore over-covers small states
    and under-covers large ones. Its valid marginal coverage is the average of
    those two errors.
  * Codebook-CQR calibrates in bin units, after Chronos has divided out the
    per-window scale, so its correction is proportional to each window's own
    magnitude.

Two consequences matter for the rebuttal:

1. The metareview's "Codebook-CQR is substantially wider than residual CQR" is
   computed as a mean ABSOLUTE width over a pool spanning three orders of
   magnitude. That statistic is dominated by the largest series. Stratified, the
   codebook method is *narrower* than CQR on the small strata and wider only on
   the large strata -- where CQR is simultaneously undercovering.

2. CQR's pooled coverage hides a conditional failure on exactly the high-volume
   series that matter operationally for hospital capacity planning.

Neither point rescues the method as a general width improvement, and the
rebuttal should not claim it does. What they establish is that the comparison in
the submitted tables is not like-for-like, and that the two methods differ in
conditional validity rather than in raw efficiency.

Usage
-----
    python rebuttal_scale_stratified.py
"""

import numpy as np

NPZ = "regime_data.npz"
ALPHA = 0.10          # regime_data.npz was written at alpha = 0.10
N_STRATA = 10
METHODS = ["Naive", "CQR", "Codebook-CQR", "Embed-CQR"]


def main():
    d = np.load(NPZ)
    fut = d["futures"]
    n, H = fut.shape

    # Per-window magnitude proxy. The tokenizer scale is not stored in this
    # file, so we use the median of the realised target, which is monotone in
    # it and needs nothing beyond what is already saved.
    level = np.maximum(np.median(fut, axis=1), 1.0)
    edges = np.quantile(level, np.linspace(0, 1, N_STRATA + 1))[1:-1]
    stratum = np.clip(np.searchsorted(edges, level), 0, N_STRATA - 1)

    print(f"{n} test windows, horizon {H}, alpha = {ALPHA} "
          f"(nominal joint coverage {1 - ALPHA:.2f})")
    print(f"series level spans {level.min():.0f} to {level.max():.0f} "
          f"({level.max() / level.min():.0f}x)\n")

    print("=" * 92)
    print("Pooled -- what the submitted tables report")
    print("=" * 92)
    print(f"{'method':<15}{'per-step':>10}{'joint':>9}{'width':>10}"
          f"{'width/level':>13}{'Winkler':>10}")
    stats = {}
    for m in METHODS:
        lo, hi = d[m + "_lo"], d[m + "_hi"]
        cov = (fut >= lo) & (fut <= hi)
        w = (hi - lo).mean(axis=1)
        wink = ((hi - lo)
                + (2 / ALPHA) * np.maximum(lo - fut, 0)
                + (2 / ALPHA) * np.maximum(fut - hi, 0)).mean()
        stats[m] = (cov, w)
        print(f"{m:<15}{cov.mean():>10.3f}{cov.all(axis=1).mean():>9.3f}"
              f"{w.mean():>10.1f}{(w / level).mean():>13.3f}{wink:>10.1f}")

    print("\n" + "=" * 92)
    print("Stratified by series magnitude -- joint coverage")
    print("=" * 92)
    hdr = "".join(f"{i:>8}" for i in range(N_STRATA))
    print(f"{'stratum':<15}{hdr}")
    print(f"{'median level':<15}"
          + "".join(f"{np.median(level[stratum == i]):>8.0f}" for i in range(N_STRATA)))
    print("-" * 92)
    for m in METHODS:
        cov, _ = stats[m]
        joint = cov.all(axis=1)
        print(f"{m:<15}"
              + "".join(f"{joint[stratum == i].mean():>8.3f}" for i in range(N_STRATA)))

    print("\n" + "=" * 92)
    print("Stratified by series magnitude -- mean interval width")
    print("=" * 92)
    print(f"{'stratum':<15}{hdr}")
    print("-" * 92)
    for m in METHODS:
        _, w = stats[m]
        print(f"{m:<15}"
              + "".join(f"{w[stratum == i].mean():>8.0f}" for i in range(N_STRATA)))

    print("\n" + "=" * 92)
    print("Summary")
    print("=" * 92)
    for m in METHODS:
        cov, w = stats[m]
        joint = cov.all(axis=1)
        per_str = np.array([joint[stratum == i].mean() for i in range(N_STRATA)])
        bad = int((per_str < 1 - ALPHA).sum())
        print(f"{m:<15} worst-stratum joint coverage {per_str.min():.3f}  "
              f"| strata below nominal: {bad}/{N_STRATA}  "
              f"| width/level {(w / level).mean():.2f}")

    print(
        "\nRead this carefully before using it. It does NOT show that\n"
        "codebook-CQR is more efficient -- pooled Winkler still favours CQR.\n"
        "It shows that the pooled mean-width comparison in the submitted tables\n"
        "is not like-for-like across a 700x magnitude range, and that CQR's\n"
        "valid marginal coverage conceals a conditional failure on the largest\n"
        "series. Those are the defensible claims."
    )


if __name__ == "__main__":
    main()
