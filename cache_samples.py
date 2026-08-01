"""
Generate Chronos sample paths ONCE and cache them to disk.

Why this script exists
----------------------
The submission states that "sample paths are cached to disk after generation
and re-used across all conformal methods". That is true of the PTF/ChatTime
pipeline but NOT of the Chronos pipeline: `method_cqr.get_quantiles` and
`method_codebook_cqr.get_sample_bins` each call `pipeline.predict` inside the
alpha loop, so every (method, alpha) cell in results_table.csv was computed on
a different random draw of 100 sample paths.

The practical consequence is that part of the reported CQR vs Codebook-CQR gap
is sampling noise rather than a method difference, and none of it is
reproducible without a seed. This script fixes both: one draw, saved, shared.

It also draws several independent seeds so the rebuttal can report error bars
on coverage and width instead of point estimates.

Outputs
-------
    cache/chronos_{cal,test}_seed{S}.npz
        samples   (N, K, H)   float32   sample paths in value space
        futures   (N, H)      float32   ground truth
        scale     (N,)        float32   per-window Chronos mean-abs scale
        sample_bins (N, K, H) int16     re-tokenized sample bin IDs
        true_bins (N, H)      int16     tokenized ground truth
        state_idx (N,)        int32     which state each window came from

Usage
-----
    python cache_samples.py                 # seed 0 only
    python cache_samples.py --seeds 0 1 2 3 4
"""

import argparse
import os

import numpy as np
import torch
from chronos import BaseChronosPipeline

from covid_dataset import load_hosp_wide, split_wide, to_tensor_list, make_all_windows

MODEL_ID = "amazon/chronos-t5-small"
CONTEXT_LEN = 60
PRED_LEN = 14
STRIDE = 7
NUM_SAMPLES = 100
BATCH_SIZE = 32
CACHE_DIR = "cache"


def make_windows_with_state(tensors, context_len, pred_len, stride):
    """
    Same as covid_dataset.make_all_windows but also returns, for each window,
    the index of the state it came from.

    We need this because the calibration/test split is chronological but the
    windows are pooled across 50 states. Reviewers reasonably ask whether the
    exchangeability assumption is being propped up by pooling; keeping the
    state label lets us report per-state coverage without re-sampling.
    """
    from covid_dataset import make_windows

    ctxs, futs, sidx = [], [], []
    for s, series in enumerate(tensors):
        c, f = make_windows(series, context_len, pred_len, stride=stride)
        if len(c) == 0:
            continue
        ctxs.append(c)
        futs.append(f)
        sidx.append(np.full(len(c), s, dtype=np.int32))
    return torch.cat(ctxs), torch.cat(futs), np.concatenate(sidx)


@torch.no_grad()
def draw_and_tokenize(pipeline, contexts, futures, seed):
    """
    Draw NUM_SAMPLES paths per window and re-tokenize both the samples and the
    ground truth through the SAME per-window scale the model used.

    Everything the four conformal methods need is derived from these arrays, so
    no method ever calls the model again.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    tokenizer = pipeline.tokenizer
    n = contexts.shape[0]

    samples_all = np.empty((n, NUM_SAMPLES, PRED_LEN), dtype=np.float32)
    bins_all = np.empty((n, NUM_SAMPLES, PRED_LEN), dtype=np.int16)
    scale_all = np.empty(n, dtype=np.float32)

    for start in range(0, n, BATCH_SIZE):
        batch = contexts[start : start + BATCH_SIZE]

        _, _, scale = tokenizer.context_input_transform(batch)
        samples = pipeline.predict(
            batch, prediction_length=PRED_LEN, num_samples=NUM_SAMPLES
        )
        samples = samples.float().cpu()

        B, K, H = samples.shape
        flat = samples.reshape(B * K, H)
        scale_rep = scale.cpu().unsqueeze(1).expand(-1, K).reshape(-1)
        bins, _, _ = tokenizer._input_transform(flat, scale=scale_rep)
        bins = bins.reshape(B, K, H)

        samples_all[start : start + B] = samples.numpy()
        bins_all[start : start + B] = bins.numpy().astype(np.int16)
        scale_all[start : start + B] = scale.cpu().numpy()

        if (start // BATCH_SIZE) % 10 == 0:
            print(f"    {min(start + BATCH_SIZE, n)}/{n}")

    fut_t = torch.as_tensor(futures, dtype=torch.float32)
    scale_t = torch.as_tensor(scale_all, dtype=torch.float32)
    true_bins, _, _ = tokenizer._input_transform(fut_t, scale=scale_t)

    return samples_all, bins_all, scale_all, true_bins.numpy().astype(np.int16)


def main(seeds):
    os.makedirs(CACHE_DIR, exist_ok=True)

    print("Loading COVID data ...")
    wide = load_hosp_wide()
    _, cal_wide, test_wide = split_wide(wide)

    cal_ctx, cal_fut, cal_state = make_windows_with_state(
        to_tensor_list(cal_wide), CONTEXT_LEN, PRED_LEN, STRIDE
    )
    test_ctx, test_fut, test_state = make_windows_with_state(
        to_tensor_list(test_wide), CONTEXT_LEN, PRED_LEN, STRIDE
    )
    print(f"  cal windows : {cal_ctx.shape[0]}")
    print(f"  test windows: {test_ctx.shape[0]}")

    print(f"\nLoading {MODEL_ID} ...")
    pipeline = BaseChronosPipeline.from_pretrained(
        MODEL_ID, device_map="auto", torch_dtype=torch.float32
    )
    n_tokens = pipeline.tokenizer.config.n_tokens
    print(f"  codebook: {n_tokens} tokens")

    for seed in seeds:
        for split, ctx, fut, sidx in [
            ("cal", cal_ctx, cal_fut, cal_state),
            ("test", test_ctx, test_fut, test_state),
        ]:
            path = os.path.join(CACHE_DIR, f"chronos_{split}_seed{seed}.npz")
            if os.path.exists(path):
                print(f"  skip (exists): {path}")
                continue
            print(f"\n[seed {seed}] {split}")
            s, b, sc, tb = draw_and_tokenize(pipeline, ctx, fut.numpy(), seed)
            np.savez_compressed(
                path,
                samples=s,
                sample_bins=b,
                scale=sc,
                true_bins=tb,
                futures=fut.numpy().astype(np.float32),
                state_idx=sidx,
                n_tokens=np.int32(n_tokens),
                # Codebook centers and the special-token offset, so the
                # downstream analysis can invert the tokenizer without
                # reloading the model on a CPU-only box.
                centers=pipeline.tokenizer.centers.cpu().numpy().astype(np.float32),
                n_special=np.int32(pipeline.tokenizer.config.n_special_tokens),
            )
            print(f"  wrote {path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    args = p.parse_args()
    main(args.seeds)
