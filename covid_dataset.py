"""
COVID hospitalization dataset loader for Chronos conformal prediction.

Data source: data_ghobadk1/chi/Covid/covid_data/timeseries/hosp_daily_state.csv
Structure:   wide-format CSV, one row per (state, date)
             columns: state, fips, date, "Confirmed COVID-19 Hospital Admissions"
             values are daily new admissions (not cumulative)

Splits (time-based, never random for time series):
  train:       2020-07-01  to  2021-12-31   (for finetuning)
  calibration: 2022-01-01  to  2022-06-30   (for fitting conformal scores)
  test:        2022-07-01  to  end           (for evaluating coverage)
"""

import numpy as np
import pandas as pd
import torch
from typing import List, Tuple

HOSP_CSV = "../Covid/covid_data/timeseries/hosp_daily_state.csv"
HOSP_COL = "Confirmed COVID-19 Hospital Admissions"

TRAIN_END  = "2021-12-31"
CAL_START  = "2022-01-01"
CAL_END    = "2022-06-30"
TEST_START = "2022-07-01"


def load_hosp_wide() -> pd.DataFrame:
    """
    Load hosp_daily_state.csv and return a wide DataFrame:
      - rows:    dates (DatetimeIndex, daily frequency)
      - columns: state abbreviations (50 states), derived from 'state' column
      - values:  daily new hospital admissions (float, NaN where missing)

    The CSV has one row per (state, date) with a single target column.
    We pivot so each state becomes its own column, then reindex to a full
    daily calendar so date gaps become NaN (Chronos handles NaN natively).
    """
    df = pd.read_csv(HOSP_CSV, parse_dates=["date"])

    # use state abbreviation from 'fips' would work too, but 'state' is cleaner
    # the CSV 'state' column holds full state names; derive abbreviation from fips
    # actually the CSV has 'state' as full name — pivot on that, rename later if needed
    wide = df.pivot_table(index="date", columns="state", values=HOSP_COL, aggfunc="first")
    wide = wide.sort_index()

    full_dates = pd.date_range(wide.index.min(), wide.index.max(), freq="D")
    wide = wide.reindex(full_dates)

    return wide


def split_wide(wide: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split wide DataFrame into train / calibration / test by date.
    Returns three DataFrames with the same columns (states).
    """
    train = wide.loc[:TRAIN_END]
    cal   = wide.loc[CAL_START:CAL_END]
    test  = wide.loc[TEST_START:]
    return train, cal, test


def to_tensor_list(df: pd.DataFrame) -> List[torch.Tensor]:
    """
    Convert a wide DataFrame to a list of 1D float32 tensors, one per state.
    NaN values are preserved (Chronos treats them as missing observations).
    """
    tensors = []
    for state in df.columns:
        arr = df[state].values.astype(np.float32)
        tensors.append(torch.tensor(arr))
    return tensors


def make_windows(
    series: torch.Tensor,
    context_len: int,
    pred_len: int,
    stride: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Slide a window over a single time series and return all (context, future) pairs.

    Parameters
    ----------
    series      : 1D tensor of length T
    context_len : number of past steps fed to Chronos as input
    pred_len    : number of future steps to predict / evaluate
    stride      : step between window starts (use > 1 to reduce overlap)

    Returns
    -------
    contexts : (N, context_len)  — input windows
    futures  : (N, pred_len)     — ground truth future windows
    """
    T = len(series)
    starts = range(0, T - context_len - pred_len + 1, stride)
    contexts = torch.stack([series[i : i + context_len] for i in starts])
    futures  = torch.stack([series[i + context_len : i + context_len + pred_len] for i in starts])
    return contexts, futures


def make_all_windows(
    tensor_list: List[torch.Tensor],
    context_len: int,
    pred_len: int,
    stride: int = 7,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply make_windows to every state and concatenate across states.

    Returns
    -------
    contexts : (N_total, context_len)
    futures  : (N_total, pred_len)
    """
    all_ctx, all_fut = [], []
    for series in tensor_list:
        ctx, fut = make_windows(series, context_len, pred_len, stride)
        all_ctx.append(ctx)
        all_fut.append(fut)
    return torch.cat(all_ctx), torch.cat(all_fut)


if __name__ == "__main__":
    wide = load_hosp_wide()
    print(f"Wide DataFrame: {wide.shape}  (dates x states)")
    print(f"Date range:     {wide.index[0].date()} → {wide.index[-1].date()}")
    print(f"States:         {list(wide.columns)[:5]} ...")
    print(f"NaN fraction:   {wide.isna().mean().mean():.3f}")

    train, cal, test = split_wide(wide)
    print(f"\nTrain:  {train.index[0].date()} → {train.index[-1].date()}  ({len(train)} days)")
    print(f"Cal:    {cal.index[0].date()} → {cal.index[-1].date()}  ({len(cal)} days)")
    print(f"Test:   {test.index[0].date()} → {test.index[-1].date()}  ({len(test)} days)")

    cal_tensors = to_tensor_list(cal)
    print(f"\nCal tensor list: {len(cal_tensors)} states, each length {cal_tensors[0].shape[0]}")

    CONTEXT_LEN = 60
    PRED_LEN    = 14
    ctx, fut = make_all_windows(cal_tensors, CONTEXT_LEN, PRED_LEN, stride=7)
    print(f"\nCalibration windows:  contexts {tuple(ctx.shape)}, futures {tuple(fut.shape)}")
