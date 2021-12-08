"""Data utilities"""

import os
import logging
import warnings
import pandas as pd
import numpy as np

from views_competition import DATA_DIR

log = logging.getLogger(__name__)


def prob_to_odds(p, clip=True):
    """ Cast probability into odds """

    if isinstance(p, list):
        p = np.array(p)

    if clip:
        offset = 1e-10
        offset = 1e-10
        upper = 1 - offset
        lower = 0 + offset
        p = np.clip(p, lower, upper)

    # Check for probs greq 1 because odds of 1 is inf which might break things
    if np.any(p >= 1):
        msg = "probs >= 1 passed to get_odds, expect infs"
        warnings.warn(msg)

    odds = p / (1 - p)
    return odds


def prob_to_logodds(p):
    """ Cast probability to log-odds """
    return np.log(prob_to_odds(p))


def mean_norm(s):
    return (s - s.mean()) / s.std()


def minmax_norm(s):
    return (s - s.min()) / (s.max() - s.min())


def determine_task(df):
    """Determines the task by looking at the month_id idx."""
    timeframes = {
        1: list(range(490, 496)),
        2: list(range(445, 481)),
        3: list(range(409, 445)),
    }
    task = ""
    for key, values in timeframes.items():
        if bool(set(values) & set(df.index.get_level_values(0))):
            task = key
            log.debug("Input data matches task %s.", key)
    if not task:
        raise RuntimeError("No task period found.")

    return task


def tlag(s: pd.Series, time: int) -> pd.Series:
    """ Time lag """
    if time < 0:
        msg = f"Time below 0 passed to tlag: {time} \n"
        msg += "Call tlead() instead \n"
        raise RuntimeError(msg)

    return s.groupby(level=1).shift(time)


def delta(s: pd.Series, time: int = 1) -> pd.Series:
    """ Return the time-delta of s """
    return s - tlag(s, time=time)


def determine_delta(s):
    """Determines whether outcome is a delta from column values."""
    if s.min() >= 0 and s.max() <= 1:
        if s.min() == 0 and s.max() == 0:
            log.info("Series looks like a null. [OK]")
        else:
            log.info("Series looks like a probability. [CHECK]")
    elif s.min() >= 0 and s.max() >= 1:
        log.info("Series 0 does not look like a delta. [CHECK]")
    else:
        log.info("Series looks like a delta. [OK]")


def add_delta_logtransforms(df, pre_patch=False):
    """Adds stepwise deltas of log-transformed ged_best_sb outcomes."""
    suffix = "_09" if pre_patch else ""
    for step in range(2, 8):
        df[f"d_ln_ged_best_sb{suffix}_s{step}"] = delta(
            np.log1p(df["ged_best_sb"]), step
        )
    df = df.drop(columns=["ged_best_sb"])

    return df


# TODO: function renames.
def add_t1_delta(df):
    """Adds t1 deltas of log-transformed ged_best_sb compared to 488."""

    def fn(g):
        constant = g.loc[488]
        return g - constant

    df["ln_ged_best_sb"] = np.log1p(df["ged_best_sb"])
    df["d_ln_ged_best_sb"] = df.groupby(level=1)["ln_ged_best_sb"].apply(fn)

    return df


def check_duplicates(df):
    """Check whether there are duplicate values in idx."""
    if df.index.duplicated().any():
        log.info("df.idx has duplicated values. [FIX]")
    else:
        log.info("No duplicated values in idx. [OK]")


def assign_into_df(df_to: pd.DataFrame, df_from: pd.DataFrame) -> pd.DataFrame:
    """Assign all columns from df_from into df_to

    Only assigns non-missing values from df_from, meaning the
    same column can be inserted multiple times and values be
    retained if the row coverage is different between calls.
    So a df_a with col_a covering months 100-110 and df_b with col_a covering
    months 111-120 could be assigned into a single df which would get
    values of col_a for months 100 - 120.
    """

    for col in df_from:
        # Get a Series of the col for all rows
        s = df_from.loc[:, col]

        # Get the "is not null" boolean series to use as mask, ~ is NOT
        mask = ~s.isnull()

        # Get the index from that mask,
        # ix is now index labels of rows with (not missing) data
        ix = s.loc[mask].index
        try:
            df_to.loc[ix, col] = s.loc[ix]
        # Handle KeyError when assigning into a df without all times or locations
        except KeyError:
            try:
                # Only assign ix positions that exist in df_to
                ix = pd.MultiIndex.from_tuples(
                    [tup for tup in ix if tup in df_to.index]
                )
                df_to.loc[ix, col] = s.loc[ix]
            # If it still doesn't work we've got no ix in common.
            except TypeError:
                raise TypeError("No index positions in common.")

    return df_to


def to_datestr(month_id):
    """Converts a month_id to date_str."""
    year, month = tuple(
        pd.read_parquet(os.path.join(DATA_DIR, "month.parquet")).loc[month_id]
    )
    return f"{year}-{month}"
