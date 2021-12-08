"""Cleanup utilities"""

import os
import pandas as pd
import numpy as np
import warnings
import logging

log = logging.getLogger(__name__)


def reshape_t1(df, groupvar, team):
    """Reshape task1 df sent in with stepwise columns."""
    out = pd.DataFrame()
    for month_id, col in zip([i for i in range(490, 496)], df.columns):
        sub_df = pd.DataFrame(df.loc[month_id, col])
        sub_df.columns = [team]
        sub_df["month_id"] = month_id
        sub_df = sub_df.reset_index().set_index(["month_id", groupvar])
        out = out.append(sub_df)

    return out


def reshape_t1_alt(df, groupvar, team):
    """Reshape task1 df sent in with stepwise columns, one month for 488."""
    out = pd.DataFrame()
    for month_id, col in zip([i for i in range(490, 496)], df.columns):
        sub_df = df[col].reset_index()
        sub_df["month_id"] = month_id
        sub_df = sub_df.set_index(["month_id", groupvar])
        sub_df.columns = [team]
        out = out.append(sub_df)

    return out


def check_missing(df):
    """Check for missingness in df."""
    check = df.isnull().values.any()
    if check:
        number = df.isnull().sum().sum()
        log.warning(f"Found {number} NAs in df.")


def delta_columns_by_step(df, actuals):
    """Transform real columns to deltas vs observed, according to step."""
    # Get actuals, groupby-shifted according to step.
    pred_cols = df.columns
    for step, col in enumerate(pred_cols, 2):
        step_obs = (
            actuals["ln_ged_best_sb"].groupby(level=1).shift(step).fillna(0)
        )
        step_obs.name = f"ln_ged_best_sb_s{step}"
        df = df.merge(step_obs, left_index=True, right_index=True)
        df[col] = df[col] - df[f"ln_ged_best_sb_s{step}"]
    return df


def delta_columns_by_step_t1(df, actuals, team):
    """Transform task 1 columns to deltas vs observed, according to step."""
    groupvar = actuals.index.names[1]
    pred_col = df.columns[0]
    actuals = actuals.reset_index()
    actuals = actuals[actuals[groupvar].isin(list(df.index.levels[1]))]
    actuals = actuals.set_index(["month_id", groupvar]).sort_index()
    actuals = actuals.loc[488]
    df = df.reset_index().merge(actuals, on=groupvar)
    df[f"{team}_fixed"] = df[pred_col] - df["ln_ged_best_sb"]
    df = df.set_index(["month_id", groupvar]).sort_index()
    return df
