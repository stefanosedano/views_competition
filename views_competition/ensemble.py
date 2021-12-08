"""Ensembling loops and functions."""

import os
import copy
import logging
import pandas as pd
import numpy as np

from views_competition import config, evaluate, datautils

log = logging.getLogger(__name__)
weights = {}


def get_simple_sc_ensemble(df, column_sets):
    """Builds a simple average ensemble series (task one)."""
    level = "cm" if "country_id" in df.index.names else "pgm"
    columns = [
        col
        for _, col in column_sets[1][level].items()
        if col not in config.DROPS_ENS_T1
    ]
    log.info(f"Ensembling t1 with {columns}.")
    ensemble = df[columns].mean(axis=1)
    ensemble.name = f"ensemble"

    return ensemble


def add_simple_ss_ensemble(df, column_sets):
    """Adds a simple average ensemble series by step to t2 data."""
    level = "cm" if "country_id" in df.index.names else "pgm"
    task = datautils.determine_task(df)
    drops = config.DROPS_ENS_T1 if task == 1 else config.DROPS_ENS_T2
    for step in range(2, 8):
        columns = [
            list(col.values())  # Column names per step.
            for team, col in column_sets[2][level].items()
            if team not in drops
        ]
        columns = [item for sublist in columns for item in sublist]
        step_selection = [col for col in columns if f"s{step}" in col]
        log.info(f"Ensembling t{task} for step {step} with {step_selection}.")
        df[f"ensemble_s{step}"] = df[step_selection].mean(axis=1)

    return df


def make_ensemble_weights(metric="MSE"):
    """Prepares the ensemble weights according to selected metric.

    Using evaluate.t2_scores, compute for each team_id-step the relevant
    weights per metric:

    {
        "cm": {
            "team_id": {
                2: 0.9,
                3: 0.7,
                ...
            }
        },
    }

    e.g. (1 / df[f"MSE_2"]) / (sum(1/df[f"MSE_2"]))

    N.B. Selected columns dropped before weights are computed. This
    affects the sum MSE in the denominator.
    """
    # Collect the step-metric sums first.
    scores = copy.deepcopy(evaluate.t2_scores)
    sums = {"cm": {}, "pgm": {}}
    for level, teams in scores.items():
        for t in teams:
            if t in config.DROPS_ENS_T1:
                log.debug("Dropped: %s", t)
        for step in range(2, 8):
            score_sum = sum(
                [
                    (1 / v[step][metric])
                    for t, v in teams.items()
                    if t not in config.DROPS_ENS_T1
                ]
            )
            sums[level][step] = score_sum
    # Then we can key the sums per team and step,
    # over a single metric in a copy of t2_scores.
    weights.update(scores)
    for level, teams in weights.items():
        for team_id, score_dict in teams.items():
            step_scores = {}
            for step in range(2, 8):
                score = score_dict[step][metric]
                metric_sum = sums[level][step]
                step_scores.update({step: (1 / score) / metric_sum})
            weights[level][team_id] = step_scores


def write_ensemble_weights(weights, out_path):
    """Writes ensemble weights to latex."""
    for level in ["cm", "pgm"]:
        weights_out = pd.DataFrame(weights[level]).T
        log.debug(
            "Dropped: {}".format(
                [
                    idx
                    for idx in weights_out.index
                    if idx in config.DROPS_ENS_T1
                ]
            )
        )
        weights_out = weights_out.drop(
            [idx for idx in weights_out.index if idx in config.DROPS_ENS_T1]
        )
        weights_out = weights_out.sort_index()
        weights_out.columns = [
            f"{month_id} (s={col})"
            for month_id, col in zip(range(490, 496), weights_out.columns)
        ]
        weights_out.to_csv(
            os.path.join(out_path, f"t1_{level}_ensemble_weights.csv")
        )
        evaluate.scores_to_tex(
            df=np.round(weights_out, 3),
            out_path=os.path.join(
                out_path, f"t1_{level}_ensemble_weights.tex"
            ),
        )


def weighted_t1_ensemble(df, column_sets):
    """Builds an ensemble using weights produced by make_ensemble_weights().

    For each row in t1 representing a step, collect the step-weight
    for each column (team). Then for that index, take a weighted average.
    Finally pd.Series the collected weighted averages.

    Runs over a groupby (groupvar).
    """
    level = "cm" if "country_id" in df.index.names else "pgm"
    columns = [
        col
        for _, col in column_sets[1][level].items()
        if col not in config.DROPS_ENS_T1
    ]
    df = df[columns]

    def _collect_wavg(g, steps, weights):
        wavgs = []
        steps = range(2, 8)
        for i, step in zip(g.index, steps):
            row_weights = []
            for team_id, col in column_sets[1][level].items():  # 1: t1
                if col not in config.DROPS_ENS_T1:
                    row_weights.append(weights[level][team_id][step])
            wavg = np.average(g.loc[i], weights=row_weights)
            wavgs.append(wavg)

        wavg_series = pd.Series(wavgs)
        wavg_series.name = f"w_ensemble"
        wavg_series = pd.DataFrame(wavg_series)
        wavg_series.index = g.index
        return wavg_series

    log.info(f"Ensembling t1_weighted with {list(df.columns)}")
    w_ens_t1 = df.groupby(level=1).apply(
        lambda g: _collect_wavg(g, range(2, 8), weights)
    )
    return w_ens_t1
