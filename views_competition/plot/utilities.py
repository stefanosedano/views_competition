"""Plotting utilities"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from views_competition import config, evaluate, PICKLE_DIR


def collect_scores(level, steps, set_metrics=None):
    """Collects scores in a dataframe format by task and step."""
    # TODO: Too layered. Would be avoided if t2_scores[level][step]...
    collection = {}
    # Task one and two ss.
    for task in [1, 2]:
        collection[f"t{task}_ss"] = {}
        scorecollect = (
            evaluate.t1_ss_scores if task == 1 else evaluate.t2_scores
        )
        for step in steps:
            scores = {}
            for team_id, score_dict in scorecollect[level].items():
                if (
                    team_id not in config.DROPS_DEFAULT
                ):  # NOTE: Keep bench/no_change.
                    # Note metrics argument.
                    metrics = (
                        score_dict[step].keys()
                        if set_metrics is None
                        else set_metrics
                    )
                    # # TODO: make pemdiv for t1 pgm?
                    # if level == "pgm" and task == 1:
                    #     metrics = [m for m in metrics if "PEMDIV" not in m]
                    scores[team_id] = {}
                    for metric in metrics:
                        try:
                            team_score = {metric: score_dict[step][metric]}
                        except:
                            team_score = {
                                metric: np.nan
                            }  # Inapplicable ensemble scores.
                        scores[team_id].update(team_score)
            collection[f"t{task}_ss"][step] = pd.DataFrame(
                scores
            ).T.sort_index()
    # Task one sc.
    scores = {}
    collection["t1_sc"] = {}
    for team_id, score_dict in evaluate.t1_sc_scores[level].items():
        if team_id not in config.DROPS_DEFAULT:  # NOTE: Keep bench/no_change.
            metrics = score_dict.keys() if set_metrics is None else set_metrics
            # metrics = [m for m in metrics if "PEMDIV" not in m]  # Unavailable.
            scores[team_id] = {}
            for metric in metrics:
                try:
                    team_score = {metric: score_dict[metric]}
                except:
                    team_score = {
                        metric: np.nan
                    }  # Inapplicable ensemble scores.
                scores[team_id].update(team_score)
    collection["t1_sc"]["sc"] = pd.DataFrame(scores).T.sort_index()
    return collection


def get_team_colors():
    """Sets up colorscheme by team."""
    randahl = [
        "randahl_hhmm_weighted",
        "randahl_hmm_weighted",
        "randahl_vmm_weighted",
    ]
    vestby = [
        "vestby_rf_fit",
        "vestby_xgb_fit",
    ]
    no_change = ["no_change_cm", "no_change_pgm"]
    team_colors = {}
    cmap = plt.cm.tab20.colors
    # First, create colors for all by level.
    for level in ["cm", "pgm"]:
        team_colors[level] = {
            col: cmap[i]
            for i, col in enumerate(
                [
                    col
                    for col in sorted(config.COLUMN_SETS[2][level].keys())
                    if col
                    not in randahl
                    + vestby
                    + no_change
                    + ["chadefaux", "dorazio", "benchmark"]
                ]
            )
        }
        team_colors[level]["benchmark"] = (0, 0, 0)
        team_colors[level][f"no_change_{level}"] = (0.5, 0.5, 0.5)
        add_index = len(team_colors[level]) + 1
        # Let chadefaux, dorazio and ensemble share the same assigned color.
        if level == "cm":
            for col in randahl:
                team_colors[level][col] = cmap[add_index + 1]
            team_colors[level]["chadefaux"] = cmap[add_index + 2]
            team_colors[level]["dorazio"] = cmap[add_index + 3]
            team_colors[level]["ensemble"] = cmap[add_index + 4]
            team_colors[level]["w_ensemble"] = cmap[add_index + 4]
        else:
            for col in vestby:
                team_colors[level][col] = cmap[add_index + 1]
            team_colors[level]["dorazio"] = team_colors["cm"]["dorazio"]
            team_colors[level]["chadefaux"] = team_colors["cm"]["chadefaux"]
            team_colors[level]["ensemble"] = team_colors["cm"]["ensemble"]
            team_colors[level]["w_ensemble"] = team_colors["cm"]["w_ensemble"]

    return team_colors
