"""Evaluation module"""

# TODO: refactor with t2 sc.
# TODO: add debug logs.

import os
import logging
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn import metrics

from views_competition import config, evallib, datautils, OUTPUT_DIR

np.seterr(divide="ignore", invalid="ignore")  # Ignore division by zero.
log = logging.getLogger(__name__)

METRICS = {
    "MSE": metrics.mean_squared_error,
    "MSE_nonzero": metrics.mean_squared_error,
    "MSE_zero": metrics.mean_squared_error,
    "MSE_negative": metrics.mean_squared_error,
    "MSE_positive": metrics.mean_squared_error,
    "CCC": evallib.concordance_correlation_coefficient,
    # "R2": evallib.r2_score,
    "TADDA_1": lambda obs, pred: evallib.tadda_score(obs, pred, 1),
    "TADDA_1_nonzero": lambda obs, pred: evallib.tadda_score(obs, pred, 1),
    "TADDA_2": lambda obs, pred: evallib.tadda_score(obs, pred, 2),
    "TADDA_2_nonzero": lambda obs, pred: evallib.tadda_score(obs, pred, 2),
    # "cal_m": lambda obs, pred: pred.mean() / obs.mean(),
    "cal_m": lambda obs, pred: (pred.mean() - obs.mean()) ** 2,
    # "cal_sd": lambda obs, pred: pred.std() / obs.std(),
    "cal_sd": lambda obs, pred: (pred.std() - obs.std()) ** 2,
    "corr": lambda obs, pred: pred.corr(obs),
    "corr_negative": lambda obs, pred: pred.corr(obs),
    "corr_positive": lambda obs, pred: pred.corr(obs),
}
PEMDIV_PATH = os.path.join(OUTPUT_DIR, "data")

# Prepare a dictionary to store all stepwise T2 and T3 scores.
t2_scores = {"cm": {}, "pgm": {}}
t3_scores = {"cm": {}, "pgm": {}}
# Prepare a dictionary to store all sc T1 scores.
t1_sc_scores = {"cm": {}, "pgm": {}}
# Prepare a dictionary to store all stepwise T1 scores.
t1_ss_scores = {"cm": {}, "pgm": {}}


def preprocess(df, metric, col_obs, col, epsilon=0):
    """Preprocess obs, preds according to metric."""
    if "_nonzero" in metric:
        obs = df.loc[df[col_obs] != 0, col_obs]
        preds = df.loc[df[col_obs] != 0, col]
    elif "_zero" in metric:
        obs = df.loc[df[col_obs] == 0, col_obs]
        preds = df.loc[df[col_obs] == 0, col]
    elif "_negative" in metric:
        obs = df.loc[df[col_obs] < 0, col_obs]
        preds = df.loc[df[col_obs] < 0, col]
    elif "_positive" in metric:
        obs = df.loc[df[col_obs] > 0, col_obs]
        preds = df.loc[df[col_obs] > 0, col]
    elif "_epsilon" in metric:
        obs = df.loc[df[col_obs] == epsilon, col_obs]
        preds = df.loc[df[col_obs] == epsilon, col]
    else:
        obs = df[col_obs]
        preds = df[col]

    return obs, preds


def scores_to_tex(df, out_path):
    """Write table of scores to tex."""
    tex = df.to_latex()
    now = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    meta = f"""
    %Produced on {now}, written to {out_path}.
    \\
    """
    tex = meta + tex
    with open(out_path, "w") as f:
        f.write(tex)
    log.info(f"Wrote {out_path}.")


def compute_scores(df, column_sets):
    """Build score dict from submission df.

    The data gets structured (level > metric > team > stepcol > score) for ss:

    {
        "cm": {
            "team_id": {
                1: {MSE: 0.3, TADDA_1: 0.2, ...},
                3: {...},
            }
        }
    }

    The step layer is dropped for t1_sc.
    """
    level = "cm" if "country_id" in df.index.names else "pgm"
    log.debug("Computing for level %s.", level)
    # Two procedures: sc for t1, and ss for t1 and t2.
    # Distinguish sc and ss versions of task one first.
    t1_ss = True if "month_id" not in df.index.names else False
    task = 2 if t1_ss else datautils.determine_task(df)  # Group t1_ss with t2.
    # For task1, get team_id:col pairs.
    if task == 1:
        # Produce the step-combined (sc) scores.
        for team_id, col in column_sets[task][level].items():
            t1_sc_scores[level][team_id] = {}
            for metric, function in METRICS.items():
                col_obs = config.COL_OBS_T1
                obs, preds = preprocess(df, metric, col_obs, col)
                t1_sc_scores[level][team_id].update(
                    {metric: function(obs, preds)}
                )
    # For task2, get column sets (step:col) by team_id.
    if task == 2:
        # Grouped with t1_ss since it's the same procedure.
        drops = config.DROPS_ENS_T1 if t1_ss else config.DROPS_ENS_T2
        # TODO: including t1_ss here is confusing!
        collection = t1_ss_scores if t1_ss else t2_scores
        for team_id, columns in column_sets[task][level].items():
            scores = {}
            collection[level][team_id] = {}
            # Then collect by step and metric:
            for step, col in columns.items():
                scores[step] = {}
                for metric, function in METRICS.items():
                    # Use pre-patch actuals to evaluate the t2 benchmark.
                    if t1_ss or team_id != "benchmark":
                        col_obs = config.COL_OBS.format(step)
                    else:
                        col_obs = config.COL_OBS_09.format(step)
                    obs, preds = preprocess(df, metric, col_obs, col)
                    scores[step].update({metric: function(obs, preds)})
            collection[level][team_id] = scores
    if task == 3:
        # Grouped with t1_ss since it's the same procedure.
        drops = config.DROPS_ENS_T2
        collection = t3_scores
        for team_id, columns in column_sets[task][level].items():
            scores = {}
            collection[level][team_id] = {}
            # Then collect by step and metric:
            for step, col in columns.items():
                scores[step] = {}
                for metric, function in METRICS.items():
                    # Use pre-patch actuals to evaluate the t2 benchmark.
                    if team_id != "benchmark":
                        col_obs = config.COL_OBS.format(step)
                    else:
                        col_obs = config.COL_OBS_09.format(step)
                    obs, preds = preprocess(df, metric, col_obs, col)
                    scores[step].update({metric: function(obs, preds)})
            collection[level][team_id] = scores


def build_divstats(df, column_sets):
    """Adds diversity scores."""
    level = "cm" if "country_id" in df.index.names else "pgm"
    # Two procedures: sc for t1, and ss for t1 and t2.
    # Distinguish sc and ss versions of task one first.
    ss = True if "month_id" not in df.index.names else False
    task = 2 if ss else datautils.determine_task(df)  # Group t1_ss with t2.
    # For step-combined task1, get team_id:col pairs.
    if task == 1:
        for team_id, col in column_sets[task][level].items():
            columns = [
                col
                for team, col in column_sets[1][level].items()
                if team not in config.DROPS_ENS_T1
            ]
            avg = df[columns].mean(axis=1)
            squared_errors = (df[col] - avg) ** 2
            t1_sc_scores[level][team_id].update({"DIV": squared_errors.mean()})
    # For ss task2 and task1, get column sets (step:col) by team_id.
    if task == 2:
        # Grouped with t1_ss since it's the same procedure.
        drops = config.DROPS_ENS_T1 if ss else config.DROPS_ENS_T2
        collection = t1_ss_scores if ss else t2_scores
        for team_id, columns in column_sets[task][level].items():
            for step, col in columns.items():
                # col_ens = f"ens_{level}_t2_s{step}" # Either use official ens.
                # Or rebuild avg ensemble with different selected columns (per nb).
                # TODO: what's the difference here?
                columns = [
                    list(col.values())  # Column names per step.
                    for team, col in column_sets[2][level].items()
                    if team not in drops
                ]
                columns = [item for sublist in columns for item in sublist]
                step_selection = [col for col in columns if f"s{step}" in col]
                # log.info(f"Ensembling div t2 for step {step} with {step_selection}.")
                avg = df[step_selection].mean(axis=1)
                squared_errors = (df[col] - avg) ** 2
                collection[level][team_id][step].update(
                    {"DIV": squared_errors.mean()}
                )


def write_ss_scores(out_path):
    """Write step-specific scores to tex.

    Note: hardcoded for steps 2 and 7. Only task 2.
    """
    for level in ["cm", "pgm"]:
        for metric in [
            "MSE",
            "TADDA_1",
            "TADDA_2",
            "MSE_nonzero",
            "TADDA_1_nonzero",
            "TADDA_2_nonzero",
            "CCC",
            # "R2",
            "DIV",
            "MAL_MSE",
            "MAL_TADDA_1",
            "MAL_TADDA_2",
            "MAL_DIV",
        ]:
            # Collect relevant stepwise columns from collected scores.
            task_dfs = []
            for task in [1, 2]:
                scores = {}
                drops = config.DROPS_DEFAULT
                # Keep ensembles in for non DIV/MAL metrics.
                if metric in (
                    "MAL_MSE",
                    "MAL_TADDA_1",
                    "MAL_TADDA_2",
                    "MAL_DIV",
                    "DIV",
                ):
                    drops = drops + [
                        "ensemble",
                        "w_ensemble",
                    ]  # TODO: Solve this dropping confusion.
                collection = t1_ss_scores if task == 1 else t2_scores
                for team_id, score_dict in collection[level].items():
                    if team_id not in drops:
                        team_scores = {
                            team_id: {
                                f"{metric}_{step}": score_dict[step][metric]
                                for step in range(2, 8)
                            }
                        }
                        scores.update(team_scores)
                df = pd.DataFrame(scores).T.sort_index()
                df.to_csv(
                    os.path.join(
                        out_path,
                        f"t{task}_{level}_{metric.lower()}.csv",
                    ),
                )
                df = np.round(df, 3)
                # Ugly reindex to put benchmark and null on top.
                nochangecol = [
                    col for col in list(df.index) if "no_change" in col
                ]
                first = ["benchmark"] + nochangecol
                df = df.sort_index().reindex(
                    (
                        first
                        + [col for col in list(df.index) if col not in first]
                    )
                )
                # Write to tex.
                scores_to_tex(
                    df=df,
                    out_path=os.path.join(
                        out_path,
                        f"t{task}_{level}_{metric.lower()}.tex",
                    ),
                )


# TODO: Candidate for removal. Or combine with write_ss_scores.
def write_ss_scores_combined(out_path):
    """Write step-specific scores to tex.

    Note: hardcoded for steps 2 and 7. Only task 2.
    """
    for level in ["cm", "pgm"]:
        for metric in [
            "MSE",
            "TADDA_1",
            "TADDA_2",
            "MSE_nonzero",
            "TADDA_1_nonzero",
            "TADDA_2_nonzero",
            "CCC",
            # "R2",
            "DIV",
            "MAL_MSE",
            "MAL_TADDA_1",
            "MAL_TADDA_2",
            "MAL_DIV",

        ]:
            # Collect relevant stepwise columns from collected scores.
            task_dfs = []
            for task in [1, 2]:
                scores = {}
                drops = config.DROPS_ENS_T2
                # Keep ensembles in for non DIV/MAL metrics.
                if metric in (
                    "MAL_MSE",
                    "MAL_TADDA_1",
                    "MAL_TADDA_2",
                    "MAL_DIV",
                    "DIV",

                ):
                    drops = drops + [
                        "ensemble",
                        "w_ensemble",
                    ]  # TODO: Solve this dropping confusion.
                collection = t1_ss_scores if task == 1 else t2_scores
                for team_id, score_dict in collection[level].items():
                    if team_id not in drops:
                        team_scores = {
                            team_id: {
                                f"{metric}_2": score_dict[2][metric],
                                f"{metric}_7": score_dict[7][metric],
                            }
                        }
                        scores.update(team_scores)
                df = pd.DataFrame(scores).T.sort_index()
                df.columns = [f"t{task}_{col}" for col in df]
                task_dfs.append(df)
            df = pd.concat(task_dfs, axis=1)
            df = np.round(df, 3)
            # Ugly reindex to put benchmark and null on top.
            nochangecol = [col for col in list(df.index) if "no_change" in col]
            first = ["benchmark"] + nochangecol
            df = df.sort_index().reindex(
                (first + [col for col in list(df.index) if col not in first])
            )
            # Write to tex.
            scores_to_tex(
                df=df,
                out_path=os.path.join(
                    out_path,
                    f"{level}_{metric.lower()}.tex",
                ),
            )


def write_calibstats(out_path):
    """Write calibration stats to tex.
    Note: hardcoded for steps 2 and 7. Only task 2.
    """
    for level in ["cm", "pgm"]:
        scores = {}
        # Collect relevant stepwise columns from t2_scores.
        for team_id, score_dict in t2_scores[level].items():
            if team_id not in config.DROPS_ENS_T2:  # Drop redundancy.
                team_scores = {
                    team_id: {
                        "cal_m_2": score_dict[2]["cal_m"],
                        "cal_m_7": score_dict[7]["cal_m"],
                        "cal_sd_2": score_dict[2]["cal_sd"],
                        "cal_sd_7": score_dict[7]["cal_sd"],
                    }
                }
                scores.update(team_scores)

        df = pd.DataFrame(scores).T.sort_index()
        df = np.round(df, 3)
        # Ugly reindex to put benchmark and null on top.
        nochangecol = [col for col in list(df.index) if "no_change" in col]
        first = ["benchmark"] + nochangecol
        df = df.reindex(
            (first + [col for col in list(df.index) if col not in first])
        )
        # Write to tex.
        scores_to_tex(
            df=df,
            out_path=os.path.join(out_path, f"{level}_calib_stats.tex"),
        )


def scores_to_csv(scores, out_path):
    """Writes selected metrics from t2_scores to csv."""
    for metric in [
        "MSE",
        "TADDA_1",
        "TADDA_2",
        "MSE_nonzero",
        "TADDA_1_nonzero",
        "TADDA_2_nonzero",
        "CCC",
        # "R2",
        "DIV",
        "MAL_MSE",
        "MAL_TADDA_1",
        "MAL_TADDA_2",
        "MAL_DIV",

    ]:
        if metric in ("MSE", "TADDA_1", "TADDA_2"):
            drops = config.DROPS_ENS_T2[:-1]  # Drop ensemble from list.
        else:
            drops = config.DROPS_ENS_T2
        for level, teams in scores.items():
            relscores = {}
            for team, v in teams.items():
                if team not in drops:  # Drop redundancy.
                    subdf = {
                        team: {
                            f"{metric}_{step}": v[step][metric]
                            for step in range(2, 8)
                        }
                    }
                    relscores.update(subdf)

            pd.DataFrame(relscores).T.sort_index().to_csv(
                os.path.join(out_path, f"t2_{level}_{metric.lower()}.csv")
            )


def compute_cross_diversity(df):
    """Computes diversity given a stepwise selection of teams."""
    diversity = {}
    avg = df.mean(axis=1)
    for col in df:
        col_sqe = (df[col] - avg) ** 2
        diversity[col] = col_sqe.mean()

    return np.mean(list(diversity.values()))


def evaluate_t1_sc_ensemble(df, ensemble):
    """Evaluates t1 sc ensemble predictions and adds to collected scores."""
    level = "cm" if "country_id" in df.index.names else "pgm"
    # Join in on our t1 frame and take obs, preds.
    df = df.join(ensemble)
    obs = df[config.COL_OBS_T1]
    ens_col = pd.DataFrame(ensemble).columns[0]  # TODO: have both be Series.
    preds = df[ens_col]
    # Compute selected metrics.
    t1_sc_scores[level][ens_col] = {}
    for metric, function in METRICS.items():
        score = function(obs, preds)
        t1_sc_scores[level][ens_col].update({metric: score})


def evaluate_t1_ss_ensemble(df, ensemble, weighted=False):
    """Evaluates t1 ss ensemble predictions and adds to collected scores."""
    level = "cm" if "country_id" in df.index.names else "pgm"
    prefix = "w_" if weighted else ""
    # Join in on our t1 frame and take obs, preds.
    df = df.join(ensemble)
    scores = {}
    t1_ss_scores[level][f"{prefix}ensemble"] = {}
    # Then collect by step and metric:
    for step in range(2, 8):
        col_obs = config.COL_OBS.format(step)
        col = f"{prefix}ensemble_s{step}"
        scores[step] = {}
        for metric, function in METRICS.items():
            obs, preds = preprocess(df, metric, col_obs, col)
            scores[step].update({metric: function(obs, preds)})
    t1_ss_scores[level][f"{prefix}ensemble"] = scores


def evaluate_t2_ensemble(df):
    """Evaluates t2 ss ensemble predictions and adds to collected scores."""
    level = "cm" if "country_id" in df.index.names else "pgm"
    scores = {}
    t2_scores[level][f"ensemble"] = {}
    # Then collect by step and metric:
    for step in range(2, 8):
        col_obs = config.COL_OBS.format(step)
        col = f"ensemble_s{step}"
        scores[step] = {}
        for metric, function in METRICS.items():
            obs, preds = preprocess(df, metric, col_obs, col)
            scores[step].update({metric: function(obs, preds)})
    t2_scores[level][f"ensemble"] = scores


def ablation_study(df, column_sets):
    """Performs ablation study over steps, for metrics + diversity."""
    level = "cm" if "country_id" in df.index.names else "pgm"
    # Two procedures: sc for t1, and ss for t1 and t2.
    # Distinguish sc and ss versions of task one first.
    ss = True if "month_id" not in df.index.names else False
    task = 2 if ss else datautils.determine_task(df)  # Group t1_ss with t2.
    log.info("Running ablation for task %s, level %s.", task, level)
    # For task one, get team:col.
    if task == 1:
        for team_id, col in column_sets[task][level].items():
            selection = [
                col
                for team, col in column_sets[task][level].items()
                if team not in config.DROPS_ENS_T1
            ]
            ablated_selection = [i for i in selection if i != col]
            ablated_ensemble = df[ablated_selection].mean(axis=1)
            full_ensemble = df[selection].mean(axis=1)
            obs = df[config.COL_OBS_T1]
            # Regular obs, preds scorers.
            for metric, function in {
                "MSE": METRICS["MSE"],
                "TADDA_1": METRICS["TADDA_1"],
                "TADDA_2": METRICS["TADDA_2"],

            }.items():
                ablated_score = function(obs, ablated_ensemble)
                ensemble_score = function(obs, full_ensemble)
                mal = ablated_score - ensemble_score
                t1_sc_scores[level][team_id].update({f"MAL_{metric}": mal})
            # Separately add diversity MAL (different signature).
            ablated_score = compute_cross_diversity(df[ablated_selection])
            ensemble_score = compute_cross_diversity(df[selection])
            mal = ablated_score - ensemble_score
            t1_sc_scores[level][team_id].update({"MAL_DIV": mal})
    # For task two, get column sets (step:col) by team_id.
    if task == 2:
        # Grouped with t1_ss since it's the same procedure.
        drops = config.DROPS_ENS_T1 if ss else config.DROPS_ENS_T2
        collection = t1_ss_scores if ss else t2_scores
        tasklab = 1 if ss else 2  # Get appropriate task label for t1.
        for team_id, columns in column_sets[task][level].items():
            for step, col in columns.items():
                selection = [
                    list(col.values())  # Column names per step.
                    for team, col in column_sets[task][level].items()
                    if team not in config.DROPS_ENS_T2
                ]  # Returns list of lists, so:
                selection = [item for sublist in selection for item in sublist]
                step_selection = [
                    col for col in selection if f"s{step}" in col
                ]
                ablated_selection = [i for i in step_selection if i != col]
                ablated_ensemble = df[ablated_selection].mean(axis=1)
                full_ensemble = df[step_selection].mean(axis=1)
                obs = df[config.COL_OBS.format(step)]
                for metric, function in {
                    "MSE": METRICS["MSE"],
                    "TADDA_1": METRICS["TADDA_1"],
                    "TADDA_2": METRICS["TADDA_2"],

                }.items():
                    ablated_score = function(obs, ablated_ensemble)
                    ensemble_score = function(obs, full_ensemble)
                    mal = ablated_score - ensemble_score
                    collection[level][team_id][step].update(
                        {f"MAL_{metric}": mal}
                    )
                # Separately add diversity MAL (different signature).
                ablated_score = compute_cross_diversity(df[ablated_selection])
                ensemble_score = compute_cross_diversity(df[step_selection])
                mal = ablated_score - ensemble_score
                collection[level][team_id][step].update({"MAL_DIV": mal})
                # Also add ensemble diversity ACROSS columns to collection.
                # TODO: this is different from the team divs...
                # collection[level][f"ensemble_{level}_t{tasklab}"][step].update(
                #     {"DIV": ensemble_score}
                # )


def add_pemdiv():
    """Adds pemdiv scores from external file."""
    # For task two...
    df = pd.read_csv(
        os.path.join(PEMDIV_PATH, "t2_pemdiv_revised.csv"), index_col=[0]
    )
    df.columns = [int(col.lower().replace("s", "")) for col in df]
    for step, teamscores in df.to_dict().items():
        for team_id, score in teamscores.items():
            if team_id == "no_change":
                team_id = "no_change_pgm"
            t2_scores["pgm"][team_id][step].update({"PEMDIV": score})
    # For task one sc...
    df = pd.read_csv(os.path.join(PEMDIV_PATH, "t1_pemdiv_revised_sc.csv"))
    for team_id, score in df.loc[0].T.to_dict().items():
        if team_id == "no_change":
            team_id = "no_change_pgm"
        t1_sc_scores["pgm"][team_id].update({"PEMDIV": score})
    # For task one ss...
    df = pd.read_csv(
        os.path.join(PEMDIV_PATH, "t1_pemdiv_revised_ss.csv"), index_col=[0]
    )
    df.columns = [int(col.lower().replace("s", "")) for col in df]
    for step, teamscores in df.to_dict().items():
        for team_id, score in teamscores.items():
            if team_id == "no_change":
                team_id = "no_change_pgm"
            t1_ss_scores["pgm"][team_id][step].update({"PEMDIV": score})


def write_ensemble_tables(path):
    """Write final ensemble tables including both t1 and t2."""
    t1ss = pd.read_pickle(os.path.join(OUTPUT_DIR, "tables/t1_ss_scores.pkl"))
    t1sc = pd.read_pickle(os.path.join(OUTPUT_DIR, "tables/t1_sc_scores.pkl"))
    t2 = pd.read_pickle(os.path.join(OUTPUT_DIR, "tables/t2_scores.pkl"))
    # For cm...
    cm_ens = (
        pd.DataFrame(t1ss["cm"]["w_ensemble"])
        .join(pd.DataFrame(t1sc["cm"]["w_ensemble"], index=["sc"]).T)
        .T[["MSE", "TADDA_1", "TADDA_2"]]
        .add_prefix("t1_")
        .join(
            pd.DataFrame(t2["cm"]["ensemble"])
            .T[["MSE", "TADDA_1", "TADDA_2"]]
            .add_prefix("t2_")
        )
    )
    round(cm_ens, 3).to_latex(os.path.join(path, "cm_ensemble_scores.tex"))
    # for pgm...
    pgm_ens = (
        pd.DataFrame(t1sc["pgm"]["w_ensemble"], index=["sc"])
        .T.join(pd.DataFrame(t1ss["pgm"]["w_ensemble"]))
        .T[["MSE", "TADDA_1", "TADDA_2", "PEMDIV"]]
        .add_prefix("t1_")
        .join(
            pd.DataFrame(t2["pgm"]["ensemble"])
            .T[["MSE", "TADDA_1", "TADDA_2", "PEMDIV"]]
            .add_prefix("t2_")
        )
    )
    round(pgm_ens.loc[list(range(2, 8)) + ["sc"]], 3).to_latex(
        os.path.join(path, "pgm_ensemble_scores.tex")
    )
