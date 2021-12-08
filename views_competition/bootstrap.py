"""Bootstrap"""

import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from views_competition import config, datautils, evaluate, OUTPUT_DIR
from views_competition.plot import utilities

log = logging.getLogger(__name__)


cm_t1 = (
    pd.read_csv(os.path.join(OUTPUT_DIR, "data/t1_cm.csv"))
    .set_index(["month_id", "country_id"])
    .sort_index()
)
cm_t1_ss = (
    pd.read_csv(os.path.join(OUTPUT_DIR, "data/t1_cm_ss.csv"))
    .set_index("country_id")
    .sort_index()
)
cm_t2 = (
    pd.read_csv(os.path.join(OUTPUT_DIR, "data/t2_cm.csv"))
    .set_index(["month_id", "country_id"])
    .sort_index()
)
pgm_t1 = (
    pd.read_csv(os.path.join(OUTPUT_DIR, "data/t1_pgm.csv"))
    .set_index(["month_id", "pg_id"])
    .sort_index()
)
pgm_t1_ss = (
    pd.read_csv(os.path.join(OUTPUT_DIR, "data/t1_pgm_ss.csv"))
    .set_index("pg_id")
    .sort_index()
)
pgm_t2 = (
    pd.read_csv(os.path.join(OUTPUT_DIR, "data/t2_pgm.csv"))
    .set_index(["month_id", "pg_id"])
    .sort_index()
)


def bootstrap_metric(
    obs,
    pred,
    func,
    seed=101,
    n_bootstraps=1000,
):
    """Evaluates n_bootstraps random samples with replacement."""
    random = np.random.RandomState(seed)
    scores = []
    for i in range(n_bootstraps):
        idx = []
        # Redo sample if only one unique id is returned.
        while len(np.unique(idx)) < 2:
            idx = random.randint(low=0, high=len(pred), size=len(pred))
        scores.append(func(obs.values[idx], pred.values[idx]))
    return scores


def do_bootstraps(
    df,
    column_sets,
):
    """Returns bootstrapped metrics by submission."""
    level = "cm" if "country_id" in df.index.names else "pgm"
    t1_ss = True if "month_id" not in df.index.names else False
    task = 2 if t1_ss else datautils.determine_task(df)  # Group t1_ss with t2.
    collection = {}
    log.info(f"Bootstrapping for {level}, task {task}.")
    # For task1, get team_id:col pairs, step-combined.
    if task == 1:
        ensembles = {
            f"ensemble": f"ensemble",
            f"w_ensemble": f"w_ensemble",
        }
    if task == 2:
        ensembles = {f"ensemble": {i: f"ensemble_s{i}" for i in range(2, 8)}}
    sets = column_sets[task][level].copy()
    sets.update(ensembles)

    if task == 1:
        for team_id, col in sets.items():
            for metric, function in evaluate.METRICS.items():
                if metric == "MSE":
                    log.debug(f"Bootstrapping {team_id} {metric}...")
                    col_obs = config.COL_OBS_T1
                    obs, preds = evaluate.preprocess(df, metric, col_obs, col)
                    collection[team_id] = {
                        metric: bootstrap_metric(obs, preds, function)
                    }
    # For task2 and task1_ss, get column sets (step:col) by team_id.
    if task == 2:
        # TODO: grouping t1_ss here is confusing!
        for team_id, columns in sets.items():
            scores = {}
            # Then collect by step and metric:
            for step, col in columns.items():
                scores[step] = {}
                for metric, function in evaluate.METRICS.items():
                    if metric == "MSE":
                        log.debug(
                            f"Bootstrapping {team_id} {step} {metric}..."
                        )
                        # Use pre-patch actuals to evaluate the t2 benchmark.
                        if t1_ss or team_id != "benchmark":
                            col_obs = config.COL_OBS.format(step)
                        else:
                            col_obs = config.COL_OBS_09.format(step)
                        obs, preds = evaluate.preprocess(
                            df, metric, col_obs, col
                        )
                        scores[step].update(
                            {metric: bootstrap_metric(obs, preds, function)}
                        )
            collection[team_id] = scores
    return collection


def plot_bootstraps_sc(
    scores,
    level,
    task,
    metric="MSE",
):
    """Plots step-combined bootstrap deltas."""
    s_benchmark = pd.Series(scores["benchmark"][metric])
    plotdata = {}
    for team, scores in scores.items():
        if team not in config.DROPS_DEFAULT + ["benchmark"]:
            s = pd.Series(scores[metric])
            delta = s - s_benchmark
            lower = sorted(delta.values)[int(0.05 * len(sorted(delta.values)))]
            upper = sorted(delta.values)[int(0.95 * len(sorted(delta.values)))]
            plotdata[team] = {
                "delta": s.mean() - s_benchmark.mean(),
                "lower": lower,
                "upper": upper,
                "color": utilities.get_team_colors()[level][team],
            }
    df = pd.DataFrame(plotdata).T.sort_index(ascending=False)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(df.delta, df.index, s=70, zorder=1, color=df.color)
    plt.scatter(
        df.lower,
        df.index,
        color="black",
        marker="|",
        alpha=1,
        s=100,
    )
    plt.scatter(
        df.upper,
        df.index,
        color="black",
        marker="|",
        alpha=1,
        s=100,
    )
    plt.hlines(
        y=df.index, xmin=df.delta, xmax=df.lower, color="black", zorder=0
    )
    plt.hlines(
        y=df.index, xmin=df.delta, xmax=df.upper, color="black", zorder=0
    )
    ax.grid(b=True, axis="x", color="grey", linestyle="--", alpha=0.2)
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    plt.tick_params(left=False, bottom=False)
    plt.suptitle(f"Difference to benchmark, {metric}", y=0.95, fontsize=14)
    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "graphs/bootstrap/",
            f"./t{task}_{level}_sc_bs_{metric.lower()}.png",
        ),
        dpi=200,
        bbox_inches="tight",
    )


def plot_bootstraps_ss(scores, level, task, metric="MSE", against="benchmark"):
    """Plots step-specific bootstrap deltas."""
    plotdata = {step: {} for step in range(2, 8)}
    benchmark = scores[against]

    # Set task- and level-dependent xlims. TODO: refactor...
    if task == 2:
        if against == "benchmark":
            xmin, xmax = (-0.4, 0.4) if level == "cm" else (-0.05, 0.05)
        if against == "mueller":
            xmin, xmax = (-0.1, 0.6)
    if task == "1_ss":
        if against == "benchmark":
            xmin, xmax = (-3, 1) if level == "cm" else (-0.05, 0.05)
        if against == "mueller":
            xmin, xmax = (-1, 3)
        if against == "dorazio":
            xmin, xmax = (0, 0.08)

    # Collect scores.
    for team, scores in scores.items():
        if team not in config.DROPS_DEFAULT + [against]:
            for step in range(2, 8):
                s_benchmark = pd.Series(benchmark[step][metric])
                s = pd.Series(scores[step][metric])
                delta = s - s_benchmark
                lower = sorted(delta.values)[
                    int(0.05 * len(sorted(delta.values)))
                ]
                upper = sorted(delta.values)[
                    int(0.95 * len(sorted(delta.values)))
                ]
                plotdata[step][team] = {
                    "delta": s.mean() - s_benchmark.mean(),
                    "lower": lower,
                    "upper": upper,
                    "color": utilities.get_team_colors()[level][team],
                }
    for step in range(2, 8):
        df = pd.DataFrame(plotdata[step]).T.sort_index(ascending=False)
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(df.delta, df.index, s=70, zorder=1, color=df.color)
        plt.scatter(
            df.lower,
            df.index,
            color="black",
            marker="|",
            alpha=1,
            s=100,
        )
        plt.scatter(
            df.upper,
            df.index,
            color="black",
            marker="|",
            alpha=1,
            s=100,
        )
        plt.xlim(xmin, xmax)
        plt.hlines(
            y=df.index, xmin=df.delta, xmax=df.lower, color="black", zorder=0
        )
        plt.hlines(
            y=df.index, xmin=df.delta, xmax=df.upper, color="black", zorder=0
        )
        ax.grid(b=True, axis="x", color="grey", linestyle="--", alpha=0.2)
        for spine in plt.gca().spines.values():
            spine.set_visible(False)
        plt.tick_params(left=False, bottom=False)
        plt.suptitle(
            f"Difference from {against}, task {task} {level}, {metric} s{step} ",
            y=0.95,
            fontsize=13,
        )
        plt.savefig(
            os.path.join(
                OUTPUT_DIR,
                "graphs/bootstrap/",
                f"./t{task}_{level}_s{step}_bs_{metric.lower()}_{against}.png",
            ),
            dpi=200,
            bbox_inches="tight",
        )
        plt.close()


def plot_bootstraps_ss_team(
    scores, level, task, metric="MSE", against="benchmark"
):
    """Plots step-specific bootstrap deltas."""
    plotdata = {}
    benchmark = scores[against]

    # Set task- and level-dependent xlims. TODO: refactor...
    xmin, xmax = (None, None)

    # Collect scores.
    for team, scores in scores.items():
        if team not in config.DROPS_DEFAULT + [against]:
            plotdata[team] = {}
            for step in range(2, 8):
                s_benchmark = pd.Series(benchmark[step][metric])
                s = pd.Series(scores[step][metric])
                delta = s - s_benchmark
                lower = sorted(delta.values)[
                    int(0.05 * len(sorted(delta.values)))
                ]
                upper = sorted(delta.values)[
                    int(0.95 * len(sorted(delta.values)))
                ]
                plotdata[team][step] = {
                    "delta": s.mean() - s_benchmark.mean(),
                    "lower": lower,
                    "upper": upper,
                    "color": utilities.get_team_colors()[level][team],
                }

            df = pd.DataFrame(plotdata[team]).T.sort_index()
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.scatter(df.delta, df.index, s=70, zorder=1, color=df.color)
            plt.scatter(
                df.lower,
                df.index,
                color="black",
                marker="|",
                alpha=1,
                s=100,
            )
            plt.scatter(
                df.upper,
                df.index,
                color="black",
                marker="|",
                alpha=1,
                s=100,
            )
            plt.gca().invert_yaxis()
            ax.set_yticks([2, 3, 4, 5, 6, 7])
            ax.set_yticklabels([f"s{i}" for i in range(2, 8)])
            plt.xticks(rotation=30)
            plt.xlim(xmin, xmax)
            plt.hlines(
                y=df.index,
                xmin=df.delta,
                xmax=df.lower,
                color="black",
                zorder=0,
            )
            plt.hlines(
                y=df.index,
                xmin=df.delta,
                xmax=df.upper,
                color="black",
                zorder=0,
            )
            ax.grid(b=True, axis="x", color="grey", linestyle="--", alpha=0.2)
            for spine in plt.gca().spines.values():
                spine.set_visible(False)
            plt.tick_params(left=False, bottom=False)
            plt.suptitle(
                f"Bootstrapped {metric}, {team} vs. {against}, task {task}",
                y=0.95,
                fontsize=13,
            )
            plt.savefig(
                os.path.join(
                    OUTPUT_DIR,
                    "graphs/bootstrap/",
                    f"./t{task}_{level}_{team}_bs_{metric.lower()}_{against}.png",
                ),
                dpi=200,
                bbox_inches="tight",
            )
            plt.close()


def plot_bootstraps_ss_line(
    scores,
    level,
    task,
    metric="MSE",
):
    """Plots step-specific bootstrap deltas."""
    plotdata = {}
    benchmark = scores["benchmark"]
    for team, scores in scores.items():
        if team not in config.DROPS_DEFAULT + ["benchmark"]:
            plotdata[team] = {}
            for step in range(2, 8):
                s_benchmark = pd.Series(benchmark[step][metric])
                s = pd.Series(scores[step][metric])
                delta = s - s_benchmark
                lower = sorted(delta.values)[
                    int(0.05 * len(sorted(delta.values)))
                ]
                upper = sorted(delta.values)[
                    int(0.95 * len(sorted(delta.values)))
                ]
                plotdata[team][step] = {
                    "delta": s.mean() - s_benchmark.mean(),
                    "lower": lower,
                    "upper": upper,
                }
            df = pd.DataFrame(plotdata[team]).T
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(
                df.index,
                df.delta,
                marker="o",
                markersize=15,
                markerfacecolor=utilities.get_team_colors()[level][team],
                zorder=1,
                color="black",
            )
            ax.set_xlabel("Step", fontsize=12)
            ax.set_ylabel(f"Difference in {metric}", fontsize=12)
            ax.fill_between(
                df.index, df.lower, df.upper, color="black", alpha=0.2
            )
            ax.grid(b=True, axis="y", color="grey", linestyle="--", alpha=0.2)
            for spine in plt.gca().spines.values():
                spine.set_visible(False)
            plt.tick_params(left=False, bottom=False)
            plt.title(f"{team} - benchmark", fontsize=14)
            plt.savefig(
                os.path.join(
                    OUTPUT_DIR,
                    "graphs/bootstrap/",
                    f"./t{task}_{level}_{team}_bs_{metric.lower()}.png",
                ),
                dpi=200,
                bbox_inches="tight",
            )


def main():
    """Produces bootstrapping outputs."""
    for df in [cm_t1, pgm_t1]:
        level = "cm" if "country_id" in df.index.names else "pgm"
        task = datautils.determine_task(df)
        scores = do_bootstraps(df, config.COLUMN_SETS)
        plot_bootstraps_sc(scores, level, task)
    for df in [cm_t1_ss, cm_t2, pgm_t1_ss]:  # pgm_t2]:
        level = "cm" if "country_id" in df.index.names else "pgm"
        if "month_id" not in df.index.names:
            task = "1_ss"  # TODO: avoid changing type.
        else:
            task = datautils.determine_task(df)
        scores = do_bootstraps(df, config.COLUMN_SETS)
        plot_bootstraps_ss(scores, level, task)
        if level == "cm":
            plot_bootstraps_ss(scores, level, task, against="mueller")
        else:
            plot_bootstraps_ss(scores, level, task, against="dorazio")
