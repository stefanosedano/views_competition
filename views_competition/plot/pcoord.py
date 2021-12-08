"""Parallel coordinate plot"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from views_competition.plot import utilities
from views_competition import config, OUTPUT_DIR


def plot_parallel_coord(
    df: pd.DataFrame,
    step: int,
    task: int,
    level: str,
    colordict,
    include_mal: bool = False,
    reverse: bool = False,
    figsize=(12, 7),
    axis_coords=1.2,
    path: str = None,
):
    """Makes parallel coordinate plots per df."""
    df = df.copy()
    drops = [col for col in df if "MAL" in col]
    if not include_mal:
        df = df[[col for col in df if col not in drops]]

    # TODO: rewrite and document.
    ys1 = df.values
    ymins1 = ys1.min(axis=0)
    ymaxs1 = ys1.max(axis=0)
    dys1 = ymaxs1 - ymins1
    ymins1 -= dys1 * 0.1  # add 0.005 padding below and above
    ymaxs1 += dys1 * 0.1

    # if level == "pgm":
    #    ymaxs1[2], ymins1[2] = ymins1[2], ymaxs1[2]  # reverse pemdiv scale
    if reverse:
        ymaxs1[1], ymins1[1] = (
            ymins1[1],
            ymaxs1[1],
        )  # reverse axis 1 to have less crossings
        ymaxs1[0], ymins1[0] = ymins1[0], ymaxs1[0]
    # If we'd want the other cols at the same scale (eg. MALS equal)
    # ymaxs1[2], ymins1[2] = ymaxs1[3], ymins1[3]
    dys1 = ymaxs1 - ymins1

    # Transform data using broadcasting to be compatible with the main axis
    zs1 = np.zeros_like(ys1)
    zs1[:, 0] = ys1[:, 0]
    zs1[:, 1:] = (ys1[:, 1:] - ymins1[1:]) / dys1[1:] * dys1[0] + ymins1[0]

    fig, host = plt.subplots(figsize=figsize)

    # Set up and adapt individual axes
    axes = [host] + [host.twinx() for i in range(ys1.shape[1] - 1)]
    for i, ax in enumerate(axes):
        # Set the tick range manually, adapting from host
        # Note that the actual lines will be plotted according to the
        # transformed values above (i.e. all in terms of axis 0.)
        # So essentially these are cosmetic axes.
        ax.set_ylim(ymins1[i], ymaxs1[i])
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        if ax != host:
            ax.spines["left"].set_visible(False)
            ax.yaxis.set_ticks_position("right")
            # Reset drawing position of non-host axes (i fraction of len cols)
            ax.spines["right"].set_position(("axes", i / (ys1.shape[1] - 1)))

    host.set_xlim(0, ys1.shape[1] - 1)
    host.set_xticks(range(ys1.shape[1]))
    host.set_xticklabels(list(df.columns), fontsize=14)
    host.tick_params(axis="x", which="major", pad=7)
    host.spines["right"].set_visible(False)
    host.xaxis.tick_top()
    host.set_title(f"{level}, task {task}, step {step} ", fontsize=18, pad=12)

    for i, j in zip(df.index, range(ys1.shape[0])):
        # For j submission, plot the row values by column
        host.plot(range(ys1.shape[1]), zs1[j, :], c=colordict[i], lw=2)

    host.legend(
        labels=df.index,
        loc="center",
        bbox_to_anchor=(axis_coords, 0, 0, 1.02),
        title=f"Contributions",
    )

    if path:
        plt.tight_layout()
        fig.savefig(
            path,
            dpi=300,
            facecolor="white",
            bbox_inches="tight",
        )
        plt.close(fig)
        print((f"Wrote {path}."))


def make_pcoordplots(level, out_path):
    """Makes parralel coordinate plots for selected metrics."""
    metrics = [
        "MSE",
        "TADDA_1",
        "TADDA_2",
        "MAL_MSE",
        "MAL_TADDA_1",
        "MAL_TADDA_2",
    ]
    metrics = metrics + ["PEMDIV"] if level == "pgm" else metrics
    steps = [*range(2, 8)]
    collection = {}
    figsize = (12, 5) if level == "pgm" else (12, 5)
    axis_coords = 1.3 if level == "pgm" else 1.4  # Due to extra axis pgm.
    collection = utilities.collect_scores(level, steps, metrics)
    for task in ["t2_ss", "t1_ss", "t1_sc"]:
        steps = ["sc"] if task == "t1_sc" else steps
        for step in steps:
            prefix = "s" if step != "sc" else ""
            if task == "t1_sc":
                df = collection["t1_sc"][step]
                figsize = (9, 5)
                df = df.drop(index="ensemble")
            else:
                df = collection[task][step]
            # Drop regular ensemble from t1.
            if task == "t1_ss":
                df = df.drop(index="ensemble")
            # Adjust for merged figure. For now write to table.
            if task == "t2_ss":
                t2_df = collection["t2_ss"][step]
                t2_df.columns = ["t2_" + col for col in t2_df]
                t1_df = collection["t1_ss"][step]
                t1_df.columns = ["t1_" + col for col in t1_df]
                combined_df = t2_df.join(t1_df)
                combined_df = combined_df[
                    [col for col in combined_df if "MAL" not in col]
                ]
                np.round(combined_df, 3).to_latex(
                    os.path.join(
                        OUTPUT_DIR, f"tables/{level}_pcoord_{step}.tex"
                    )
                )
            # # TODO: append "merged" ensemble scores? Discuss.
            df = df.fillna(0)  # Fills missing pemdiv for pgm ensembles.
            colors = {
                team_id: color
                for team_id, color in utilities.get_team_colors()[
                    level
                ].items()
                if team_id not in config.DROPS_DEFAULT
            }
            plot_parallel_coord(
                df=df,
                step=step,
                task=task,
                level=level,
                colordict=colors,
                include_mal=False,
                figsize=figsize,
                axis_coords=axis_coords,
                path=os.path.join(
                    out_path,
                    f"{task}_coord_contributions_{level}_{prefix}{step}.png",
                ),
            )
