"""Radar plot"""

import os
from math import pi
import matplotlib.pyplot as plt

from views_competition.plot import utilities


def radar_plot(
    categories,
    values,
    label="",
    set_ax=None,
    figsize=(5, 5),
    tick_size=12,
    minmax=None,
    color="darkblue",
    alpha=1,
    fill=False,
    lw=1,
    linestyle="solid",
    title=None,
    titlesize=14,
    path=None,
):
    """Plots a radar chart."""
    # number of nodes.
    N = len(categories)

    # We are going to plot the first line of the data frame.
    # But we need to repeat the first value to close the circular graph:
    values = values.values.flatten().tolist()
    values += values[:1]

    # What will be the angle of each axis in the plot?
    # (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    # Initialise the spider plot
    if set_ax is None:
        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
    else:
        ax = set_ax

    # Draw one tick per var
    plt.xticks(angles[:-1], categories, size=tick_size)

    # Draw ylabels
    ax.set_rlabel_position(0)
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    # Adjust ticks
    plt.yticks([0, 0.5, 1], ["Worst", "Median", "Best"], color="grey", size=10)
    if minmax is not None:
        plt.ylim(minmax[0], minmax[1])

    # Plot data
    ax.plot(
        angles,
        values,
        linewidth=lw,
        linestyle=linestyle,
        color=color,
        alpha=alpha,
        label=label,
    )

    # Fill area
    if fill:
        ax.fill(angles, values, color=color, alpha=0.1)

    if title:
        ax.set_title(title, fontdict={"fontsize": titlesize})

    ax.spines["polar"].set_visible(False)

    if path is not None:
        plt.savefig(path, dpi=200, bbox_inches="tight")
        plt.close()

    if set_ax is None:
        return fig, ax
    return ax


def make_radarplots(level, out_path):
    """Makes radarplots for selected scores by level."""
    steps = [2, 7]
    reverse_cols = [
        "MSE",
        "TADDA_1",
        "TADDA_2",
        "MSE_zero",
        "MSE_negative",
        "MSE_positive",
        "cal_m",
        "cal_sd",
        "DIV",  # Lower is less MSE compared to ensemble, so revert.
    ]

    def minmax_norm(s):
        return (s - s.min()) / (s.max() - s.min())

    # Iterate over collected score dataframes.
    collection = utilities.collect_scores(level, steps)
    for task in ["t1_ss", "t2_ss", "t1_sc"]:
        for step, df in collection[task].items():
            # df.columns = [col.split(f"_{step}")[0] for col in df]
            # if "t1" in task:
            #     add = [f"ensemble_{level}_t1", f"w_ensemble_{level}_t1"]
            # else:
            #     add = [f"ensemble_{level}_t2"]  # Add to colors.
            prefix = "s" if step != "sc" else ""
            # Prepare selection of metrics.
            metrics = [
                "MSE",
                "TADDA_1",
                "TADDA_2",
                "MSE_zero",
                "MSE_negative",
                "MSE_positive",
                "MAL_MSE",
                "MAL_TADDA_1",
                "MAL_TADDA_2",
                "DIV",
                "cal_m",
                "cal_sd",
                "corr",
            ]
            if level == "pgm" and task in ("t2_ss", "t1_sc"):
                metrics = metrics + ["PEMDIV"]
                reverse = reverse_cols + ["PEMDIV"]
            else:
                reverse = reverse_cols
            # Adjustment per 04-2021: normalize first, then reverse.
            for col in metrics:
                df[col] = minmax_norm(df[col])
            for col in reverse:
                df[col] = df[col].max() - df[col]

            df = df.T
            df = df.fillna(0.5)  # Fill ensembles NaN with 0.5.
            # Get the no-change column, drop ensemble column at t1.
            no_change = [col for col in df if "no_change" in col][0]
            # teamcols = [col for col in df if f"ensemble_{level}" not in col]
            teamcols = df.columns

            for team in teamcols:
                _, ax1 = radar_plot(
                    metrics,
                    df.loc[metrics, team],
                    title=f"{team}, {prefix}{step}",
                    lw=2,
                    label=team,
                    minmax=(-0.5, 1),
                    color=utilities.get_team_colors()[level][team],
                    fill=True,
                )
                radar_plot(
                    metrics,
                    df.loc[metrics, "benchmark"],
                    lw=1.5,
                    set_ax=ax1,
                    color=utilities.get_team_colors()[level]["benchmark"],
                    alpha=0.5,
                    linestyle="solid",
                    label="benchmark",
                    fill=False,
                )
                radar_plot(
                    metrics,
                    df.loc[metrics, no_change],
                    lw=1.5,
                    set_ax=ax1,
                    color=utilities.get_team_colors()[level][no_change],
                    alpha=0.5,
                    linestyle="dashed",
                    label="no_change",
                    fill=False,
                )
                ax1.legend(
                    bbox_to_anchor=(1.005, 1),
                    loc="upper left",
                    bbox_transform=ax1.transAxes,
                    frameon=False,
                )
                plt.savefig(
                    os.path.join(
                        out_path,
                        f"{task}_radar_{level}_{team}_{prefix}{step}.png",
                    ),
                    dpi=200,
                    bbox_inches="tight",
                )
                plt.close()
