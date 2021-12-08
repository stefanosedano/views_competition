"""Ablation plot module"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from views_competition import evaluate, config


def ablation_plot(
    df,
    steps,
    metric,
    title=None,
    circle_size=500,
    x_offset=0.1,
    text_size=30,
    ylabel=True,
    legend_t="model",
    cmap="viridis",
    vmin=0,
    vmax=3,
    path=None,
):
    """Makes ablation plot for a df.

    TODO: rewrite to be series-based and without hardcoding
    """
    n_steps = len(steps)
    fig, (axes) = plt.subplots(
        1, n_steps, sharey=True
    )  # Labels only for first y-axis.
    fig.set_size_inches(10 * n_steps, 12)

    # Set tight layout.
    plt.subplots_adjust(wspace=0, hspace=0, left=0)

    # Set up plot.
    for step, ax in zip(steps, axes):
        ax.scatter(
            df[f"MAL_{metric}_{step}"],
            df.index,
            s=circle_size,
            c=df[f"{metric}_{step}"],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            alpha=1,
            marker="o",
            zorder=10,
        )

        # Plot left - right lines.
        ax.grid(color="silver", linestyle="-.", linewidth=1, axis="y")

        # Plot gray field (TODO).
        metric_tot_mal = 0
        ax.axvline(metric_tot_mal, color="silver", linestyle="--", linewidth=1)

        # Limit the x_axis.
        x_min, x_max = metric_tot_mal - x_offset, metric_tot_mal + x_offset
        ax.set_xlim([x_min, x_max])

        # Set ax title.
        ax.set_title(f"S = {step}", fontdict={"fontsize": text_size})
        ax.set_xlabel(f"Ablation loss, {metric.upper()}", fontsize=text_size)

        # Set x-ticks.
        ax.set_xticks(ax.get_xticks()[1:-1])
        ax.set_xticklabels(
            [np.round(i, 3) for i in ax.get_xticks()], rotation=30
        )

        # Set x and y axis tick label size.
        ax.tick_params(axis="both", which="major", labelsize=text_size - 7)

    # Get bounding box information for the axes of the last subplot.
    bbox_ax0 = axes[0].get_position()

    # Write models/features as the Y axis label.
    if ylabel:
        fig.text(
            bbox_ax0.x0 - 0.21,
            0.5,
            f"{legend_t}s",
            va="center",
            rotation="vertical",
            fontsize=text_size,
        )

    # Add colorbar.
    # Make ax for colorbar and add to canvas.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="10%", pad=0.1)
    # Fill in the colorbar and adjust the ticks.
    sm = plt.cm.ScalarMappable(
        cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax)
    )
    sm._A = []  # pylint: disable=protected-access
    cbar = plt.colorbar(sm, cax=cax)
    cbar = cbar.set_label(
        f"{legend_t} {metric.upper()}",
        horizontalalignment="center",
        fontsize=text_size,
        labelpad=10,
    )
    cax.tick_params(labelsize=text_size - 6)

    if title is not None:
        fig.suptitle(title, fontsize=text_size + 5, y=1.1)

    if path is not None:
        plt.tight_layout()
        fig.savefig(path, dpi=200)
        plt.close(fig)


def make_ablation_plots(out_path):
    """Produces ablation plots per level and task."""
    for task in [1, 2]:
        for level in ["cm", "pgm"]:
            collection = (
                evaluate.t1_ss_scores if task == 1 else evaluate.t2_scores
            )
            drops = config.DROPS_ENS_T1 if task == 1 else config.DROPS_ENS_T2
            steps = [2, 4, 7]
            smin = min(steps)
            smax = max(steps)
            scores = {}
            # Collect all relevant scores.
            for team_id, score_dict in collection[level].items():
                if team_id not in drops:  # TODO: drops same t1_ss?
                    scores[team_id] = {}
                    for metric in [
                        "MSE",
                        "TADDA_1",
                        "TADDA_2",
                        "MAL_MSE",
                        "MAL_TADDA_1",
                        "MAL_TADDA_2",
                    ]:
                        team_score = {
                            f"{metric}_{step}": score_dict[step][metric]
                            for step in steps
                        }
                        scores[team_id].update(team_score)

            df = pd.DataFrame(scores).T.sort_index()
            max_mse = df[[f"MSE_{step}" for step in steps]].max().max()
            # TODO: taking the max for 2 here, assuming 1 is close enough.
            max_tadda = df[[f"TADDA_2_{step}" for step in steps]].max().max()
            # Prepare ablation plot per MSE and TADDA.
            for metric in ["MSE", "TADDA_1", "TADDA_2"]:
                metric_df = df[[col for col in df if f"MAL_{metric}" in col]]
                metmin, metmax = metric_df.min().min(), metric_df.max().max()
                x_offset = (
                    1.2 * abs(metmin) if abs(metmin) > metmax else 1.1 * metmax
                )
                vmax = max_mse if metric == "MSE" else max_tadda
                ablation_plot(
                    df.sort_index(ascending=False),
                    steps,
                    circle_size=500,
                    x_offset=x_offset,
                    text_size=30,
                    ylabel=True,
                    legend_t="model",
                    cmap="viridis",
                    metric=metric,
                    vmin=0,
                    vmax=vmax,
                    title=level,
                    path=os.path.join(
                        out_path,
                        f"t{task}_ablation_{metric.lower()}_{level}_{smin}_{smax}.png",
                    ),
                )
