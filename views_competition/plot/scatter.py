"""Scatter plot"""

import os
import matplotlib.pyplot as plt

from views_competition.plot import utilities


def simple_scatter(
    df,
    x,
    y,
    s,
    xlabel,
    ylabel,
    title,
    marker="o",
    xlim=None,
    ylim=None,
    alpha=0.8,
    colordict=None,
    path=None,
):
    """Plot a simple scatter."""
    if colordict is not None:
        colors = colordict
    else:
        cmap = plt.cm.tab20.colors
        colors = {col: cmap[i] for i, col in enumerate(df.index)}
    fig, ax = plt.subplots(figsize=(5, 5))
    for idx, row in df.iterrows():
        ax.scatter(
            row[x],
            row[y],
            s=s,
            label=idx,
            alpha=alpha,
            marker=marker,
            edgecolor="black",
            color=colors[idx],
        )
    plt.legend(
        bbox_to_anchor=(1.005, 1),
        loc="upper left",
        bbox_transform=ax.transAxes,
        frameon=False,
    )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)

    if path is not None:
        plt.savefig(path, dpi=200, bbox_inches="tight")
        plt.close()

    return fig, ax


def make_scatterplots(level, out_path):
    """Makes scatter plots for MSE and corr by level."""
    # Collect all scores per team, per step.
    # TODO: would be easier if t2_scores[level][step].
    steps = [2, 7]
    collection = utilities.collect_scores(level, steps)
    for task in ["t1_ss", "t2_ss", "t1_sc"]:
        for step, df in collection[task].items():
            # Drop the ensembles and no-change ex-post.
            df = df.loc[[idx for idx in df.index if "ensemble" not in idx]]
            df = df.loc[[idx for idx in df.index if "no_change" not in idx]]
            prefix = "s" if step != "sc" else ""
            simple_scatter(
                df=df,
                x=f"MSE",
                y=f"corr",
                s=100,
                ylim=(-1, 1),
                xlabel="MSE",
                ylabel="Correlation",
                title=f"Step {step}",
                colordict=utilities.get_team_colors()[level],
                path=os.path.join(
                    out_path,
                    f"{task}_scatter_{level}_{prefix}{step}.png",
                ),
            )
            # Also layered with positive/negative.
            # TODO: set specific lims for task one.
            ylim = None
            if task == 2:
                if step == 2:
                    ylim = (-0.2, 0.6) if level == "cm" else (-0.8, 0.85)
                else:
                    ylim = (-0.2, 0.6) if level == "cm" else (-0.1, 0.85)
            else:
                ylim = (-1, 1)

            # Plot.
            _, ax = simple_scatter(
                df=df,
                x=f"MSE_zero",
                y=f"corr",
                s=100,
                ylim=ylim,
                xlabel="MSE",
                ylabel="Correlation",
                title=f"Step {step}",
                alpha=0.5,
                colordict=utilities.get_team_colors()[level],
            )
            for idx, row in df.iterrows():
                ax.scatter(
                    row[f"MSE_positive"],
                    row[f"corr_positive"],
                    s=100,
                    label=idx,
                    alpha=0.5,
                    edgecolor="black",
                    marker="^",
                    color=utilities.get_team_colors()[level][idx],
                )
            for idx, row in df.iterrows():
                ax.scatter(
                    row[f"MSE_negative"],
                    row[f"corr_negative"],
                    s=100,
                    label=idx,
                    alpha=0.5,
                    marker="v",
                    edgecolor="black",
                    color=utilities.get_team_colors()[level][idx],
                )
            plt.savefig(
                os.path.join(
                    out_path, f"{task}_partial_{level}_{prefix}{step}.png"
                ),
                dpi=200,
                bbox_inches="tight",
            )
            plt.close()

            # Also do a separate plot for each category, while keeping proper lims.
            for layer in ["positive", "negative"]:
                xlim = None
                if task == 2:
                    xlim = (-0.2, 4) if level == "cm" else (-1, 15)
                    xlim = (
                        (-0.01, 0.05)
                        if level == "pgm" and layer == "zero"
                        else xlim
                    )
                    xlim = (
                        (-0.01, 0.3)
                        if level == "cm" and layer == "zero"
                        else xlim
                    )
                marker = "^" if layer == "positive" else "v"
                simple_scatter(
                    df=df,
                    x=f"MSE_{layer}",
                    y=f"corr_{layer}",  # Just the regular correlation here.
                    s=100,
                    xlim=xlim,
                    ylim=ylim,
                    xlabel="MSE",
                    ylabel="Correlation",
                    title=f"Step {step}",
                    alpha=0.5,
                    colordict=utilities.get_team_colors()[level],
                    marker=marker,
                )
                plt.savefig(
                    os.path.join(
                        out_path,
                        f"{task}_{level}_{layer}_{prefix}{step}.png",
                    ),
                    dpi=200,
                    bbox_inches="tight",
                )
                plt.close()
