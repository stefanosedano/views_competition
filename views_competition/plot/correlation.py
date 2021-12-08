"""Correlation plot"""

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from views_competition import config


def draw_corrplot(
    df: pd.DataFrame,
    task: int,
    level: str,
    month_id: int = None,
    step: int = None,
    title: str = None,
    annot: bool = False,
    cmap: str = "Spectral_r",
    vmin: float = -0.1,
    vmax: float = 1,
    center: float = 0.5,
    line_width: float = 0,
    rotate_yticks: int = 0,
    rotate_xticks: int = 0,
    labels=True,
    path: str = None,
):
    """Plot correlation of predictions between models."""
    # Time period
    if month_id:
        df = df.loc[month_id]

        # Month id to step:
        times = list(range(490, 496))  # Oct 2020 - March 2020
        all_steps = [2, 3, 4, 5, 6, 7]
        time_steps = dict(zip(times, all_steps))
        step = time_steps[month_id]

    # Correlation.
    corr = df.corr()

    # Set up figure.
    fig, ax = plt.subplots(figsize=(14, 7))
    sns.set(style="white")

    # Plot.
    sns.heatmap(
        corr,
        mask=False,
        cmap=cmap,
        square=True,
        vmin=vmin,
        vmax=vmax,
        center=center,
        annot=annot,
        xticklabels=labels,
        yticklabels=labels,
        linewidths=line_width,
    )

    # Set axes ticks.
    plt.yticks(rotation=rotate_yticks)
    plt.xticks(
        rotation=rotate_xticks,
        horizontalalignment="center",
    )

    # Title.
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"Correlation Plot: Task {task} ({level}), Step {step}")

    # Save figure.
    if path:
        plt.tight_layout()
        fig.savefig(
            path,
            dpi=300,
            facecolor="white",
            bbox_inches="tight",
        )
        print((f"Wrote {path}."))


def make_t1_corrplots(df, column_sets, out_path):
    """Makes correlation plots for task one."""
    level = "cm" if "country_id" in df.index.names else "pgm"
    columns = [
        col
        for _, col in column_sets[1][level].items()
        if col not in config.DROPS_ENS_T1
    ]
    for month_id in [490, 495]:
        draw_corrplot(
            df=df[sorted(columns)],
            task=1,
            level=level,
            month_id=month_id,
            cmap="RdBu_r",
            rotate_xticks=90,
            labels=[col.split("_")[0] for col in sorted(columns)],
            path=os.path.join(
                out_path,
                f"t1_corr_contributions_{level}_{month_id}.png",
            ),
        )


def make_t2_corrplots(df, column_sets, out_path):
    """Makes correlation plots for task two."""
    # Note that no_change is excluded here though it is in the t2 ensemble.
    level = "cm" if "country_id" in df.index.names else "pgm"
    columns = [
        list(col.values())  # Column names per step.
        for team, col in column_sets[2][level].items()
        if team not in config.DROPS_ENS_T2 + ["no_change_cm", "no_change_pgm"]
    ]
    columns = [item for sublist in columns for item in sublist]
    for step in [2, 7]:
        # TODO: use alternative to substring matching.
        step_selection = [col for col in columns if f"s{step}" in col]
        draw_corrplot(
            df=df[sorted(step_selection)],
            task=2,
            level=level,
            step=step,
            cmap="RdBu_r",
            rotate_xticks=90,
            labels=[col.split("_")[0] for col in sorted(step_selection)],
            path=os.path.join(
                out_path, f"t2_corr_contributions_{level}_{step}.png"
            ),
        )
