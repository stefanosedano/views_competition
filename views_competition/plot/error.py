from typing import Optional, Tuple, Any
import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import colors as pltcol, gridspec, text
from matplotlib.patches import ConnectionPatch
from mpl_toolkits.axes_grid1 import make_axes_locatable

from views_competition import config, datautils, DATA_DIR, TIMEFRAMES
from views_competition.plot import maps


mapdata = maps.MapData()
log = logging.getLogger(__name__)


def plot_mc_delta(
    s_pred: pd.Series,
    s_obs: pd.Series,
    s_lab: pd.Series,
    title: str,
    axs: Optional[Any] = None,
    labels: bool = True,
    delta_alpha: bool = False,
    limit: float = None,
    loess: bool = True,
    notation: str = r"\Delta Y",
    worstn: int = 5,
    cmap: str = "RdBu_r",
    textsize: int = 12,
    figsize: Tuple[float, float] = (4, 6),
    path: Optional[str] = None,
) -> None:
    """Produce a model criticism-style plot for a delta regressor.

    Colored according to the observed value.

    Args:
        s: Series to compare.
        s_obs: Series to compare to.
        s_lab: Series of labels with same index as s.
        title: Title to put on plot.
        delta_alpha: Adjust alpha of scatter according to abs delta between
            s_pred and s_obs.
        loess: Add loess to margin plot.
        notation: Notation for the unit predicted. Delta by default.
        worstn: Number of worst error annotations to plot.
        cmap: Matplotlib colormap to apply to top ax.
        textsize: Base textsize. Tick and title are relative to this.
        figsize: Tuple of (width, height) to pass to figsize.
        path: Optional path to write figure to.
    """
    # Sort, reindex labels and observed (opt), and reset to single index.
    delta = s_pred - s_obs
    s_pred = s_pred.sort_values()
    s_lab = s_lab.reindex(s_pred.index)
    s_obs = s_obs.reindex(s_pred.index)
    delta = delta.reindex(s_pred.index)

    # Convert back to single basic index.
    delta = delta.reset_index(drop=True)
    s_pred = s_pred.reset_index(drop=True)
    s_lab = s_lab.reset_index(drop=True)
    s_obs = s_obs.reset_index(drop=True)

    # Set up figure space.
    if axs is None:
        _, axs = plt.subplots(
            nrows=2,
            figsize=figsize,
            sharex=True,
            gridspec_kw={"height_ratios": [4, 1]},
        )
    vmin, vmax = s_pred.min(), s_pred.max()

    if limit is None:
        limit = 1.25 * max(abs(vmin), abs(vmax))  # Added space for rugplot.
    plt.xlim(-limit, limit)

    # Set up colorscheme.
    colormap = plt.get_cmap(cmap)
    norm = plt.Normalize(s_obs.min(), s_obs.max())
    colors = colormap(norm(s_obs.values))

    # Set up alphas if requested.
    if delta_alpha:
        dalpha = abs(delta.copy())
        # Normalize that 0-1 so it can be used as alpha.
        alphas = (dalpha - np.min(dalpha)) / (np.max(dalpha) - np.min(dalpha))
        alphas.loc[alphas < 0.3] = 0.3  # Set alpha 0.3 as the minimum.
        colors[:, 3] = alphas  # Alpha is the fourth col in a color array.

    # Set up top ax.
    axs[0].scatter(s_pred, s_pred.index, color=colors)
    axs[0].margins(0.02)
    axs[0].grid(
        which="major",
        axis="x",
        lw=1,
        color="black",
        alpha=0.1,
    )
    axs[0].set_ylabel(
        "Observation (ordered by {})".format(r"$\hat {}$".format(notation)),
        size=textsize + 3,
    )

    # Draw hline at first index where our predictions pass zero.
    positive_mask = s_pred >= 0
    zero_index = positive_mask.idxmax()
    axs[0].axhline(
        y=zero_index, xmin=0, xmax=1, color="black", lw=0.8
    )  # axhline uses axes coordinate system, so just 0-1.
    axs[0].axvline(x=0, ymin=0, ymax=len(s_pred), color="black", lw=0.8)

    if labels:
        # Prepare annotations.
        trans = axs[0].get_yaxis_transform()  # Axis coords x, data coords y.
        delta = delta.sort_values()
        spacing = len(delta) / 25
        step = 0.0
        start_loc = round(0.75 * len(delta))

        # Hang annotations for positive deltas.
        for index, value in (
            delta[-worstn:].sort_index(ascending=False).items()
        ):
            # Add label above the annotations.
            if step == 0:
                axs[0].annotate(
                    "Overpredicted",
                    xy=(limit, index),
                    xycoords="data",
                    xytext=(1.15, start_loc),  # A bit above list.
                    textcoords=trans,
                )
                step = step + spacing
            # Set up color, horizontal lines, annotations.
            edgecolor = pltcol.to_hex(colormap(norm(s_obs[index])))
            axs[0].hlines(
                y=index,
                xmin=s_pred[index],
                xmax=limit,
                color=edgecolor,
                alpha=0.2,
            )
            axs[0].annotate(
                s_lab[index],  # Get associated label by index.
                xy=(limit, index),
                xycoords="data",
                xytext=(1.15, start_loc - step - spacing * 0.5),
                textcoords=trans,
                va="center",
                ha="left",
                size=textsize,
            )
            # Little trick here to actually attach to the left center point.
            axs[0].annotate(
                "",
                xy=(limit, index),
                xycoords="data",
                xytext=(1.15, start_loc - step),
                textcoords=trans,
                arrowprops=dict(
                    arrowstyle="-", edgecolor=edgecolor, shrinkB=0, shrinkA=0
                ),
            )
            step = step + spacing

        # Stack annotations for negatives deltas.
        step = 0.0
        start_loc = start_loc - ((worstn + 3) * spacing)

        for index, value in delta[0:worstn].sort_values().items():
            # Add label above the annotations.
            if step == 0:
                axs[0].annotate(
                    "Underpredicted",
                    xy=(limit, index),
                    xycoords="data",
                    xytext=(
                        1.15,
                        start_loc,
                    ),  # Place a bit above list.
                    textcoords=trans,
                )
                step = step + spacing
            # Draw line to right axis and to annotation base.
            edgecolor = pltcol.to_hex(colormap(norm(s_obs[index])))
            axs[0].hlines(
                y=index,
                xmin=s_pred[index],
                xmax=limit,
                color=edgecolor,
                alpha=0.2,
            )
            # Set up color, horizontal lines, annotations.
            axs[0].annotate(
                s_lab[index],  # Get associated label by index.
                xy=(limit, index),
                xycoords="data",
                xytext=(1.15, start_loc - step - spacing * 0.5),
                textcoords=trans,
                va="center",
                ha="left",
                size=textsize,
            )
            # Little trick here to actually attach to the left center point.
            axs[0].annotate(
                "",
                xy=(limit, index),
                xycoords="data",
                xytext=(1.15, start_loc - step),
                textcoords=trans,
                arrowprops=dict(
                    arrowstyle="-", edgecolor=edgecolor, shrinkB=0, shrinkA=0
                ),
            )
            step = step + spacing

    # Add rug to axs[0].
    rug = s_obs
    rax = axs[0].inset_axes(bounds=[0.95, 0, 0.05, 1], zorder=1)
    for index, value in rug.items():
        edgecolor = pltcol.to_hex(colormap(norm(value)))
        rax.hlines(y=index, xmin=0, xmax=1, color=edgecolor, alpha=1)
    rax.set_xticks([])
    rax.set_yticks([])
    rax.margins(0.02)  # Equivalent to axs[0].
    rax.axis("off")

    # Loess breaks in axs[1] due to repeated values. Drop zero for a "fix".
    # TODO: See if we can get LOESS to work despite repeated values.
    reg_x = s_pred.loc[s_obs != 0] if delta_alpha and loess else s_pred
    reg_y = s_obs.loc[s_obs != 0] if delta_alpha and loess else s_obs

    # Replace color array for marginal plot if deta_alpha and loess.
    if delta_alpha and loess:
        colors = colormap(norm(s_obs.loc[s_obs != 0].values))
        dalpha = dalpha[s_obs[s_obs != 0].index]
        alphas = (dalpha - np.min(dalpha)) / (np.max(dalpha) - np.min(dalpha))
        alphas.loc[alphas < 0.3] = 0.3
        colors[:, 3] = alphas

    sns.regplot(
        x=reg_x,
        y=reg_y,
        ax=axs[1],
        lowess=loess,
        line_kws={
            "color": "black",
            "linewidth": 1,
            "linestyle": "dashed",
            "alpha": 0.5,
        },
        scatter_kws={"color": colors},
    )

    # Other adaptations bottom ax.
    diagonal = np.linspace(*axs[1].get_xlim())
    axs[1].plot(diagonal, diagonal, color="black", lw=0.8)
    axs[1].grid(
        which="major",
        axis="x",
        lw=1,
        color="black",
        alpha=0.2,
    )
    axs[1].set_xlabel(
        "Prediction ({})".format(r"$\hat {}$".format(notation)),
        size=textsize + 3,
    )
    axs[1].set_ylabel("Observed {}".format(r"${}$".format(notation)))

    # Title, make sure subplots are attached and save to path.
    plt.suptitle(title, y=0.94, size=textsize + 2)  # Default is a bit high.
    plt.subplots_adjust(hspace=0)
    if path is not None:
        plt.savefig(path, dpi=300, bbox_inches="tight")
        log.info(f"Saved figure to {path}.")
        plt.close()


def make_error_plots(df, column_sets, out_path):
    """Makes error plots for all submissions in df."""
    level = "cm"
    task = datautils.determine_task(df)
    # Prepare a label columns (e.g. "CAR 2021-07").
    df = df.join(pd.read_parquet(os.path.join(DATA_DIR, "country.parquet")))
    df = df.join(pd.read_parquet(os.path.join(DATA_DIR, "month.parquet")))
    df["date_str"] = df["year"].astype(str) + "-" + df["month"].astype(str)
    df["lab"] = df["isoab"] + " " + df["date_str"]

    # Add ensembles to the sets to roll over.
    sets = column_sets[task][level].copy()
    if task == 1:
        ensembles = {
            f"ensemble": f"ensemble",
            f"w_ensemble": f"w_ensemble",
        }
    if task == 2:
        ensembles = {f"ensemble": {i: f"ensemble_s{i}" for i in range(2, 8)}}
    sets.update(ensembles)

    # Plot for t1 or t2.
    for team_id, columns in sets.items():
        if task == 1:
            for step, t in enumerate(TIMEFRAMES[task], 2):
                plot_mc_delta(
                    s_pred=df.loc[t, columns],
                    s_obs=df.loc[t, config.COL_OBS_T1],
                    s_lab=df.loc[t, "lab"],
                    title=f"Error {team_id}, step {step}",
                    path=os.path.join(
                        out_path, f"t1_{level}_errorp_{team_id}_s{step}.png"
                    ),
                    loess=False,
                    limit=4,
                )
            # Also plot for sc.
            plot_mc_delta(
                s_pred=df[columns],
                s_obs=df[config.COL_OBS_T1],
                s_lab=df["lab"],
                title=f"Error {team_id}, step-combined",
                path=os.path.join(
                    out_path, f"t1_{level}_errorp_{team_id}_sc.png"
                ),
                loess=False,
                limit=4,
            )
        else:
            for step, col in columns.items():
                plot_mc_delta(
                    s_pred=df[col],
                    s_obs=df[config.COL_OBS.format(step)],
                    s_lab=df["lab"],
                    title=f"Error {team_id}, step {step}",
                    path=os.path.join(
                        out_path, f"t2_{level}_errorp_{team_id}_s{step}.png"
                    ),
                    loess=False,
                    limit=4,
                )


def get_annotation_coords(fig, ax):
    """Get coords from annotations."""
    annotations = [
        child
        for child in ax.get_children()
        if isinstance(child, text.Annotation)
    ]
    coords = {}
    for annotation in annotations:
        if "predicted" not in annotation._text and "-" in annotation._text:
            # For the right-end x coords, use bbox.
            tbbox = annotation.get_window_extent(
                renderer=fig.canvas.get_renderer()
            )
            # Right-end is [1][0] for x, [1][1] for y, in data coords.
            right_end = ax.transData.inverted().transform(tbbox)[1][0]
            # Convert to fraction coords.
            x = ax.transLimits.transform((right_end, 1))[0]
            # Also transform y data coords of annot to fraction coords.
            y = ax.transLimits.transform((1, annotation._y))[1]
            pg_id = annotation._text.split(" ")[0]
            coords[int(pg_id)] = (x, y)
    return coords


def plot_mc_delta_pgm(
    s_pred,
    s_obs,
    s_lab,
    path,
    title,
    limit=None,
    bbox=[-18.5, 52.0, -35.5, 38.0],
    mc_cmap="RdBu_r",
    map_cmap="BrBG_r",
):
    """Plots error plots along with geographic locators."""
    delta = s_pred - s_obs
    delta.name = "delta"
    gdf = mapdata.gdf_from_series_patch(delta)

    # Gridspec figure.
    fig = plt.figure(figsize=(15, 6))
    gs0 = gridspec.GridSpec(
        1, 2, figure=fig, wspace=0.4, width_ratios=[1.2, 2]
    )
    gs00 = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=gs0[0], height_ratios=[4, 1]
    )
    ax1 = fig.add_subplot(gs00[0])
    ax2 = fig.add_subplot(gs00[1], sharex=ax1)
    ax1.get_xaxis().set_visible(False)

    # Plot error plot into ax1 and ax2.
    plot_mc_delta(
        s_pred=s_pred,
        s_obs=s_obs,
        s_lab=s_lab,
        title=title,
        cmap=mc_cmap,
        axs=[ax1, ax2],
        limit=limit,
    )

    # Set up ax3 (map).
    gs01 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs0[1])
    ax3 = fig.add_subplot(gs01[0])

    # Subset gdf according to supplied bbox.
    gdf_sub = gdf.cx[bbox[0] : bbox[1], bbox[2] : bbox[3]]

    # Optional visual dropping of low-range data in map.
    gdf_sub.loc[
        (gdf_sub.delta > -0.1) & (gdf_sub.delta < 0.1), "delta"
    ] = np.nan

    # Determine vmin and vmax and plot map.
    vmin, vmax = -max(gdf.delta.max(), abs(gdf.delta.min())), max(
        gdf.delta.max(), abs(gdf.delta.min())
    )
    gdf_sub.plot(ax=ax3, column="delta", cmap=map_cmap, vmin=vmin, vmax=vmax)

    # Also plot country borders.
    geom_c = mapdata.gdf_cm.copy()
    geom_c = geom_c.cx[bbox[0] : bbox[1], bbox[2] : bbox[3]]  # type: ignore
    geom_c.geometry.boundary.plot(
        ax=ax3,
        edgecolor="black",
        facecolor="none",
        linewidth=0.5,
    )

    # Ax3 adjustments.
    ax3 = maps.adjust_axlims(ax3, bbox)
    ax3.tick_params(
        top=False,
        bottom=False,
        left=False,
        right=False,
        labelleft=False,
        labelbottom=False,
    )

    # Make ax for colorbar and add to canvas.
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    # Fill in the colorbar and adjust the ticks.
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=map_cmap, norm=norm)
    sm._A = []  # pylint: disable=protected-access
    cbar = plt.colorbar(sm, cax=cax)

    # Prepare label-cell coordinates and plot connections.
    textcoords = get_annotation_coords(fig, ax1)
    for pg_id, textcoord in textcoords.items():
        mapcoord = gdf_sub.loc[pg_id, "geometry"].centroid.coords[0]
        con = ConnectionPatch(
            xyA=textcoord,
            xyB=mapcoord,
            coordsA="axes fraction",
            coordsB="data",
            axesA=ax1,
            axesB=ax3,
            color="black",
            alpha=0.5,
        )
        ax3.add_artist(con)

    if path is not None:
        plt.savefig(path, dpi=200)
        plt.close()


def make_error_plots_pgm(df, column_sets, out_path):
    """Makes error plots for all submissions in df."""
    level = "pgm"
    task = datautils.determine_task(df)
    # Prepare a label column (e.g. "123456 2021-07").
    df = df.join(pd.read_parquet(os.path.join(DATA_DIR, "month.parquet")))
    df["date_str"] = df["year"].astype(str) + "-" + df["month"].astype(str)
    df.reset_index(inplace=True)
    df["lab"] = df["pg_id"].astype(str) + " " + df["date_str"]
    df = df.set_index(["month_id", "pg_id"]).sort_index()

    # Add ensembles to the sets to roll over.
    sets = column_sets[task][level].copy()
    if task == 1:
        ensembles = {
            f"ensemble": f"ensemble",
            f"w_ensemble": f"w_ensemble",
        }
    if task == 2:
        ensembles = {f"ensemble": {i: f"ensemble_s{i}" for i in range(2, 8)}}
    sets.update(ensembles)

    # Plot for t1 or t2.
    for team_id, columns in sets.items():
        if task == 1:
            for step, t in enumerate(TIMEFRAMES[task], 2):
                date_str = datautils.to_datestr(t)
                plot_mc_delta_pgm(
                    s_pred=df.loc[t, columns],  # columns is singular here.
                    s_obs=df.loc[t, config.COL_OBS_T1],
                    s_lab=df.loc[t, "lab"],
                    path=os.path.join(
                        out_path, f"t1_{level}_errorp_{team_id}_s{step}.png"
                    ),
                    title=f"{columns}, {date_str}",
                    limit=6,
                )
            # Also plot for sc.
            # plot_mc_delta_pgm(
            #     s_pred=df[columns],
            #     s_obs=df[config.COL_OBS_T1],
            #     s_lab=df["lab"],
            #     path=os.path.join(
            #         out_path, f"t1_{level}_errorp_{team_id}_sc.png"
            #     ),
            # )
        else:
            for step, col in columns.items():
                if step in (2, 7):
                    t = 474
                    date_str = datautils.to_datestr(t)
                    plot_mc_delta_pgm(
                        s_pred=df.loc[t, col],
                        s_obs=df.loc[t, config.COL_OBS.format(step)],
                        s_lab=df.loc[t, "lab"],
                        path=os.path.join(
                            out_path,
                            f"t2_{level}_errorp_{team_id}_s{step}.png",
                        ),
                        title=f"{col} s{step}, {date_str}",
                        limit=4,
                    )
