"""Line plot"""

from typing import List
import os
import logging
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt, image as mpimg
from matplotlib.offsetbox import AnchoredText

from views_competition.plot import utilities
from views_competition import config, DATA_DIR, OUTPUT_DIR

log = logging.getLogger(__name__)


def add_years_countryinfos_pgm(df):
    df_info = (
        pd.read_parquet(os.path.join(DATA_DIR, "skeleton_pgm_africa.parquet"))
        .reset_index(["year", "country_id"])
        .drop(columns=["in_africa"])
    )
    df = df.merge(df_info, left_index=True, right_index=True, how="left")
    return df


def plot_lines_cols_cm(
    df: pd.DataFrame,
    start: int,
    end: int,
    cols: List[str],
    country_id: int = None,
    country_name: str = None,
    run_id: str = "development",
    sort_last: bool = True,
    fig_scale: float = 1,
    ymin: float = 0,
    ymax: float = 1,
    tick_xfreq: int = 4,
    tick_yfreq: float = 0.2,
    h_line: int = None,
    cmap: str = "tab20",
    colors: List[str] = None,
    line_width: int = 3,
    text_size: int = 15,
    text_box: str = None,
    views_logo: bool = False,
    title: str = None,
    legend_title: str = None,
    dpi: int = 300,
    path: str = None,
):

    """
    Plot for multiple models/features/ensembles for single country.
    Args:
        df: Pandas dataframe with multi-index "month_id","country_id".
        run_id: String of identifier of run to put into default textbox.
        start: Either month_id or step to subset.
        end: Either month_id or step to subset.
        cols: Name of the columns (e.g. model names, feature names, ensemble names).
        country_ids: Country_id to plot.
        country_names: Country name to plot instead of id.
        fig_scale: Figure size scaler.
        ymin: Minimum to plot the data to.
        ymax: Maximum to plot the data to.
        tick_freq: Set the number of maximum ticks.
        line_width: Deinfe thickness of plotted lines.
        text_size: Base text size for all text on plot. Title is textsize +5.
        title: Title to add to the figure.
        dpi: Dots per inch, defaults to 300. Lower to reduce file size.
        path: Destination to write figure file.
    """

    df = df.copy()

    # Add years if they are not in df.
    if not "year" in df.columns:
        df = df.join(pd.read_parquet(os.path.join(DATA_DIR, "month.parquet")))

    # Add country_names if they are not in df.
    if not "name" in df.columns:
        df = df.join(
            pd.read_parquet(os.path.join(DATA_DIR, "country.parquet"))
        )

    df["date_str"] = (
        df.year.astype(str) + "-" + df.month.astype(str)
    )  # Add string with month-yr

    # Subset period.
    if start in list(range(1, 36)):
        first_m = df.index.min()[0]
        start_m = first_m + start - 1
    else:
        start_m = start

    if end in list(range(1, 36)):
        first_m = df.index.min()[0]
        end_m = first_m + end - 1
    else:
        end_m = end

    df_sub = df.loc[start_m:end_m]

    # Subset country/countries.
    if country_id is not None:
        df_sub = df_sub.swaplevel().loc[country_id].sort_index()

    if country_name is not None:
        country_id = (
            df_sub[df_sub["name"] == country_name]
            .swaplevel()
            .sort_index()
            .index.get_level_values(0)
            .drop_duplicates()
            .astype(int)[0]
        )
        df_sub = df_sub.swaplevel().loc[country_id].sort_index()

    # Country name.
    countryname = df.swaplevel().loc[country_id, "name"].iloc[0]

    # Set up figure space.
    fig = plt.figure(figsize=(20 * fig_scale, 7.5 * fig_scale))
    ax = fig.add_subplot(111)

    # Set up y-axis lim.
    plt.ylim([ymin, ymax])

    # Remove plot framelines.
    for spine in plt.gca().spines.values():
        spine.set_visible(False)

    # Format grid lines.
    ax.grid(
        which="major",
        axis="y",
        linestyle="--",
        dashes=(2, 3),
        lw=1,
        color="black",
        alpha=0.2,
    )

    # Format ticks.
    plt.tick_params(
        colors="black", labelsize=text_size, bottom=False, left=False
    )
    ax.set_xticks(np.arange(start, end + 1, 1)[::tick_xfreq])
    ax.set_xticklabels(
        list(df_sub["date_str"].drop_duplicates()[::tick_xfreq])
    )

    plt.yticks(np.arange(ymin, ymax + tick_yfreq, tick_yfreq))

    # Set line colours for multiple cols
    n = len(cols)
    cm = plt.cm.get_cmap(cmap)

    if colors is not None:
        colors = colors
    else:
        colors = cm(np.linspace(0, 1, n))

    # Plot.
    for m, clr in zip(cols, range(n)):
        df_cols = df_sub[m].reset_index()
        plt.plot(
            df_cols["month_id"],
            df_cols[m],
            label=cols[clr],
            color=colors[clr],
            linewidth=line_width,
        )

    # Add horizontal line.
    if h_line:
        ax.axhline(y=h_line, color="r", linestyle="--", lw=line_width - 1)

    # Plot title.
    plt.title(title, fontsize=text_size + 5)

    # Plot legend.
    if legend_title is not None:
        l_title = legend_title
    else:
        l_title = f"Names"

    plt.legend(
        loc="upper left",
        fontsize=text_size,
        title_fontsize=text_size,
        bbox_to_anchor=(1, 1),
        borderaxespad=0.2,
        edgecolor="lightgrey",
        fancybox=False,
        title=l_title,
    )._legend_box.align = "left"

    # Add textbox
    if text_box is not None:
        meta = text_box
    else:
        meta = f"Level of analysis: cm\nCountry: {countryname}\nRun: {run_id}"

    anchored_text = AnchoredText(
        meta,
        loc="lower left",
        bbox_to_anchor=(1, 0),
        bbox_transform=ax.transAxes,
        borderpad=0.2,
        prop={"fontsize": text_size - 3},
        frameon=True,
    )
    anchored_text.patch.set_edgecolor("lightgrey")
    ax.add_artist(anchored_text)

    if views_logo:
        url_text = AnchoredText(
            f"http://views.pcr.uu.se",
            loc="lower left",
            bbox_to_anchor=(1, 0.13),
            bbox_transform=ax.transAxes,
            borderpad=0.2,
            prop={"fontsize": text_size - 3},
            frameon=False,
        )
        url_text.patch.set_edgecolor("lightgrey")
        ax.add_artist(url_text)

        ax2 = fig.add_axes([0.905, 0.25, 0.08, 0.08])
        this_dir = os.path.dirname(__file__)
        path_logo_views = os.path.join(this_dir, "logo_transparent.png")
        logo_views = mpimg.imread(path_logo_views)
        ax2.imshow(logo_views, interpolation="none")
        ax2.axis("off")

    # Save
    if path:
        fig.savefig(path, bbox_inches="tight", dpi=dpi, transparent=False)
        log.info(f"Wrote {path}.")


def plot_lines_cols_pgm(
    df: pd.DataFrame,
    start: int,
    end: int,
    pg_id: int,
    cols: List[str],
    run_id: str = "development",
    fig_scale: float = 1,
    ymin: float = 0,
    ymax: float = 1,
    ylabs: List = None,
    tick_xfreq: int = 4,
    tick_yfreq: float = 0.2,
    cmap: str = "tab20",
    colors: List[str] = None,
    line_width: int = 3,
    text_size: int = 15,
    text_box: str = None,
    views_logo: bool = False,
    title: str = None,
    legend_title: str = None,
    dpi: int = 300,
    path: str = None,
):

    """
    Plot for multiple models/features/ensembles for single pgm cell.
    Args:
        df: Pandas dataframe with multi-index "month_id","country_id".
        start: Either month_id or step to subset.
        end: Either month_id or step to subset.
        cols: Name of the columns (e.g. model names, feature names, ensemble names).
        run_id: String of identifier of run to put into default textbox.
        country_ids: Country_id to plot.
        country_names: Country name to plot instead of id.
        fig_scale: Figure size scaler.
        ymin: Minimum to plot the data to.
        ymax: Maximum to plot the data to.
        tick_freq: Set the number of maximum ticks.
        cmap: The matplotlib colormap to be altered.
        line_width: Deinfe thickness of plotted lines.
        text_size: Base text size for all text on plot. Title is textsize +5.
        text_box: Adjust text in info box.
        views_logo: Add ViEWS logo to plot.
        title: Title to add to the figure.
        legend_title: Customized title to add to legend.
        dpi: Dots per inch, defaults to 300. Lower to reduce file size.
        path: Destination to write figure file.
    """

    df = df.copy()

    df = add_years_countryinfos_pgm(df)

    df["date_str"] = (
        df.year.astype(str) + "-" + df.month.astype(str)
    )  # Add string with month-yr

    # Subset period.
    if start in list(range(1, 36)):
        first_m = df.index.min()[0]
        start_m = first_m + start - 1
    else:
        start_m = start

    if end in list(range(1, 36)):
        first_m = df.index.min()[0]
        end_m = first_m + end - 1
    else:
        end_m = end

    df_sub = df.loc[start_m:end_m]

    # Country name.
    countryname = df.swaplevel().loc[pg_id, "country_name"].iloc[0]

    df_sub = df_sub.swaplevel().loc[pg_id].sort_index()

    # Set up figure space.
    fig = plt.figure(figsize=(20 * fig_scale, 7.5 * fig_scale))
    ax = fig.add_subplot(111)

    # Set up y-axis lim.
    plt.ylim([ymin, ymax])

    # Remove plot framelines.
    for spine in plt.gca().spines.values():
        spine.set_visible(False)

    # Format grid lines.
    ax.grid(
        which="major",
        axis="y",
        linestyle="--",
        dashes=(2, 3),
        lw=1,
        color="black",
        alpha=0.2,
    )

    # Format ticks.
    plt.tick_params(
        colors="black", labelsize=text_size, bottom=False, left=False
    )
    ax.set_xticks(np.arange(start, end + 1, 1)[::tick_xfreq])
    ax.set_xticklabels(
        list(df_sub["date_str"].drop_duplicates()[::tick_xfreq])
    )
    plt.yticks(np.arange(ymin, ymax + tick_yfreq, tick_yfreq))

    # Set line colours for multiple cols
    n = len(cols)
    cm = plt.cm.get_cmap(cmap)
    if colors is not None:
        colors = colors
    else:
        colors = cm(np.linspace(0, 1, n))

    # Plot.
    for m, clr in zip(cols, range(n)):
        df_cols = df_sub[m].reset_index()
        plt.plot(
            df_cols["month_id"],
            df_cols[m],
            label=cols[clr],
            color=colors[clr],
            linewidth=line_width,
        )

    # Plot legend.
    if legend_title is not None:
        l_title = legend_title
    else:
        l_title = f"Names"

    # Plot title.
    plt.title(title, fontsize=text_size + 5)

    # Plot legend.
    plt.legend(
        loc="upper left",
        fontsize=text_size,
        title_fontsize=text_size,
        bbox_to_anchor=(1, 1),
        borderaxespad=0.2,
        edgecolor="lightgrey",
        fancybox=False,
        title=l_title,
    )._legend_box.align = "left"

    # Add textbox
    if text_box is not None:
        meta = text_box
    else:
        meta = f"Level of analysis: pgm\nCountry: {countryname}\nCell ID: {pg_id}\nRun: {run_id}"

    anchored_text = AnchoredText(
        meta,
        loc="lower left",
        bbox_to_anchor=(1, 0),
        bbox_transform=ax.transAxes,
        borderpad=0.2,
        prop={"fontsize": text_size - 3},
        frameon=True,
    )
    anchored_text.patch.set_edgecolor("lightgrey")
    ax.add_artist(anchored_text)

    if views_logo:
        url_text = AnchoredText(
            f"http://views.pcr.uu.se",
            loc="lower left",
            bbox_to_anchor=(1, 0.17),
            bbox_transform=ax.transAxes,
            borderpad=0.2,
            prop={"fontsize": text_size - 3},
            frameon=False,
        )
        url_text.patch.set_edgecolor("lightgrey")
        ax.add_artist(url_text)

        ax2 = fig.add_axes([0.905, 0.28, 0.08, 0.08])
        this_dir = os.path.dirname(__file__)
        path_logo_views = os.path.join(this_dir, "logo_transparent.png")
        logo_views = mpimg.imread(path_logo_views)
        ax2.imshow(logo_views, interpolation="none")
        ax2.axis("off")

    # Save
    if path:
        fig.savefig(path, bbox_inches="tight", dpi=dpi, transparent=False)
        log.info(f"Wrote {path}.")


def make_t1_lineplots(df, column_sets, out_path):
    """Makes lineplots per submission."""
    level = "cm" if "country_id" in df.index.names else "pgm"
    columns = [
        col
        for _, col in column_sets[1][level].items()
        if col not in config.DROPS_ENS_T1
    ]
    colors = {
        team_id: color
        for team_id, color in utilities.get_team_colors()[level].items()
        if team_id not in config.DROPS_ENS_T1
    }  # NB: assumes index is ordered!
    if level == "cm":
        for country in config.LINE_COUNTRIES:
            plot_lines_cols_cm(
                df=df[columns],
                start=490,
                end=495,
                cols=sorted(columns),
                colors=[
                    utilities.get_team_colors()[level][col]
                    for col in sorted(columns)
                ],
                country_name=country,
                ymax=4,
                ymin=-4,
                tick_yfreq=1,
                tick_xfreq=1,
                legend_title="Teams",
                run_id="Prediction Competition 2020",
                path=os.path.join(out_path, f"t1_cm_lines_{country}.png"),
            )
    else:
        for pgid, ymax in zip(config.LINE_PGIDS, [1, 1, 1, 4]):
            plot_lines_cols_pgm(
                df=df[columns],
                start=490,
                end=495,
                cols=sorted(columns),
                colors=[
                    utilities.get_team_colors()[level][col]
                    for col in sorted(columns)
                ],
                pg_id=pgid,
                ymax=ymax,
                ymin=-ymax,
                tick_yfreq=1,
                tick_xfreq=1,
                legend_title="Teams",
                run_id="Prediction Competition 2020",
                path=os.path.join(out_path, f"t1_pgm_lines_{pgid}.png"),
            )
        # Also for selected pg regions.
        for name, region in config.REGION_PGIDS.items():
            region_df = df.swaplevel().loc[region].swaplevel()
            region_df = region_df.groupby(level=0).mean()
            region_df["pg_id"] = 114918  # Placeholder to multilevel.
            region_df = region_df.reset_index().set_index(
                ["month_id", "pg_id"]
            )
            plot_lines_cols_pgm(
                df=region_df[columns],
                start=490,
                end=495,
                cols=sorted(columns),
                colors=[
                    utilities.get_team_colors()[level][col]
                    for col in sorted(columns)
                ],
                pg_id=114918,
                ymax=1,
                ymin=-1,
                tick_yfreq=1,
                tick_xfreq=1,
                legend_title="Teams",
                run_id="Prediction Competition 2020",
                text_box=f"Level of analysis: pgm\nRegion: {name}\nRun: Prediction Competition 2020",
                path=os.path.join(out_path, f"t1_pgm_lines_{name}_avg.png"),
            )


def make_actual_lineplots(level, out_path):
    """Makes lineplots for the selected actuals."""
    if level == "cm":
        df = pd.read_parquet(
            os.path.join(DATA_DIR, "ged_cm_postpatch.parquet")
        )[["ged_best_sb"]]
        df["ln_ged_best_sb"] = np.log1p(df["ged_best_sb"])
        for country in config.LINE_COUNTRIES:
            plot_lines_cols_cm(
                df=df,
                start=480,
                end=488,
                cols=["ln_ged_best_sb"],
                country_name=country,
                ymax=6,
                ymin=0,
                tick_yfreq=1,
                tick_xfreq=1,
                legend_title="",  # "History of violence"
                run_id="Prediction Competition 2020",
                path=os.path.join(
                    out_path, f"t1_cm_lines_lngedbest_{country}.png"
                ),
            )
    else:
        df = pd.read_parquet(
            os.path.join(DATA_DIR, "ged_pgm_postpatch.parquet")
        )[["ged_best_sb"]]
        df["ln_ged_best_sb"] = np.log1p(df["ged_best_sb"])
        for pgid in config.LINE_PGIDS:
            plot_lines_cols_pgm(
                df=df,
                start=480,
                end=488,
                cols=["ln_ged_best_sb"],
                pg_id=pgid,
                ymax=2,
                ymin=-1,
                tick_yfreq=0.2,
                tick_xfreq=1,
                legend_title="",
                run_id="Prediction Competition 2020",
                path=os.path.join(
                    out_path, f"t1_pgm_lines_lngedbest_{pgid}.png"
                ),
            )


def make_mse_lines(level):
    """Make MSE lines with box-whisker."""
    colors = utilities.get_team_colors()
    medianprops = dict(linewidth=2.5, color='firebrick')
    obs = pd.read_csv(os.path.join(OUTPUT_DIR, "data", f"t1_{level}_ss.csv"))
    obs = obs[[col for col in obs if "d_ln_ged" in col]]
    score = pd.read_csv(
        os.path.join(OUTPUT_DIR, "tables", f"t1_{level}_mse.csv"),
        index_col=[0],
    )
    # Fig.
    meanlineprops = dict(linestyle="--", linewidth=2.5, color="grey")
    fig = plt.figure(figsize=(7, 10))
    gs = fig.add_gridspec(2, hspace=0)
    axs = gs.subplots(sharex=True)
    # Top
    for col in score.T:
        if col != "ensemble":
            axs[0].plot(
                score.T.index,
                score.T[col],
                label=col,
                color=colors[level][col],
                lw=3,
            )
    axs[0].grid(b=True, axis="x", color="grey", linestyle="--", alpha=0.2)
    axs[0].legend(
        loc="best", bbox_to_anchor=(1, 1, 0.45, 0), frameon=False
    )
    axs[0].set_ylabel("MSE", fontsize=14)
    # Bottom
    data = []
    for col in obs:  # Note: assumes order.
        data.append(obs[obs[col] != 0][col].values)
    axs[1].boxplot(
        data,
        positions=[0, 1, 2, 3, 4, 5],
        medianprops=medianprops,
        showmeans=True,
        meanprops=meanlineprops,
        meanline=True,
    )  # Adjust positions, thank you.
    for i, col in enumerate(obs):
        axs[1].scatter(i, obs[col].mean(), color="black")
    # axs[1].set_ylim([-6, 6])
    axs[1].set_xticklabels([f"s{i}" for i in range(2, 8)])
    axs[1].grid(b=True, axis="x", color="grey", linestyle="--", alpha=0.2)
    axs[1].set_ylabel("Observed delta", fontsize=14)
    plt.savefig(
        os.path.join(OUTPUT_DIR, "graphs/line", f"{level}_mse_box.png"),
        dpi=200,
        bbox_inches="tight",
    )
