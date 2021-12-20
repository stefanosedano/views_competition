"""Compute PEMDIV scores for task one, step-combined."""

import os
import pickle
import pandas as pd
import numpy as np

import matplotlib
import matplotlib.colors as colors
import matplotlib.pyplot as plt

from views_competition.plot import maps
from views_competition.pemdiv import pemdiv_single_step
from views_competition.pemdiv.pemdiv import *
from views_competition import OUTPUT_DIR

mapdata = maps.MapData()


def compute_pemdiv():
    """Computes PEMDIV for task one sc entries."""
    # Get data table for Task 1
    df_bench_task1 = pd.read_csv(
        os.path.join(OUTPUT_DIR, "data", "t1_pgm.csv")
    )
    df_bench_task1 = df_bench_task1.sort_values(by=["pg_id", "month_id"])

    # Get unique pg_ids and initialise parameters for computing the distance matrix
    pg_ids = list(pd.unique(df_bench_task1["pg_id"].values))
    size = len(pg_ids)
    dist_matrix = np.zeros((size, size))
    interactions = []
    row_col_names = []
    row_col_names_int = []
    nspacestep = 1

    def sp_qlag_pg(pg_id, steps=1):
        """Compute queen-contiguity spatial lags as an step x step kernel.

        Will produce a list of ids representing a steps x steps kernel.
        This kernel can be used as a convolution kernel
        """
        lags = []
        if steps == 0:
            lags = [pg_id]
            return lags
        else:
            lags.extend(sp_qlag_pg(pg_id - 719, steps - 1))
            lags.extend(sp_qlag_pg(pg_id + 719, steps - 1))
            lags.extend(sp_qlag_pg(pg_id + 720, steps - 1))
            lags.extend(sp_qlag_pg(pg_id - 720, steps - 1))
            lags.extend(sp_qlag_pg(pg_id - 721, steps - 1))
            lags.extend(sp_qlag_pg(pg_id + 721, steps - 1))
            lags.extend(sp_qlag_pg(pg_id + 1, steps - 1))
            lags.extend(sp_qlag_pg(pg_id - 1, steps - 1))
            lags.extend(sp_qlag_pg(pg_id, steps - 1))
            return list(set(lags))

    # Compute the distance matrix, specifying which cells are allowed to exchange
    # information with each other in the spatial dimensions. Here a space-step of 1
    # is used, so cells are only allowed to communicate with their immediate
    # neighbours.

    # This is very time-consuming, so save to disk once done.
    compute_matrix = True
    if compute_matrix:
        for pg_id in pg_ids:
            interactions.append(sp_qlag_pg(pg_id, steps=nspacestep))
            row_col_names.append(str(pg_id))
            row_col_names_int.append(pg_id)

        for i in range(size):
            interaction = interactions[i]
            for pg_id in interaction:
                if pg_id in pg_ids:
                    j = pg_ids.index(pg_id)
                    if i != j:
                        dist_matrix[i, j] = 1

        with open(
            os.path.join(OUTPUT_DIR, "data/pemdiv_pg_dist_matrix.pickle"), "wb"
        ) as f:
            pickle.dump(dist_matrix, f, pickle.HIGHEST_PROTOCOL)
    else:
        with open(
            os.path.join(OUTPUT_DIR, "data/pemdiv_pg_dist_matrix.pickle"), "rb"
        ) as f:
            dist_matrix = pickle.load(f)

    # Create a list of columns containing predictions, and a parallel list of the
    # column containing the quantities to compare with.
    predicted_columns = [
        "benchmark",
        "no_change",
        "ensemble",
        "w_ensemble",
        "vestby_xgb_fit",
        "fritz",
        "lindholm",
        "chadefaux",
        "vestby_rf_fit",
        "radford",
        "brandt",
        "dorazio",
        "hultman",
    ]

    bench_columns = ["d_ln_ged_best_sb" for _ in predicted_columns]

    print(predicted_columns)
    print(bench_columns)

    # Add PEMDIV node ids to dataframe.
    add_int_node_id_to_df(df_bench_task1)

    # Generate PEMDIV graph - this is expensive, so save to disk once done
    compute_edges = True
    if compute_edges:
        panel_edges_dict = make_edgesDict_from_df_panel(
            dist_matrix,
            df_bench_task1,
            row_col_names=row_col_names_int,
            sloc_name_col="pg_id",
            tloc_col="month_id",
            node_id_col="node_id",
        )
        with open(
            os.path.join(
                OUTPUT_DIR, "data/pemdiv_pg_panel_edges_dict_t1.pickle"
            ),
            "wb",
        ) as f:
            pickle.dump(panel_edges_dict, f, pickle.HIGHEST_PROTOCOL)
    else:
        with open(
            os.path.join(
                OUTPUT_DIR, "data/pemdiv_pg_panel_edges_dict_t1.pickle"
            ),
            "rb",
        ) as f:
            panel_edges_dict = pickle.load(f)

    print(len(panel_edges_dict["edge_heads"]))

    i = 0
    for key in panel_edges_dict:
        i += 1
        if i > 10:
            break
        print(key)

    # Specify 'corners' of graph (found by building a complex hull around the map
    # of allowed pg cells) and compute the cost of moving probability mass through
    # the graph.
    if True:
        space_corners = [
            62356,
            80317,
            101287,
            146623,
            150791,
            153670,
            154390,
            173950,
            174669,
            181068,
            182523,
            183253,
            183263,
        ]
        corners = []
        for spc_crn in space_corners:
            corner = df_bench_task1[
                (df_bench_task1["pg_id"] == spc_crn)
                & (
                    (df_bench_task1["month_id"] == 490)
                    | (df_bench_task1["month_id"] == 495)
                )
            ]["node_id"].values.tolist()
            corners.append(corner[0])
            corners.append(corner[1])

        extra_cost_per_unit = calc_min_extra_cost_per_unit_for_panel(
            corner_ids=corners, edges_dict=panel_edges_dict
        )
        extra_cost_per_unit
    else:
        extra_cost_per_unit = 74.0

    int_supplies_scalar = 1000

    # Compute, print out, and save to dictionary, PEMDIV values for all prediction
    # columns
    results_task1 = {}
    for predicted_column, benchmark_column in zip(
        predicted_columns, bench_columns
    ):
        print(predicted_column, benchmark_column)
        add_supplies_int64(
            df_bench_task1,
            "node_id",
            predicted_column,
            benchmark_column,
            int_supplies_scalar=int_supplies_scalar,
        )
        suppliesList = list(df_bench_task1["supplies"])
        panel_nodes_supply_dict = make_nodes_supplies_dict(suppliesList)
        sum_predict_mass = (
            (df_bench_task1[predicted_column] * int_supplies_scalar)
            .astype("int64")
            .sum()
        )
        sum_actual_mass = (
            (df_bench_task1[benchmark_column] * int_supplies_scalar)
            .astype("int64")
            .sum()
        )
        output = calc_pemdiv(
            edge_dict=panel_edges_dict,
            node_supplies_dict=panel_nodes_supply_dict,
            extra_cost_per_unit=extra_cost_per_unit,
        )
        key = predicted_column  # +'_'+benchmark_column

        results_task1[key] = "{:.3f}".format(output["score"] / 1e9)
        not_needed = df_bench_task1.pop("supplies")

    for key in results_task1:
        print(key, results_task1[key])

    periods = ["test 1"]
    results_dict = {}
    for period in periods:
        results_dict[period] = []
    for key in results_task1:
        for period in periods:
            if period in key:
                results_dict[period].append(
                    "{:.3f}".format(results_task1[key]["score"] / 1e9)
                )

    df_dict = {}
    for col in predicted_columns:
        df_dict[col] = []
    print(df_dict)
    for key in results_task1:
        df_dict[key].append(results_task1[key])

    df_results = pd.DataFrame(df_dict)
    df_results.to_csv(
        os.path.join(OUTPUT_DIR, "data", "task1_pemdiv_revised.csv"),
        index=False,
    )

    # Recompute PEMDIV for subset of cells for map
    subcells = [
        149426,
        149427,
        149428,
        149429,
        149430,
        148706,
        148707,
        148708,
        148709,
        148710,
        147986,
        147987,
        147988,
        147989,
        147990,
        147266,
        147267,
        147268,
        147269,
        147270,
        146546,
        146547,
        146548,
        146549,
        146550,
    ]

    df_subset = df_bench_task1.loc[df_bench_task1["pg_id"].isin(subcells)]
    # add_int_node_id_to_df(df_subset)

    pg_ids = subcells
    size = len(pg_ids)
    dist_matrix = np.zeros((size, size))
    interactions = []
    row_col_names = []
    row_col_names_int = []
    nspacestep = 1

    compute_matrix = True
    if compute_matrix:
        for pg_id in pg_ids:
            interactions.append(sp_qlag_pg(pg_id, steps=nspacestep))
            row_col_names.append(str(pg_id))
            row_col_names_int.append(pg_id)

        for i in range(size):
            interaction = interactions[i]
            for pg_id in interaction:
                if pg_id in pg_ids:
                    j = pg_ids.index(pg_id)
                    if i != j:
                        dist_matrix[i, j] = 1

        with open(
            os.path.join(
                OUTPUT_DIR, "data/pemdiv_pg_dist_matrix_subset.pickle"
            ),
            "wb",
        ) as f:
            pickle.dump(dist_matrix, f, pickle.HIGHEST_PROTOCOL)
    else:
        with open(
            os.path.join(
                OUTPUT_DIR, "data/pemdiv_pg_dist_matrix_subset.pickle"
            ),
            "rb",
        ) as f:
            dist_matrix = pickle.load(f)

    compute_edges = True
    if compute_edges:
        panel_edges_dict = make_edgesDict_from_df_panel(
            dist_matrix,
            df_subset,
            row_col_names=row_col_names_int,
            sloc_name_col="pg_id",
            tloc_col="month_id",
            node_id_col="node_id",
        )
        with open(
            os.path.join(
                OUTPUT_DIR, "data/pemdiv_pg_panel_edges_dict_subset_t1.pickle"
            ),
            "wb",
        ) as f:
            pickle.dump(panel_edges_dict, f, pickle.HIGHEST_PROTOCOL)
    else:
        with open(
            os.path.join(
                OUTPUT_DIR, "data/pemdiv_pg_panel_edges_dict_subset_t1.pickle"
            ),
            "rb",
        ) as f:
            panel_edges_dict = pickle.load(f)

    month = 490
    if True:
        space_corners = subcells
        corners = []
        for spc_crn in space_corners:
            corner = df_subset[
                (df_subset["pg_id"] == spc_crn)
                & (
                    (df_subset["month_id"] == month)
                    | (df_subset["month_id"] == month + 1)
                )
            ]["node_id"].values.tolist()
            corners.append(corner[0])
            corners.append(corner[1])

        extra_cost_per_unit = calc_min_extra_cost_per_unit_for_panel(
            corner_ids=corners, edges_dict=panel_edges_dict
        )
        extra_cost_per_unit
    else:
        extra_cost_per_unit = 2.5

    int_supplies_scalar = 1000

    results_subset_task1 = {}
    for predicted_column, benchmark_column in zip(
        predicted_columns, bench_columns
    ):
        add_supplies_int64(
            df_subset,
            "node_id",
            predicted_column,
            benchmark_column,
            int_supplies_scalar=int_supplies_scalar,
        )
        suppliesList = list(df_subset["supplies"])
        panel_nodes_supply_dict = make_nodes_supplies_dict(suppliesList)
        sum_predict_mass = (
            (df_subset[predicted_column] * int_supplies_scalar)
            .astype("int64")
            .sum()
        )
        sum_actual_mass = (
            (df_subset[benchmark_column] * int_supplies_scalar)
            .astype("int64")
            .sum()
        )
        output = calc_pemdiv(
            edge_dict=panel_edges_dict,
            node_supplies_dict=panel_nodes_supply_dict,
            extra_cost_per_unit=extra_cost_per_unit,
        )
        key = predicted_column + "_" + benchmark_column
        results_subset_task1[key] = output
        boring = df_subset.pop("supplies")

    for key in results_subset_task1:
        print(key, results_subset_task1[key]["score"] / 1e9)

    df_subset_index = df_subset.set_index(["month_id", "pg_id"], inplace=False)
    df_subset_index = df_subset_index.sort_values(by=["month_id", "pg_id"])

    real_columns = []
    for column in df_subset_index:
        if "d_ln_ged" in column:
            real_columns.append(column)

    for pred_col, real_col in zip(predicted_columns, real_columns):
        location = len(df_subset_index.columns)
        colname = pred_col + "_diff"
        df_subset_index.insert(
            location,
            colname,
            df_subset_index[pred_col].values
            - df_subset_index[real_col].values,
        )

    for pred_col, real_col in zip(predicted_columns, real_columns):
        location = len(df_subset_index.columns)
        colname = pred_col + "_ratio"
        df_subset_index.insert(
            location,
            colname,
            df_subset_index[pred_col].values
            / df_subset_index[real_col].values,
        )

    norm = colors.Normalize(vmin=-4.0, vmax=4.0)
    tick_values = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0]
    s = 2
    t = month
    diff_columns = []
    for column in df_subset_index:
        if "diff" in column:
            diff_columns.append(column)
    plotcol = diff_columns[s - 2]
    print(plotcol)
    for key in results_subset_task1:
        if plotcol[:-4] in key:
            score = np.round(results_subset_task1[key]["score"] / 1e6, 2)
    box_string = "PEMDIV score/$10^{6}$:\n" + str(score)
    fig, ax = maps.plot_map(
        s_patch=df_subset_index[plotcol].loc[t : t + 1],
        cmap="bwr",
        mapdata=mapdata,
        textbox=box_string,
        logodds=False,
        # bbox="mainland_africa", # Disable this to zoom in on a country.
        #    ymin=-df_subset_index[plotcol].loc[t:t+1].max(),
        #    ymax=df_subset_index[plotcol].loc[t:t+1].max(),
        ymin=-3.0,
        ymax=3.0,
        tick_values=tick_values,
        tick_labels=[f"{i}" for i in tick_values],
        ##    country_id=70,
        bbox_pad=[-1, 1, -0.05, 0.05],
        title=f"(Predicted - actual) at month {t}",
    )
    fig.axes[-1].set_ylabel("ln(predicted)-ln(actual)")
    fig.savefig(
        os.path.join(OUTPUT_DIR, "maps/pemdiv", "pemdiv_subset_t1.png")
    )

    norm = colors.Normalize(vmin=-4.0, vmax=4.0)
    tick_values = [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0]
    s = 2
    t = month
    diff_columns = []
    for column in df_subset_index:
        if "diff" in column:
            diff_columns.append(column)
    plotcol = "d_ln_ged_best_sb"
    print(plotcol)
    for key in results_subset_task1:
        if plotcol[:-4] in key:
            score = "0.00"
    box_string = "PEMDIV score/$10^{6}$:\n" + str(score)
    fig, ax = maps.plot_map(
        s_patch=df_subset_index[plotcol].loc[t : t + 1],
        cmap="bwr",
        mapdata=mapdata,
        textbox=box_string,
        logodds=False,
        # bbox="mainland_africa", # Disable this to zoom in on a country.
        #    ymin=-df_subset_index[plotcol].loc[t:t+1].max(),
        #    ymax=df_subset_index[plotcol].loc[t:t+1].max(),
        ymin=-3.0,
        ymax=3.0,
        tick_values=tick_values,
        tick_labels=[f"{i}" for i in tick_values],
        ##    country_id=70,
        bbox_pad=[-1, 1, -0.05, 0.05],
        title=f"Actual values at month {t}",
    )
    fig.axes[-1].set_ylabel("$\Delta $ln(ged-best)")
    fig.axes[-1].set_ylabel("$\Delta $ln(ged-best)")
    fig.axes[0].text(
        12.7, 11.4, "A             B             C             D             E"
    )
    fig.axes[0].text(12.4, 11.7, "1")
    fig.axes[0].text(12.4, 12.2, "2")
    fig.axes[0].text(12.4, 12.7, "3")
    fig.axes[0].text(12.4, 13.2, "4")
    fig.axes[0].text(12.4, 13.7, "5")
    fig.savefig(
        os.path.join(OUTPUT_DIR, "maps/pemdiv", "pemdiv_subset_t1_actual.png")
    )

    norm = colors.Normalize(vmin=-4.0, vmax=4.0)
    tick_values = [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0]
    s = 2
    t = month
    diff_columns = []
    for column in df_subset_index:
        if "diff" in column:
            diff_columns.append(column)
    plotcol = "benchmark"
    print(plotcol)
    for key in results_subset_task1:
        if plotcol[:-4] in key:
            score = "0.19"
    box_string = "PEMDIV score/$10^{6}$:\n" + str(score)
    fig, ax = maps.plot_map(
        s_patch=df_subset_index[plotcol].loc[t : t + 1],
        cmap="bwr",
        mapdata=mapdata,
        textbox=box_string,
        logodds=False,
        # bbox="mainland_africa", # Disable this to zoom in on a country.
        #    ymin=-df_subset_index[plotcol].loc[t:t+1].max(),
        #    ymax=df_subset_index[plotcol].loc[t:t+1].max(),
        ymin=-3.0,
        ymax=3.0,
        tick_values=tick_values,
        tick_labels=[f"{i}" for i in tick_values],
        ##    country_id=70,
        bbox_pad=[-1, 1, -0.05, 0.05],
        title=f"High PEMDIV example at month {t}",
    )
    fig.axes[-1].set_ylabel("$\Delta $ln(ged-best, s=2)")
    fig.axes[0].text(
        12.7, 11.4, "A             B             C             D             E"
    )
    fig.axes[0].text(12.4, 11.7, "1")
    fig.axes[0].text(12.4, 12.2, "2")
    fig.axes[0].text(12.4, 12.7, "3")
    fig.axes[0].text(12.4, 13.2, "4")
    fig.axes[0].text(12.4, 13.7, "5")
    fig.savefig(
        os.path.join(OUTPUT_DIR, "maps/pemdiv", "pemdiv_subset_t1_bench.png")
    )

    norm = colors.Normalize(vmin=-4.0, vmax=4.0)
    tick_values = [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0]
    s = 2
    t = month
    diff_columns = []
    for column in df_subset_index:
        if "diff" in column:
            diff_columns.append(column)
    plotcol = "dorazio"
    print(plotcol)
    for key in results_subset_task1:
        if plotcol[:-4] in key:
            score = "0.12"
    box_string = "PEMDIV score/$10^{6}$:\n" + str(score)
    fig, ax = maps.plot_map(
        s_patch=df_subset_index[plotcol].loc[t : t + 1],
        cmap="bwr",
        mapdata=mapdata,
        textbox=box_string,
        logodds=False,
        # bbox="mainland_africa", # Disable this to zoom in on a country.
        #    ymin=-df_subset_index[plotcol].loc[t:t+1].max(),
        #    ymax=df_subset_index[plotcol].loc[t:t+1].max(),
        ymin=-3.0,
        ymax=3.0,
        tick_values=tick_values,
        tick_labels=[f"{i}" for i in tick_values],
        ##    country_id=70,
        bbox_pad=[-1, 1, -0.05, 0.05],
        title=f"Low PEMDIV example at month {t}",
    )
    fig.axes[-1].set_ylabel("$\Delta $ln(ged-best, s=2)")
    fig.axes[0].text(
        12.7, 11.4, "A             B             C             D             E"
    )
    fig.axes[0].text(12.4, 11.7, "1")
    fig.axes[0].text(12.4, 12.2, "2")
    fig.axes[0].text(12.4, 12.7, "3")
    fig.axes[0].text(12.4, 13.2, "4")
    fig.axes[0].text(12.4, 13.7, "5")
    fig.savefig(
        os.path.join(
            OUTPUT_DIR, "maps/pemdiv", "pemdiv_subset_t1_low_bench.png"
        )
    )

    ###########
    # Continued for t1 ss.
    df_bench_task1 = pd.read_csv(
        os.path.join(OUTPUT_DIR, "data", "t1_pgm_ss.csv")
    )

    # Generate lists of predicted colummns and corresponding list of columns to compare with
    predicted_columns = []
    predicted_columns_wanted = [
        "benchmark",
        "no_change",
        "ensemble",
        "w_ensemble",
        "vestby_xgb_fit",
        "fritz",
        "lindholm",
        "chadefaux",
        "vestby_rf_fit",
        "radford",
        "brandt",
        "dorazio",
        "hultman",
    ]
    for pred in predicted_columns_wanted:
        for col in df_bench_task1:
            if pred in col:
                predicted_columns.append(col)

    bench_columns = [f"d_ln_ged_best_sb_s{i}" for i in range(2, 8)]
    benchmark_columns = (len(predicted_columns_wanted) + 1) * (bench_columns)

    # Since we are now comparing predicted results at a single timestep with real
    # data at the same single timestep, movement of probability mass through time
    # is not meaningful and the single-timestep version of PEMDIV is needed.
    int_supplies_scalar = 1000

    results = pemdiv_single_step.get_pemdiv(
        df_bench_task1,
        int_supplies_scalar,
        predicted_columns,
        benchmark_columns,
    )

    for key in results:
        print(key, results[key])

    periods = ["test 1"]
    results_dict = {}
    for period in periods:
        results_dict[period] = []
    for key in results_task1:
        for period in periods:
            if period in key:
                results_dict[period].append(
                    "{:.3f}".format(results_task1[key]["score"] / 1e9)
                )

    periods = ["s2", "s3", "s4", "s5", "s6", "s7"]
    results_dict = {}
    for period in periods:
        results_dict[period] = []
    for key in results:
        for period in periods:
            if period in key:
                # print(results[key])
                results_dict[period].append(
                    "{:.3f}".format(results[key] / 1e7)
                )

    df_dict = {}
    df_dict["competitor"] = predicted_columns_wanted

    for key in results_dict:
        df_dict[key] = results_dict[key]
    df_results = pd.DataFrame(df_dict)

    df_results.to_csv(
        os.path.join(OUTPUT_DIR, "data", "task1_ss_pemdiv_revised.csv"),
        index=False,
    )


if __name__ == "__main__":
    compute_pemdiv()
