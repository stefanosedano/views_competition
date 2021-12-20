"""Compute PEMDIV scores for task two."""

import os
import pickle
import pandas as pd
import numpy as np

from views_competition.pemdiv.pemdiv import *
from views_competition import OUTPUT_DIR


def compute_pemdiv():
    """Computes PEMDIV for task one sc entries."""
    # Change to location of data table
    df_bench_task2 = pd.read_csv(
        os.path.join(OUTPUT_DIR, "data", "t2_pgm.csv")
    )
    df_bench_task2 = df_bench_task2.sort_values(by=["pg_id", "month_id"])
    bench_columns = []
    bench_columns_calibrated = []

    for column in df_bench_task2.columns:
        #    if "d_ln_ged_best_sb" in column:
        if "d_ln_ged" in column and "09" not in column:
            if "calibrated" in column:
                bench_columns_calibrated.append(column)
            else:
                bench_columns.append(column)
    print(bench_columns)

    # Get unique pg_ids and initialise parameters for computing the distance matrix
    pg_ids = list(pd.unique(df_bench_task2["pg_id"].values))
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

    # Generate list of predicted columns.
    predicted_columns_base = [
        "no_change",
        "benchmark",
        "ensemble",
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
    nbase = len(predicted_columns_base)
    periods = ["s2", "s3", "s4", "s5", "s6", "s7"]
    predicted_columns = [
        pred_col_base + "_" + period
        for pred_col_base in predicted_columns_base
        for period in periods
    ]

    # Generate list of columns to compare with.
    bench_columns = bench_columns * nbase

    # Add PEMDIV node ids to dataframe.
    add_int_node_id_to_df(df_bench_task2)

    # Generate PEMDIV graph - this is expensive, so save to disk once done.
    compute_edges = True
    if compute_edges:
        panel_edges_dict = make_edgesDict_from_df_panel(
            dist_matrix,
            df_bench_task2,
            row_col_names=row_col_names_int,
            sloc_name_col="pg_id",
            tloc_col="month_id",
            node_id_col="node_id",
        )
        with open(
            os.path.join(
                OUTPUT_DIR, "data/pemdiv_pg_panel_edges_dict_t2.pickle"
            ),
            "wb",
        ) as f:
            pickle.dump(panel_edges_dict, f, pickle.HIGHEST_PROTOCOL)
    else:
        with open(
            os.path.join(
                OUTPUT_DIR, "data/pemdiv_pg_panel_edges_dict_t2.pickle"
            ),
            "rb",
        ) as f:
            panel_edges_dict = pickle.load(f)

    print(len(panel_edges_dict["edge_heads"]))

    # Specify "corners" of graph (found by building a complex hull around the map
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
            corner = df_bench_task2[
                (df_bench_task2["pg_id"] == spc_crn)
                & (
                    (df_bench_task2["month_id"] == 445)
                    | (df_bench_task2["month_id"] == 480)
                )
            ]["node_id"].values.tolist()
            corners.append(corner[0])
            corners.append(corner[1])

        extra_cost_per_unit = calc_min_extra_cost_per_unit_for_panel(
            corner_ids=corners, edges_dict=panel_edges_dict
        )
        extra_cost_per_unit
    else:
        extra_cost_per_unit = 89.0

    int_supplies_scalar = 1000

    # Compute, print out, and save to dictionary, PEMDIV values for all prediction
    # columns.
    results_task2 = {}
    for predicted_column, benchmark_column in zip(
        predicted_columns, bench_columns
    ):
        print(predicted_column)
        add_supplies_int64(
            df_bench_task2,
            "node_id",
            predicted_column,
            benchmark_column,
            int_supplies_scalar=int_supplies_scalar,
        )
        suppliesList = list(df_bench_task2["supplies"])
        panel_nodes_supply_dict = make_nodes_supplies_dict(suppliesList)
        sum_predict_mass = (
            (df_bench_task2[predicted_column] * int_supplies_scalar)
            .astype("int64")
            .sum()
        )
        sum_actual_mass = (
            (df_bench_task2[benchmark_column] * int_supplies_scalar)
            .astype("int64")
            .sum()
        )
        output = calc_pemdiv(
            edge_dict=panel_edges_dict,
            node_supplies_dict=panel_nodes_supply_dict,
            extra_cost_per_unit=extra_cost_per_unit,
        )
        key = predicted_column + "_" + benchmark_column
        results_task2[key] = output
        boring = df_bench_task2.pop("supplies")

    for key in results_task2:
        print(key, results_task2[key]["score"] / 1e9)

    results_dict = {}
    for period in periods:
        results_dict[period] = []
    for key in results_task2:
        for period in periods:
            if period in key:
                results_dict[period].append(
                    "{:.3f}".format(results_task2[key]["score"] / 1e9)
                )

    df_dict = {}
    df_dict["competitor"] = predicted_columns_base
    for key in results_dict:
        df_dict[key] = results_dict[key]
    df_results = pd.DataFrame(df_dict)

    df_results.to_csv(
        os.path.join(OUTPUT_DIR, "data", "t2_pemdiv_revised.csv"), index=False
    )


if __name__ == "__main__":
    compute_pemdiv()
