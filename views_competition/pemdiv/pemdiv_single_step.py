import pandas as pd
import numpy as np
from scipy.spatial import ConvexHull
import pickle
from views_competition.pemdiv import pemdiv


def sp_qlag_pg(pg_id, steps=1):
    """Compute queen-contiguity spatial lags as an step x step kernel"""
    """Will produce a list of ids representing a steps x steps kernel"""
    """This kernel can be used as a convolution kernel"""
    """Author = Mihai"""
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


def map_pgids(df):

    PG_STRIDE = 720

    # get unique pgids

    #    pgids=np.array(list({idx[1] for idx in df.index.values}))
    pgids = df["pg_id"].values

    pgids = np.sort(pgids)

    # convert pgids to longitudes and latitudes

    longitudes = pgids % PG_STRIDE
    latitudes = pgids // PG_STRIDE

    latmin = np.min(latitudes)
    latmax = np.max(latitudes)
    longmin = np.min(longitudes)
    longmax = np.max(longitudes)

    latrange = latmax - latmin
    longrange = longmax - longmin

    # shift to a set of indices that starts at [0,0]

    latitudes -= latmin
    longitudes -= longmin

    # make dicts to transform between pgids and (long,lat) coordinates

    pgid_to_longlat = {}
    longlat_to_pgid = {}

    pgid_to_index = {}
    index_to_pgid = {}

    for i, pgid in enumerate(pgids):
        pgid_to_longlat[pgid] = (longitudes[i], latitudes[i])
        longlat_to_pgid[(longitudes[i], latitudes[i])] = pgid
        pgid_to_index[pgid] = i
        index_to_pgid[i] = pgid

    return (
        list(pgids),
        pgid_to_longlat,
        longlat_to_pgid,
        pgid_to_index,
        index_to_pgid,
        longitudes,
        latitudes,
    )


def get_pemdiv(df, cap, predicted_columns, bench_columns):

    print(df["pg_id"].values)
    #    pg_ids=np.array(list({idx[1] for idx in df.index.values}))
    pg_ids = df["pg_id"].values
    pg_ids = np.sort(pg_ids)

    pg_ids = list(pg_ids)

    npgids = len(pg_ids)

    index_to_pgid = {}
    pgid_to_index = {}
    for idx, pg_id in enumerate(pg_ids):
        index_to_pgid[idx] = pg_id
        pgid_to_index[pg_id] = idx

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

        with open("pemdiv_pg_dist_matrix.pickle", "wb") as f:
            pickle.dump(dist_matrix, f, pickle.HIGHEST_PROTOCOL)
    else:
        with open("pemdiv_pg_dist_matrix.pickle", "rb") as f:
            dist_matrix = pickle.load(f)

    heads = []
    tails = []
    capacities = []
    unit_costs = []
    unconnected = []
    for ipgid in range(size):
        at_least_one = False
        pgidi = index_to_pgid[ipgid]
        for jpgid in range(size):
            if dist_matrix[ipgid, jpgid] > 0.0:
                at_least_one = True
                pgidj = index_to_pgid[jpgid]
                heads.append(ipgid)
                tails.append(jpgid)
                capacities.append(int(cap))
                unit_costs.append(int(dist_matrix[ipgid, jpgid]))
        if not (at_least_one):

            unconnected.append(pgidi)

    for unc in unconnected:
        df.drop(pgid_to_index[unc], inplace=True)

    print(df["pg_id"].values)
    #    pg_ids=np.array(list({idx[1] for idx in df.index.values}))
    pg_ids = df["pg_id"].values
    pg_ids = np.sort(pg_ids)

    pg_ids = list(pg_ids)

    npgids = len(pg_ids)

    index_to_pgid = {}
    pgid_to_index = {}
    for idx, pg_id in enumerate(pg_ids):
        index_to_pgid[idx] = pg_id
        pgid_to_index[pg_id] = idx

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

        with open("pemdiv_pg_dist_matrix.pickle", "wb") as f:
            pickle.dump(dist_matrix, f, pickle.HIGHEST_PROTOCOL)
    else:
        with open("pemdiv_pg_dist_matrix.pickle", "rb") as f:
            dist_matrix = pickle.load(f)

    heads = []
    tails = []
    capacities = []
    unit_costs = []
    unconnected = []
    for ipgid in range(size):
        at_least_one = False
        pgidi = index_to_pgid[ipgid]
        for jpgid in range(size):
            if dist_matrix[ipgid, jpgid] > 0.0:
                at_least_one = True
                pgidj = index_to_pgid[jpgid]
                heads.append(ipgid)
                tails.append(jpgid)
                capacities.append(int(cap))
                unit_costs.append(int(dist_matrix[ipgid, jpgid]))
        if not (at_least_one):
            unconnected.append(pgidi)

    pemdiv.add_int_node_id_to_df(df)

    panel_edges_dict = {}

    panel_edges_dict["edge_heads"] = heads
    panel_edges_dict["edge_tails"] = tails
    panel_edges_dict["capacities"] = capacities
    panel_edges_dict["unit_costs"] = unit_costs

    (
        _,
        pgid_to_longlat,
        longlat_to_pgid,
        pgid_to_index,
        index_to_pgid,
        longitudes,
        latitudes,
    ) = map_pgids(df)

    longlatpoints = np.zeros((npgids, 2))
    longlatpoints[:, 0] = longitudes
    longlatpoints[:, 1] = latitudes

    hull = ConvexHull(longlatpoints)

    space_corners = []

    for simplex in hull.simplices:
        point = (
            int(longlatpoints[simplex, 0][0]),
            int(longlatpoints[simplex, 1][0]),
        )
        corner = longlat_to_pgid[point]
        if corner not in space_corners:
            space_corners.append(corner)
        point = (
            int(longlatpoints[simplex, 0][1]),
            int(longlatpoints[simplex, 1][1]),
        )
        corner = longlat_to_pgid[point]
        if corner not in space_corners:
            space_corners.append(corner)

    corners = [pgid_to_index[icorn] for icorn in space_corners]

    extra_cost_per_unit = pemdiv.calc_min_extra_cost_per_unit_for_panel(
        corner_ids=corners, edges_dict=panel_edges_dict
    )

    int_supplies_scalar = 1000

    results = {}

    for predicted_column, benchmark_column in zip(
        predicted_columns, bench_columns
    ):
        pemdiv.add_supplies_int64(
            df,
            "node_id",
            predicted_column,
            benchmark_column,
            int_supplies_scalar=int_supplies_scalar,
        )
        suppliesList = list(df["supplies"])
        panel_nodes_supply_dict = pemdiv.make_nodes_supplies_dict(suppliesList)
        sum_predict_mass = (
            (df[predicted_column] * int_supplies_scalar).astype("int64").sum()
        )
        sum_actual_mass = (
            (df[benchmark_column] * int_supplies_scalar).astype("int64").sum()
        )
        output = pemdiv.calc_pemdiv(
            edge_dict=panel_edges_dict,
            node_supplies_dict=panel_nodes_supply_dict,
            extra_cost_per_unit=extra_cost_per_unit,
        )
        key = predicted_column + "_" + benchmark_column
        print(key)
        results[key] = output["score"]
        boring = df.pop("supplies")

    return results
