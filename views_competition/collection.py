"""Runner for teamwise-collection of outputs and scores.

! Sets dataframes as global variables at import.
"""

# TODO: Global dataframes on import could be skipped as well.

import os
import tempfile
import pickle
import logging
import pandas as pd

from views_competition import io, CLEAN_DIR, DATA_DIR, PICKLE_DIR
import datautils


log = logging.getLogger(__name__)


log.info("Collecting skeleton...")
# Skeleton for cm.
cm = pd.read_parquet(os.path.join(DATA_DIR, "skeleton_cm_africa.parquet"))
cm = cm.reset_index().set_index(["month_id", "country_id"])
cm = cm.drop(columns=cm.columns)  # Empty dataframe.
cm_t1 = cm.copy().loc[490:495]
cm_t2 = cm.copy().loc[445:480]
cm_t3 = cm.copy().loc[409:444]

# Skeleton for pgm.
pgm = pd.read_parquet(os.path.join(DATA_DIR, "skeleton_pgm_africa.parquet"))
pgm = pgm.reset_index().set_index(["month_id", "pg_id"])
pgm = pgm.drop(columns=pgm.columns)
pgm_t1 = pgm.copy().loc[490:495]
pgm_t2 = pgm.copy().loc[445:480]
pgm_t3 = pgm.copy().loc[409:444]

log.info("Collecting actuals and computing deltas...")
# Add t1 actuals (delta ln ged_best compared to 488).
# Postpatch data is also used for non-benchmark t2 and t3 predictions.
obs_cm_t1 = datautils.add_t1_delta(
    pd.DataFrame(
        pd.read_parquet(os.path.join(DATA_DIR, "ged_cm_postpatch.parquet"))[
            "ged_best_sb"
        ]
    ).loc[488:495]
)
obs_pgm_t1 = datautils.add_t1_delta(
    pd.DataFrame(
        pd.read_parquet(os.path.join(DATA_DIR, "ged_pgm_postpatch.parquet"))[
            "ged_best_sb"
        ]
    ).loc[488:495]
)
cm_t1 = cm_t1.join(obs_cm_t1)
pgm_t1 = pgm_t1.join(obs_pgm_t1)

# Pre-patch actuals to evaluate benchmark (t2, t3) by.
obs_cm_09 = datautils.add_delta_logtransforms(
    (
        pd.DataFrame(
            pd.read_parquet(os.path.join(DATA_DIR, "ged_cm_prepatch.parquet"))[
                "ged_best_sb"
            ]
        )
    ),
    pre_patch=True,
)
obs_pgm_09 = datautils.add_delta_logtransforms(
    (
        pd.DataFrame(
            pd.read_parquet(
                os.path.join(DATA_DIR, "ged_pgm_prepatch.parquet")
            )["ged_best_sb"]
        )
    ),
    pre_patch=True,
)
cm_t2 = cm_t2.join(obs_cm_09)
pgm_t2 = pgm_t2.join(obs_pgm_09)
cm_t3 = cm_t3.join(obs_cm_09)
pgm_t3 = pgm_t3.join(obs_pgm_09)

# Submissions (t2, t3) all evaluated on post-patch data.
obs_cm = datautils.add_delta_logtransforms(
    pd.DataFrame(
        pd.read_parquet(os.path.join(DATA_DIR, "ged_cm_postpatch.parquet"))[
            "ged_best_sb"
        ]
    )
)
obs_pgm = datautils.add_delta_logtransforms(
    pd.DataFrame(
        pd.read_parquet(os.path.join(DATA_DIR, "ged_pgm_postpatch.parquet"))[
            "ged_best_sb"
        ]
    )
)
cm_t2 = cm_t2.join(obs_cm)
pgm_t2 = pgm_t2.join(obs_pgm)
cm_t3 = cm_t3.join(obs_cm)
pgm_t3 = pgm_t3.join(obs_pgm)

log.info("Finished data preparations.")


def add_into_skeleton(df, task, level):
    """Add columns into relevant skeleton table."""
    if task == 3:
        skeleton = cm_t3 if level == "cm" else pgm_t3
        datautils.assign_into_df(skeleton, df)
    if task == 2:
        skeleton = cm_t2 if level == "cm" else pgm_t2
        datautils.assign_into_df(skeleton, df)
    if task == 1:
        skeleton = cm_t1 if level == "cm" else pgm_t1
        datautils.assign_into_df(skeleton, df)


def collect(input_path):
    """Collects all relevant teamwise predictions via a tempdir.

    Column sets are structured (task > level > team > step > col):

    {
        2: {
            "cm": {
                "team_id": {
                    2: "col_s2",
                    3: "col_s3",
                    ...
                }
            }
        }
    }
    """
    column_sets = {
        1: {"cm": {}, "pgm": {}},
        2: {"cm": {}, "pgm": {}},
        3: {"cm": {}, "pgm": {}},
    }

    with tempfile.TemporaryDirectory() as tempdir:
        for root, _, files in os.walk(input_path):
            zipfiles = [os.path.join(root, f) for f in files if ".zip" in f]
            for zipfile in zipfiles:
                log.info("Collecting %s.", os.path.basename(zipfile))
                # Initialize a dict entry for the column sets per team.
                team_id = os.path.basename(zipfile).split(".zip")[0]
                # Unpack zip contents and add columns to skeleton.
                file_paths = io.unpack_zipfile(zipfile, tempdir)
                file_paths = [
                    file for file in file_paths if "__MACOSX" not in file
                ]  # Annoying add by osx.
                for file in file_paths:
                    log.info("Reading %s.", os.path.basename(file))
                    df = pd.read_csv(file)
                    # Some minor prep before add to skeleton.
                    level = "pgm" if "pg_id" in df.columns else "cm"
                    groupvar = "pg_id" if level == "pgm" else "country_id"
                    df = df.set_index(["month_id", groupvar]).sort_index()
                    task = datautils.determine_task(df)
                    # Prepare a lookup dictionary for all columns.
                    if task != 1:
                        column_sets[task][level][team_id] = {}
                        column_sets[task][level][team_id].update(
                            {
                                step: col
                                for step, col in enumerate(df.columns, 2)
                            }
                        )
                    else:
                        column_sets[task][level][team_id] = df.columns[0]
                    # Run checks.
                    datautils.determine_delta(df[df.columns[0]])
                    datautils.check_duplicates(df)
                    # Add into skeleton.
                    df.to_pickle("test.pkl")
                    add_into_skeleton(df, task, level)

    return column_sets


def fill_missingness(column_sets):
    """Fill any missingness with the benchmark results."""
    # See ./columns.txt to see how the column sets are structured.
    for level in ["cm", "pgm"]:
        df_t1 = cm_t1 if level == "cm" else pgm_t1
        df_t2 = cm_t2 if level == "cm" else pgm_t2
        df_t3 = cm_t3 if level == "cm" else pgm_t3
        for _, cols in column_sets[3][level].items():
            for step, column in cols.items():
                if df_t3[column].isna().any().any():
                    log.info(
                        "%s (t3) contains missing values. "
                        "Filling with benchmark.",
                        column,
                    )
                    df_t3[column] = df_t3[column].fillna(
                        df_t3[column_sets[3][level]["benchmark"][step]]
                    )
                    # df_t2[column] = df_t2[column].fillna(0)
        for _, cols in column_sets[2][level].items():
            for step, column in cols.items():
                if df_t2[column].isna().any().any():
                    log.info(
                        "%s (t2) contains missing values. "
                        "Filling with benchmark.",
                        column,
                    )
                    df_t2[column] = df_t2[column].fillna(
                        df_t2[column_sets[2][level]["benchmark"][step]]
                    )
                    # df_t2[column] = df_t2[column].fillna(0)
        for _, col in column_sets[1][level].items():
            if df_t1[col].isna().any().any():
                log.info(
                    "%s (t1) contains missing values. "
                    "Filling with benchmark.",
                    col,
                )
                df_t1[col] = df_t1[col].fillna(
                    df_t1[column_sets[1][level]["benchmark"]]
                )
                # df_t1[col] = df_t1[col].fillna(0)


def reshape_t1_to_ss():
    """Reshape t1 tables to step-specific columns."""
    global cm_t1_ss
    cm_t1_ss = cm_t1.reset_index().pivot(
        index="country_id",
        columns="month_id",
        values=[col for col in cm_t1],
    )
    cm_t1_ss.columns = [
        f"{team}_s{i}"
        for team in cm_t1_ss.columns.get_level_values(0).unique()
        for i in range(2, 8)
    ]
    global pgm_t1_ss
    pgm_t1_ss = pgm_t1.reset_index().pivot(
        index="pg_id",
        columns="month_id",
        values=[col for col in pgm_t1],
    )
    pgm_t1_ss.columns = [
        f"{team}_s{i}"
        for team in pgm_t1_ss.columns.get_level_values(0).unique()
        for i in range(2, 8)
    ]


def reshape_t2_to_sc():
    """TODO"""
    pass


def save_to_pickled():
    """Save prepared dataframes to scratch."""
    cm_t1.to_pickle(os.path.join(PICKLE_DIR, "cm_t1.pkl"))
    pgm_t1.to_pickle(os.path.join(PICKLE_DIR, "pgm_t1.pkl"))
    cm_t1_ss.to_pickle(os.path.join(PICKLE_DIR, "cm_t1_ss.pkl"))
    pgm_t1_ss.to_pickle(os.path.join(PICKLE_DIR, "pgm_t1_ss.pkl"))
    cm_t2.to_pickle(os.path.join(PICKLE_DIR, "cm_t2.pkl"))
    pgm_t2.to_pickle(os.path.join(PICKLE_DIR, "pgm_t2.pkl"))
    cm_t3.to_pickle(os.path.join(PICKLE_DIR, "cm_t3.pkl"))
    pgm_t3.to_pickle(os.path.join(PICKLE_DIR, "pgm_t3.pkl"))


def collect_submissions():
    """Main collection function."""
    log.info("Adding submission data.")
    column_sets = collect(CLEAN_DIR)
    fill_missingness(column_sets)
    reshape_t1_to_ss()
    save_to_pickled()
    # Also dump the column sets for optional retrieval.
    with open(os.path.join(PICKLE_DIR, "column_sets.pkl"), "wb") as f:
        pickle.dump(column_sets, f)
    return column_sets


def collect_submissions_from_pickles():
    """Collect submissions from pickles."""
    # Read pickles into globals.
    log.info("Collecting prepared data from pickles.")
    globals()["cm_t1"] = pd.read_pickle(os.path.join(PICKLE_DIR, "cm_t1.pkl"))
    globals()["pgm_t1"] = pd.read_pickle(
        os.path.join(PICKLE_DIR, "pgm_t1.pkl")
    )
    globals()["cm_t1_ss"] = pd.read_pickle(
        os.path.join(PICKLE_DIR, "cm_t1_ss.pkl")
    )
    globals()["pgm_t1_ss"] = pd.read_pickle(
        os.path.join(PICKLE_DIR, "pgm_t1_ss.pkl")
    )
    globals()["cm_t2"] = pd.read_pickle(os.path.join(PICKLE_DIR, "cm_t2.pkl"))
    globals()["pgm_t2"] = pd.read_pickle(
        os.path.join(PICKLE_DIR, "pgm_t2.pkl")
    )
    globals()["cm_t3"] = pd.read_pickle(os.path.join(PICKLE_DIR, "cm_t3.pkl"))
    globals()["pgm_t3"] = pd.read_pickle(
        os.path.join(PICKLE_DIR, "pgm_t3.pkl")
    )
