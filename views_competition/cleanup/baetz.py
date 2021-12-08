"""Baetz"""

import os
import tempfile
import logging
import pandas as pd
from views_competition import io, SOURCE_DIR, CLEAN_DIR
from . import utilities


log = logging.getLogger(__name__)


def clean():
    log.info(f"Starting {__name__}.")

    with tempfile.TemporaryDirectory() as tempdir:
        file_paths = io.unpack_zipfile(SOURCE_DIR + "/baetz.zip", tempdir)
        file_paths = [file for file in file_paths if ".csv" in file]
        file_paths = [file for file in file_paths if "__MACOSX" not in file]
        out_paths = []
        # Clean up tables.
        for file in file_paths:
            # Task one looks broken!
            if "task1" in file:
                df = pd.read_csv(file)
                df = df[df.columns[1:]]  # Drop unnamed col.
                df = df.set_index(["month_id", "country_id"]).sort_index()
                df = df[[col for col in df.columns if "calibrated" not in col]]
                df = df[df.columns[1:]]  # Drop redundant step 1.
                # df = utilities.delta_columns_by_step_t1(df, cm, "baetz")
                df = utilities.reshape_t1(df, "country_id", "baetz")
                out_paths.append(file)
                utilities.check_missing(df)
                df.to_csv(file)  # Overwrite tempfiles with fix.
            if "model2_delta" in file:
                df = pd.read_csv(file)
                df = df[df.columns[1:]]  # Drop unnamed col.
                df = df.set_index(["month_id", "country_id"]).sort_index()
                df = df[df.columns[1:]]  # Drop redundant step 1.
                if "task3" in file:
                    df = df.loc[409:444]
                df = df[
                    [
                        col
                        for col in df.columns
                        if "calibrated" not in col and "sc" not in col
                    ]
                ]
                df.columns = [
                    f"baetz_s{s}" for s in range(2, 8)
                ]  # Rename columns.
                out_paths.append(file)
                utilities.check_missing(df)
                df.to_csv(file)  # Overwrite tempfiles with fix.
        io.make_zipfile(
            path_zip=os.path.join(CLEAN_DIR, "baetz.zip"),
            paths_members=out_paths,
        )

    log.info(f"Finished {__name__}.")
