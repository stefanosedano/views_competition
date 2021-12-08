"""Oswald"""

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
        file_paths = io.unpack_zipfile(SOURCE_DIR + "/oswald.zip", tempdir)
        file_paths = [file for file in file_paths if ".csv" in file]
        # Clean up tables.
        for file in file_paths:
            df = pd.read_csv(file)
            df = df.set_index(["month_id", "country_id"]).sort_index()
            if "task1" in file:  # This is task one.
                df = utilities.reshape_t1(df, "country_id", "oswald")
            else:
                df = df[
                    [
                        col
                        for col in df.columns
                        if "calibrated" not in col and "sc" not in col
                    ]
                ]  # Our calibration makes it worse.
                df.columns = [f"oswald_s{s}" for s in range(2, 8)]
            utilities.check_missing(df)
            df.to_csv(file)
        io.make_zipfile(
            path_zip=os.path.join(CLEAN_DIR, "oswald.zip"),
            paths_members=file_paths,
        )

    log.info(f"Finished {__name__}.")
