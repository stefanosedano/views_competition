"""Hultman"""

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
        file_paths = io.unpack_zipfile(SOURCE_DIR + "/hultman.zip", tempdir)
        file_paths = [file for file in file_paths if ".csv" in file]
        file_paths = [file for file in file_paths if "geopko" not in file]
        # Clean up tables.
        for file in file_paths:
            df = pd.read_csv(file).set_index(["month_id", "pg_id"])
            if "hultman_geopko_true" not in df.columns:
                df = df.drop(columns=["hultman_geopko_s1"])
            else:
                df.columns = ["hultman"]
            df.columns = [col.replace("geopko_", "") for col in df.columns]
            utilities.check_missing(df)
            df.to_csv(file)  # Overwrite tempfiles with fix.

        io.make_zipfile(
            path_zip=os.path.join(CLEAN_DIR, "hultman.zip"),
            paths_members=file_paths,
        )

    log.info(f"Finished {__name__}.")
