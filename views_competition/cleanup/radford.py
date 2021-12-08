"""Radford"""

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
        file_paths = io.unpack_zipfile(SOURCE_DIR + "/radford.zip", tempdir)
        file_paths = [file for file in file_paths if "__MACOSX" not in file]
        file_paths = [file for file in file_paths if ".csv" in file]
        file_paths = [
            file for file in file_paths if "set1.csv" not in file
        ]  # use updated version.
        # Clean up tables.
        for file in file_paths:
            df = pd.read_csv(file)
            df.columns = [col.lower() for col in df.columns]
            df = df.set_index(["month_id", "pg_id"])
            cols = [
                col for col in df.columns if "_s" in col
            ]  # Drop unneeded cols.
            df = df[cols]
            df.columns = [
                col.replace("clstm_", "") for col in df.columns
            ]  # Drop clstm from names.
            if "set1" in file:
                df = utilities.reshape_t1(df, "pg_id", "radford")
            utilities.check_missing(df)
            # Add pgm to filename for clarity.
            df.to_csv(
                file.replace("radford_set", "radford_pgm_set")
            )  # Overwrite tempfiles with fix.
        # Zip to /clean.
        file_paths = [
            file.replace("radford_set", "radford_pgm_set")
            for file in file_paths
        ]
        io.make_zipfile(
            path_zip=os.path.join(CLEAN_DIR, "radford.zip"),
            paths_members=file_paths,
        )

    log.info(f"Finished {__name__}.")
