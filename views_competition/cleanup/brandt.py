"""Brandt"""

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
        file_paths = io.unpack_zipfile(SOURCE_DIR + "/brandt.zip", tempdir)
        file_paths = [
            file for file in file_paths if ".csv" in file or ".xlsx" in file
        ]
        for file in file_paths:
            df = pd.read_csv(file)
            if "task1" in file:
                df = utilities.reshape_t1_alt(
                    df.set_index("pg_id"), "pg_id", "brandt"
                )
            else:
                df = df.set_index(["month_id", "pg_id"]).sort_index()
                df.columns = [f"brandt_s{s}" for s in range(2, 8)]
            utilities.check_missing(df)
            df.to_csv(file)
        io.make_zipfile(
            path_zip=os.path.join(CLEAN_DIR, "brandt.zip"),
            paths_members=file_paths,
        )

    log.info(f"Finished {__name__}.")
