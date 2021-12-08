"""Malone"""

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
        file_paths = io.unpack_zipfile(SOURCE_DIR + "/malone.zip", tempdir)
        file_paths = [file for file in file_paths if ".csv" in file]
        file_paths = [file for file in file_paths if "__MACOSX" not in file]
        # Clean up tables.
        for file in file_paths:
            df = pd.read_csv(file)
            groupvar = "country_id" if "country_id" in df.columns else "pg_id"
            df = df.rename(columns={"last_month_data_id": "month_id"})
            df = df.set_index(["month_id", groupvar]).sort_index()
            if "Country" in df.columns:
                df = df.drop(
                    columns=["Country"]
                )  # Task 2 and 3 contain "Country" and s1.
            if "task1" in file:
                df = df.pivot_table(
                    values="value",
                    index=["month_id", "country_id"],
                    columns="variable",
                )
                df = df[df.columns[1:]]  # Drop redundant s1.
                # subset s2 and put on 490, s3 on 491 and so on.
                df = utilities.reshape_t1_alt(df, "country_id", "malone")
            df.columns = [
                col.lower().replace("lstm_", "") for col in df.columns
            ]
            if "malone_s1" in df.columns:
                df = df.drop(columns=["malone_s1"])  # Drop redundant s1.
            utilities.check_missing(df)
            df.to_csv(file)  # Overwrite tempfiles with fix.
        io.make_zipfile(
            path_zip=os.path.join(CLEAN_DIR, "malone.zip"),
            paths_members=file_paths,
        )

    log.info(f"Finished {__name__}.")
