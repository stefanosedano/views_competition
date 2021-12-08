"""Chadefaux"""

import os
import logging
import tempfile
import pandas as pd
from views_competition import io, SOURCE_DIR, CLEAN_DIR
from . import utilities

log = logging.getLogger(__name__)


def clean():
    log.info(f"Starting {__name__}.")

    with tempfile.TemporaryDirectory() as tempdir:
        file_paths = io.unpack_zipfile(SOURCE_DIR + "/chadefaux.zip", tempdir)
        file_paths = [file for file in file_paths if "__MACOSX" not in file]
        file_paths = [file for file in file_paths if ".csv" in file]
        # Rename so we can exchange set3 for set1, and vice versa.
        for i, file in enumerate(file_paths):
            file_paths[i] = file.replace("set", "task")
            os.rename(file, file.replace("set", "task"))
        # Clean up tables.
        for file in file_paths:
            groupvar = "country_id" if "cm" in file else "pg_id"
            df = pd.read_csv(file)
            df = df.rename(columns={"unit_id": groupvar})
            df = df[df.columns[1:]]  # Drop Unnamed col.
            df = df.set_index(["month_id", groupvar]).sort_index()
            df = df.groupby(level=[0, 1]).mean()  # Take average of duplicates.
            utilities.check_missing(df)
            # Rename set3 to set1.
            if "task1" in file:
                df.to_csv(file.replace("task1", "set3"))
            elif "task3" in file:
                df = utilities.reshape_t1(
                    df=df, groupvar=groupvar, team="chadefaux"
                )
                df.to_csv(file.replace("task3", "set1"))
            else:
                df.to_csv(file.replace("task", "set"))
        # Zip to /clean.
        file_paths = [file.replace("task", "set") for file in file_paths]
        io.make_zipfile(
            path_zip=os.path.join(CLEAN_DIR, "chadefaux.zip"),
            paths_members=file_paths,
        )

    log.info(f"Finished {__name__}.")
