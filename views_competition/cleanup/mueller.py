"""Mueller"""

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
        file_paths = io.unpack_zipfile(SOURCE_DIR + "/mueller.zip", tempdir)
        file_paths = [file for file in file_paths if ".csv" in file]
        file_paths = [file for file in file_paths if "__MACOSX" not in file]
        out_paths = []
        # Clean up tables.
        for file in file_paths:
            df = pd.read_csv(file)
            groupvar = "country_id" if "country_id" in df.columns else "pg_id"
            df = df.set_index(["month_id", groupvar]).sort_index()
            if "predictions_2020" in file:
                df = utilities.reshape_t1(df, "country_id", "mueller")
                df.to_csv(file)
                out_paths.append(file)
                utilities.check_missing(df)
            else:
                df.columns = [f"mueller_{c}" for c in df.columns]
                df.columns = [
                    col.replace("yourmodel_", "") for col in df.columns
                ]
                task2 = df.loc[445:480]
                task3 = df.loc[409:444]
                utilities.check_missing(task2)
                utilities.check_missing(task3)
                task2.to_csv(
                    file.replace("earlier", "task2")
                )  # Overwrite tempfiles with fix.
                task3.to_csv(file.replace("earlier", "task3"))
                out_paths.append(file.replace("earlier", "task2"))
                out_paths.append(file.replace("earlier", "task3"))
        io.make_zipfile(
            path_zip=os.path.join(CLEAN_DIR, "mueller.zip"),
            paths_members=out_paths,
        )

    log.info(f"Finished {__name__}.")
