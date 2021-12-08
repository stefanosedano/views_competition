"""No-change

Uses cleaned-up chadefaux to just fill with zeroes.
"""

import os
import tempfile
import logging
import pandas as pd
from views_competition import io, CLEAN_DIR
from . import utilities


log = logging.getLogger(__name__)


def clean():
    log.info(f"Starting {__name__}.")

    with tempfile.TemporaryDirectory() as tempdir:
        file_paths = io.unpack_zipfile(CLEAN_DIR + "/chadefaux.zip", tempdir)
        file_paths = [
            file for file in file_paths if ".csv" in file and "cm" in file
        ]
        for file in file_paths:
            df = pd.read_csv(file)
            df = df.set_index(["month_id", "country_id"]).sort_index()
            file = file.replace("chadefaux", "no_change")
            if "set1" not in file:
                null_cols = []
                for col in df.columns:
                    step = col.split("_")[1]
                    df[col].values[:] = 0
                    null_cols.append(f"no_change_{step}")
                df.columns = null_cols
            else:
                df[df.columns[0]].values[:] = 0
                df.columns = ["no_change"]
            utilities.check_missing(df)
            df.to_csv(file)
        io.make_zipfile(
            path_zip=os.path.join(CLEAN_DIR, "no_change_cm.zip"),
            paths_members=[
                file.replace("chadefaux", "no_change") for file in file_paths
            ],
        )

    log.info(f"Finished {__name__}.")
