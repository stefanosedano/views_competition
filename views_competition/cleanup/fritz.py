"""Fritz"""

import os
import tempfile
import logging
import pandas as pd
from views_competition import io, ROOT_DIR, SOURCE_DIR, CLEAN_DIR
from . import utilities


log = logging.getLogger(__name__)


def clean():
    log.info(f"Starting {__name__}.")

    month = pd.read_parquet(os.path.join(ROOT_DIR, "data/month.parquet"))
    month = month.reset_index()

    with tempfile.TemporaryDirectory() as tempdir:
        file_paths = io.unpack_zipfile(SOURCE_DIR + "/fritz.zip", tempdir)
        file_paths = [file for file in file_paths if ".csv" in file]
        file_paths = [file for file in file_paths if "__MACOSX" not in file]
        for file in file_paths:
            df = pd.read_csv(file)
            df["month"] = pd.DatetimeIndex(df["date"]).month
            df["year"] = pd.DatetimeIndex(df["date"]).year
            df = df.merge(month, on=["month", "year"])
            df = df.set_index(["month_id", "pg_id"]).sort_index()
            df = df.drop(columns=["date", "month", "year"])
            if "task_1" in file:  # Get correct s by month.
                df = df[["predicted_log_change"]]
                df.columns = ["fritz"]
            else:
                df = df.pivot_table(
                    values="predicted_log_change",
                    index=["month_id", "pg_id"],
                    columns="s",
                )
                df.columns = [f"fritz_s{s}" for s in range(2, 8)]
            utilities.check_missing(df)
            df.to_csv(file)
        io.make_zipfile(
            path_zip=os.path.join(CLEAN_DIR, "fritz.zip"),
            paths_members=file_paths,
        )

    log.info(f"Finished {__name__}.")
