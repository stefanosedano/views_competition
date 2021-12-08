"""Lindholm"""

import os
import tempfile
import logging
import pandas as pd
from functools import reduce
from views_competition import io, SOURCE_DIR, CLEAN_DIR
from . import utilities


log = logging.getLogger(__name__)


def clean():
    log.info(f"Starting {__name__}.")

    with tempfile.TemporaryDirectory() as tempdir:
        file_paths = io.unpack_zipfile(SOURCE_DIR + "/lindholm.zip", tempdir)
        file_paths = [file for file in file_paths if "__MACOSX" not in file]
        file_paths = [file for file in file_paths if ".csv" in file]
        # Clean up tables.
        for file in file_paths:
            df = pd.read_csv(file).set_index("pg_id")
            df.columns = [col.split(" ")[0] for col in df.columns]
            df = pd.DataFrame(df.stack()).reset_index()
            df.columns = [
                "pg_id",
                "month_id",
                f"{os.path.basename(file)}".lower().replace(".csv", ""),
            ]
            df = df.set_index(["month_id", "pg_id"])
            utilities.check_missing(df)
            if "task1" in file:
                df.columns = ["lindholm"]
                out = os.path.join(tempdir, f"Lindholm_task1.csv")
                df.to_csv(out)
            else:
                df.to_csv(file)
        # Collect all individual csvs for task 2 and 3...
        for task in ["task2", "task3"]:
            task_list = [file for file in file_paths if task in file]
            list_of_dataframes = [
                pd.read_csv(s).set_index(["month_id", "pg_id"])
                for s in task_list
            ]
            df = reduce(
                lambda x, y: pd.merge(x, y, left_index=True, right_index=True),
                list_of_dataframes,
            )
            df = df[sorted(df.columns)]
            df.columns = [
                f"lindholm_s{s}" for s in range(2, 8)
            ]  # Rename columns.
            out = os.path.join(tempdir, f"Lindholm_{task}.csv")
            df.to_csv(out)
        # Only zip the three csvs to /clean.
        tasks = ["task1", "task2", "task3"]
        file_paths = [
            os.path.join(tempdir, f"Lindholm_{task}.csv") for task in tasks
        ]
        io.make_zipfile(
            path_zip=os.path.join(CLEAN_DIR, "lindholm.zip"),
            paths_members=file_paths,
        )

    log.info(f"Finished {__name__}.")
