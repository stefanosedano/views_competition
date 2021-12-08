"""Benchmark"""

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
        file_paths = io.unpack_zipfile(SOURCE_DIR + "/benchmark.zip", tempdir)
        file_paths = [file for file in file_paths if "__MACOSX" not in file]
        for file in file_paths:
            df = pd.read_csv(file)
            groupvar = "pg_id" if "pgm" in file else "country_id"
            df = df.set_index(["month_id", groupvar]).sort_index()
            if "task1" not in file:
                if groupvar == "pg_id":
                    df = df[
                        [
                            col
                            for col in df
                            if "calibrated" in col and "sc" not in col
                        ]
                    ]
                else:
                    df = df[
                        [
                            col
                            for col in df
                            if "calibrated" not in col and "sc" not in col
                        ]
                    ]  # Uncalibrated for cm!
                df.columns = [f"benchmark_s{s}" for s in range(2, 8)]
                utilities.check_missing(df)
                df.to_csv(file)
            else:
                if groupvar == "pg_id":
                    df = df[
                        [
                            col
                            for col in df
                            if "calibrated" in col and "sc" not in col
                        ]
                    ]
                else:
                    df = df[
                        [
                            col
                            for col in df
                            if "calibrated" not in col and "sc" not in col
                        ]
                    ]  # Uncalibrated for cm!
                df.columns = [f"benchmark_s{s}" for s in range(2, 8)]
                utilities.check_missing(df)
                df = utilities.reshape_t1(df, groupvar, "benchmark")
                df.to_csv(file)
        io.make_zipfile(
            path_zip=os.path.join(CLEAN_DIR, "benchmark.zip"),
            paths_members=file_paths,
        )

    log.info(f"Finished {__name__}.")
