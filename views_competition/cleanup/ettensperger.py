"""Ettensperger"""

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
        file_paths = io.unpack_zipfile(
            SOURCE_DIR + "/ettensperger.zip", tempdir
        )
        file_paths = [file for file in file_paths if ".csv" in file]
        file_paths = [file for file in file_paths if "__MACOSX" not in file]
        file_paths = [
            file
            for file in file_paths
            if "Ensemble1.csv" in file or "Ensemble.csv" in file
        ]
        # Clean up tables.
        for file in file_paths:
            df = pd.read_csv(file)
            groupvar = "country_id" if "country_id" in df.columns else "pg_id"
            df = df.set_index(["month_id", groupvar])
            df = df[[col for col in df.columns if "in_africa" not in col]]
            df = df.drop(columns=["country_name"])
            df.columns = [col.replace(".", "_").lower() for col in df.columns]
            df.columns = [f"ettensperger_{c}" for c in df]
            if "ettensperger_s_1" in df.columns:
                df = df[df.columns[1:]]  # Drop redundant s1
                df.columns = [col.replace("s_", "s") for col in df.columns]
            else:
                df.columns = ["ettensperger"]
            utilities.check_missing(df)
            df.to_csv(file)  # Overwrite tempfiles with fix.
        io.make_zipfile(
            path_zip=os.path.join(CLEAN_DIR, "ettensperger.zip"),
            paths_members=file_paths,
        )

    log.info(f"Finished {__name__}.")
