"""Randahl"""

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
        file_paths = io.unpack_zipfile(SOURCE_DIR + "/randahl.zip", tempdir)
        file_paths = [file for file in file_paths if ".csv" in file]
        file_paths = [file for file in file_paths if "__MACOSX" not in file]
        out_paths = []
        models = ["_vmm_", "_hmm_", "_hhmm_"]
        for file in file_paths:
            df = pd.read_csv(file)
            df = df.set_index(["month_id", "country_id"]).sort_index()
            df = df[
                [col for col in df.columns if not "s1" in col]
            ]  # Drop redundant s1.
            utilities.check_missing(df)
            for model in models:
                sub_df = df[[col for col in df if model in col]]
                modelname = model.replace("_", "")
                out_paths.append(
                    file.replace("predictions", f"predictions_{modelname}")
                )
                sub_df.columns = [
                    col.replace("drjv", "randahl") for col in sub_df.columns
                ]
                sub_df.to_csv(
                    file.replace("predictions", f"predictions_{modelname}")
                )  # Overwrite tempfiles with fix.
        # Zip to /clean.
        for model in models:
            modelname = model.replace("_", "")
            io.make_zipfile(
                path_zip=os.path.join(CLEAN_DIR, f"randahl_{modelname}_weighted.zip"),
                paths_members=[path for path in out_paths if model in path],
            )

    log.info(f"Finished {__name__}.")
