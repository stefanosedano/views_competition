"""Vestby"""

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
        file_paths = io.unpack_zipfile(SOURCE_DIR + "/vestby.zip", tempdir)
        sub_zip = [file for file in file_paths if ".zip" in file][0]
        file_paths = io.unpack_zipfile(sub_zip, tempdir)
        out_paths = []
        # Clean up tables.
        for file in file_paths:
            df = pd.read_csv(file)
            df = df.pivot_table(
                values="estimate",
                index=["month_id", "pg_id"],
                columns="stat_model",
            )
            df.columns = [col.replace(" ", "_").lower() for col in df.columns]
            utilities.check_missing(df)
            for model in df.columns:
                sub_df = pd.DataFrame(
                    {f"s{i}": df[model] for i in range(2, 8)}
                )
                modelname = (
                    "no_change" if "null" in model else model
                )  # Small name adjustment here...
                # If it's the null, drop the "vestby" prefix.
                prefix = "vestby_" if modelname != "no_change" else ""
                sub_df.columns = [f"{prefix}{modelname}_{c}" for c in sub_df]
                out_paths.append(file.replace("out", f"out_{prefix}{modelname}"))
                if 490 in df.index[0]:
                    sub_df = utilities.reshape_t1(
                        df=sub_df,
                        groupvar="pg_id",
                        team=f"{prefix}{modelname}",
                    )
                sub_df.to_csv(
                    file.replace("out", f"out_{prefix}{modelname}")
                )  # Overwrite tempfiles with fix.
        # Zip to /clean.
        models = ["no_change", "vestby_rf_fit", "vestby_xgb_fit"]
        for model in models:
            suffix = "_pgm" if model == "no_change" else ""
            io.make_zipfile(
                path_zip=os.path.join(CLEAN_DIR, f"{model}{suffix}.zip"),
                paths_members=[path for path in out_paths if model in path],
            )
