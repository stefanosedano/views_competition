"""Dorazio"""

import os
import tempfile
import logging
import pandas as pd
import numpy as np
from views_competition import io, DATA_DIR, SOURCE_DIR, CLEAN_DIR
from . import utilities


log = logging.getLogger(__name__)


def clean():
    log.info(f"Starting {__name__}.")

    # # Initially delivered as non-delta.
    # pgm = pd.read_parquet(os.path.join(DATA_DIR, "ged_pgm_postpatch.parquet"))
    # pgm["ln_ged_best_sb"] = np.log1p(pgm["ged_best_sb"])
    # cm = pd.read_parquet(os.path.join(DATA_DIR, "ged_cm_postpatch.parquet"))
    # cm["ln_ged_best_sb"] = np.log1p(cm["ged_best_sb"])

    with tempfile.TemporaryDirectory() as tempdir:
        file_paths = io.unpack_zipfile(SOURCE_DIR + "/dorazio.zip", tempdir)
        file_paths = [file for file in file_paths if ".csv" in file]
        file_paths = [file for file in file_paths if "__MACOSX" not in file]
        for file in file_paths:
            df = pd.read_csv(file)
            groupvar = "pg_id" if "pgm" in file else "country_id"
            #actuals = pgm if "pgm" in file else cm
            if "true" in file:  # This is task one.
                df = utilities.reshape_t1_alt(
                    df.set_index(groupvar), groupvar, "dorazio"
                )
                # df = np.exp(df)
                # df = np.log1p(df)
                # df = utilities.delta_columns_by_step_t1(df, actuals, "dorazio")
            else:
                df = df.set_index(["month_id", groupvar]).sort_index()
                df.columns = [f"dorazio_s{s}" for s in range(2, 8)]
                # df = np.exp(df)  # Exponentiate, assuming they did regular np.log.
                # df = np.log1p(df)
                # df = utilities.delta_columns_by_step(df, actuals)
            utilities.check_missing(df)
            df.to_csv(file)
        io.make_zipfile(
            path_zip=os.path.join(CLEAN_DIR, "dorazio.zip"),
            paths_members=file_paths,
        )

    log.info(f"Finished {__name__}.")

if __name__ == "__main__":
    clean()