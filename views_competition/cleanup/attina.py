"""Attina"""

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

    cm = pd.read_parquet(os.path.join(DATA_DIR, "ged_cm_prepatch.parquet"))
    cm["ln_ged_best_sb"] = np.log1p(cm["ged_best_sb"])

    with tempfile.TemporaryDirectory() as tempdir:
        file_paths = io.unpack_zipfile(SOURCE_DIR + "/attina.zip", tempdir)
        file_paths = [
            file for file in file_paths if ".csv" in file or ".xlsx" in file
        ]
        file_paths = [file for file in file_paths if "__MACOSX" not in file]
        for file in file_paths:
            if ".csv" in file:
                df = pd.read_csv(file, sep=";")
            else:
                df = pd.read_excel(file)
            df.columns = [col.lower() for col in df.columns]
            df = df.set_index(["month_id", "country_id"]).sort_index()
            df = df[[col for col in df.columns if "_pred" in col]]
            # print(df.dtypes)
            if "set1" in file:
                df = utilities.delta_columns_by_step_t1(
                    df=df, actuals=cm, team="attina"
                )
                df = df[["attina_fixed"]]
                df.columns = ["attina"]
            if "set3" in file:
                for col in df.columns:
                    df[col] = df[col].str.replace(",", ".")
                    df[col] = df[col].astype(float)
            if not "set1" in file:
                df = utilities.delta_columns_by_step(df=df, actuals=cm)
                df = df[[col for col in df.columns if "aci" in col]]
                df.columns = [f"attina_s{s}" for s in range(2, 8)]
            utilities.check_missing(df)
            df.to_csv(file.replace(".xlsx", ".csv"))
        file_paths = [file.replace(".xlsx", ".csv") for file in file_paths]
        io.make_zipfile(
            path_zip=os.path.join(CLEAN_DIR, "attina.zip"),
            paths_members=file_paths,
        )

    log.info(f"Finished {__name__}.")
