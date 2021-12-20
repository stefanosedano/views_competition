import os
import sys
import logging
import datetime

# TODO: find solution to root path not depending on virtualenv.
# Perhaps if VIRTUAL_ENV not in environ, just get the absolute path.
# Test that with conda.
ROOT_DIR = os.path.split(os.environ["VIRTUAL_ENV"])[0]
LOG_DIR = os.path.join(ROOT_DIR, "logs")
DATA_DIR = os.path.join(ROOT_DIR, "data")
PICKLE_DIR = os.path.join(ROOT_DIR, "data", "pickled")
SOURCE_DIR = os.path.join(ROOT_DIR, "data", "raw")
CLEAN_DIR = os.path.join(ROOT_DIR, "data", "clean")
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")

utc_now = datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")
LOG = os.path.join(LOG_DIR, f"{utc_now}_views_competition.log")
LOG_FMT = (
    "[%(asctime)s] :: %(levelname)s :: %(module)s :: %(lineno)d :: %(message)s"
)
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FMT,
    handlers=[
        logging.FileHandler(LOG),
        logging.StreamHandler(),
    ],
)

TIMEFRAMES = {
    1: list(range(490, 496)),  # Oct 2020-March 2021
    2: list(range(469, 475)),  # Jan 2019-June 2019
}


def setup_dirs():
    """Main directories in root."""
    dirs = [
        LOG_DIR,
        SOURCE_DIR,
        CLEAN_DIR,
        PICKLE_DIR,
        OUTPUT_DIR,
    ]
    for path_dir in dirs:
        if not os.path.isdir(path_dir):
            os.mkdir(path_dir)


def output_dirs(output_path):
    """Creates the necessary output dirs per out_path."""
    out_paths = {
        "maps": ["error", "observed", "predicted", "ensemble", "pemdiv"],
        "graphs": [
            "error",
            "line",
            "ablation",
            "correlation",
            "coordinates",
            "radar",
            "scatter",
            "bootstrap",
        ],
        "tables": [],  # No subs.
        "data": [],  # No subs.
    }
    for dirname, subdirnames in out_paths.items():
        path = os.path.join(output_path, dirname)
        if not os.path.isdir(path):
            os.makedirs(path)
        for subdir in subdirnames:
            subpath = os.path.join(path, subdir)
            if not os.path.isdir(subpath):
                os.makedirs(subpath)


setup_dirs()
output_dirs(OUTPUT_DIR)
