"""I/O functions."""

from typing import List
import os
import zipfile
import logging

logger = logging.getLogger(__name__)


def unpack_zipfile(path_zip: str, destination: str) -> List[str]:
    """ Unpack a zipfile """
    with zipfile.ZipFile(path_zip, "r") as f:
        msg = f"Extracting {f.namelist()} from {path_zip} to {destination}"
        logger.info(msg)
        f.extractall(path=destination)
        names: List[str] = f.namelist()
    destination_paths: List[str] = [
        os.path.join(destination, name) for name in names
    ]
    return destination_paths


def make_zipfile(path_zip: str, paths_members: List[str]) -> None:
    """ Compress files at paths_members into path_zip """
    logger.info(f"Compressing files: {paths_members} to {path_zip}.")
    with zipfile.ZipFile(
        path_zip, "w", compression=zipfile.ZIP_DEFLATED
    ) as zf:
        for path_member in paths_members:
            zf.write(
                filename=path_member, arcname=os.path.basename(path_member)
            )
