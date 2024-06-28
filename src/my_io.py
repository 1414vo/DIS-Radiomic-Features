"""Module for custom input/output operations.

Contains utilities for loading the raw data and writing to non-existing folders.
"""
import os
from pathlib import Path
import pandas as pd


def load_raw(path: str) -> pd.DataFrame:
    """
    Loads a concatenated DataFrame from the raw datafiles.

    Parameters
    ----------
    path: str
        The location of the raw data

    Returns
    -------
    pd.DataFrame
        A concatenated DataFrame containing all factors and all the raw data.
    """
    df_list = []
    for f_name in os.listdir(path):
        df = pd.read_csv(f"{path}/{f_name}")
        df["Reconstruction"] = "MB" if f_name.split(".")[-2][-2:] == "MB" else "BP"
        df_list.append(df)
    return pd.concat(df_list).reset_index(drop=True)


def create_folder(path: str) -> None:
    """
    Creates a folder if it does not exist.

    Parameters
    ----------
    path: str
        A file path within the desired folder.
    """
    outfile = Path(path)
    outfile.parent.mkdir(exist_ok=True, parents=True)
