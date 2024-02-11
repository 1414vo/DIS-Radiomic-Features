import os
from pathlib import Path
import pandas as pd


def load_raw(path):
    df_list = []
    for f_name in os.listdir(path):
        df = pd.read_csv(f"{path}/{f_name}")
        df["Reconstruction"] = "MB" if f_name.split(".")[-2][-2:] == "MB" else "BP"
        df_list.append(df)
    return pd.concat(df_list).reset_index(drop=True)


def create_folder(path):
    outfile = Path(path)
    outfile.parent.mkdir(exist_ok=True, parents=True)
