""" A script for applying the VOI feature normalization.

Usage:

.. code:: bash

    $ python -m scripts.normalize_features <data_path> -o <output_path>

- *data_path*: The location of the data.
- *output_path*: Where to store the relevant outputs.

"""
from src.my_io import load_raw
from src.utils import NoVoxelsScaler
import numpy as np
import pandas as pd
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Feature Normalization",
        usage="python -m scripts.normalize_features <data_path> --output <out_path>",
        description="Normalizes the features by the number of voxels",
    )
    parser.add_argument("data_path", help="Location of the raw data")
    parser.add_argument(
        "-o",
        "--output",
        default="./data",
        help="Where to store the normalized features.",
    )
    args = parser.parse_args()
    data_path = args.data_path
    out_path = args.output
    try:
        data = load_raw(data_path)
    except FileNotFoundError:
        print(f"Could not find the data in folder {data_path}.")
        exit(1)

    # Generate seeds for random selection
    np.random.seed(0)
    seeds = np.random.randint(0, int(2**32 - 1), size=5)
    no_voxels = data["diagnostics_Mask-original_VoxelNum"]
    patient_names = data["PatientName"]

    # Cut off all metadata + shape features
    data = data.iloc[:, 41:-1]

    print(data.columns)
    scaler = NoVoxelsScaler(
        transform_options=[
            "lin",
            "sq",
            "cub",
            "log",
            "inv",
            "inv_sq",
            "inv_cub",
            "inv_log",
        ]
    )

    rescaled_data = scaler.fit_transform(data.values, no_voxels.values)
    for i, col in enumerate(data.columns):
        print(f"{col}: {scaler.fits[i].model_transform}")
    rescaled_data = pd.DataFrame(rescaled_data, index=data.index, columns=data.columns)
    rescaled_data["PatientName"] = patient_names
    rescaled_data.to_csv(f"{out_path}/normalized_features.csv")
