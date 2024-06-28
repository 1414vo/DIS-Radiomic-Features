""" A script for performing k-fold sensitivity analysis using balanced ANOVA.

Usage:

.. code:: bash

    $ python -m scripts.k_fold_sensitivity_analysis <data_path> -o <output_path>

- *data_path*: The location of the data.
- *output_path*: Where to store the relevant outputs.
"""
from src.anova import manual_anova
from src.utils import extract_factor_summary
from src.my_io import load_raw, create_folder
import numpy as np
import pandas as pd
import argparse
import warnings

warnings.filterwarnings("ignore")


def sensitivity_summary(col_means: pd.Series, col_stds: pd.Series) -> None:
    """
    Summarizes the sensitivities of each factor.

    Parameters
    ----------
    col_means: pd.Series
        The mean sensitivity for the given feature.
    col_stds: pd.Series
        The standard deviation of the sensitivity for the given feature.
    """
    for factor in col_means.index:
        print(
            f"{factor}: {col_means[factor]:.3f}+-{col_stds[factor]:.3f}, CoV:{col_stds[factor]/col_means[factor]:.3f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="SensitivityAnalysis",
        description="Calculates the sensitivity of different factors on the raw data",
    )
    parser.add_argument("data_path", help="Location of the raw data")
    parser.add_argument(
        "-o",
        "--output",
        default="./out/k_fold_analysis",
        help="Where to store the summary tables.",
    )
    args = parser.parse_args()
    data_path = args.data_path
    out_path = args.output

    # Load raw data
    try:
        data = load_raw(data_path)
    except FileNotFoundError:
        print(f"Could not find the data in folder {data_path}.")
        exit(1)

    # Generate seeds for random selection
    np.random.seed(0)
    seeds = np.random.randint(0, int(2**32 - 1), size=5)

    factor_names = ["Model", "GLbins", "Wavelength", "Reconstruction"]
    luminal_ids = data[data["Model"] == "luminal"]["PatientName"].unique()
    basal_ids = data[data["Model"] == "basal"]["PatientName"].unique()
    sensitivities = []

    for i in range(5):
        np.random.seed(seeds[i])

        # Select which patients to use
        selected_lum = set(np.random.choice(luminal_ids, size=8, replace=False))
        selected_bas = set(np.random.choice(basal_ids, size=8, replace=False))

        to_include = selected_lum.union(selected_bas)

        data_filtered = data[data["PatientName"].isin(to_include)]
        factor_data = data_filtered[factor_names]
        # Remove metadata
        features = data_filtered.iloc[:, 41:-1]

        # Full analysis
        interactions = features.apply(
            lambda col: manual_anova(col, factor_data.values.T, factor_names), axis=0
        )
        sensitivities.append(
            interactions.apply(lambda x: extract_factor_summary(x, factor_names))
        )

    # Concatenate results
    sensitivity_df = pd.concat(sensitivities, axis=1)

    # Compute the mean sensitivity value for each feature-factor pair
    means = sensitivity_df.groupby(axis=1, level=0).mean()
    create_folder(f"{out_path}/means.csv")
    means.to_csv(f"{out_path}/means.csv")

    # Compute the standard deviation of the sensitivity value for each feature-factor pair
    stds = sensitivity_df.groupby(axis=1, level=0).std()
    stds.to_csv(f"{out_path}/std.csv")

    print("Skewness summary:")
    print("-------------------")
    sensitivity_summary(
        means["original_firstorder_Skewness"], stds["original_firstorder_Skewness"]
    )

    print("\nKurtosis summary:")
    print("-------------------")
    sensitivity_summary(
        means["original_firstorder_Kurtosis"], stds["original_firstorder_Kurtosis"]
    )
