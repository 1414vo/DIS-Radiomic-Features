from src.anova import manual_anova
from src.utils import extract_factor_summary
from src.io import load_raw, create_folder
from src.plotting import plot_interaction_summary
import numpy as np
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="SensitivityAnalysis",
        description="Calculates the sensitivity of different factors on the raw data",
    )
    parser.add_argument("data_path", help="Location of the raw data")
    parser.add_argument(
        "-o",
        "--output",
        default="./out/sens_analysis",
        help="Where to store the analysis plots.",
    )
    args = parser.parse_args()

    data_path = args["data_path"]
    out_path = args["output"]
    try:
        data = load_raw(data_path)
    except FileNotFoundError:
        print(f"Could not find the data in folder {data_path}.")
        exit(1)

    # Remove arbitrary luminal-B patient
    np.random.seed(0)
    luminal_ids = data[data["Model"] == "luminal"]["PatientName"].unique()
    to_remove = np.random.choice(luminal_ids)
    data_filtered = data[data["PatientName"] != to_remove]
    factor_names = ["Model", "GLbins", "Wavelength", "Reconstruction"]
    factor_data = data_filtered[factor_names]
    # Remove shape features
    features = data_filtered.iloc[:, 27:-1]

    # Full analysis
    interactions = features.apply(
        lambda col: manual_anova(col, factor_data.values.T, factor_names), axis=0
    )
    interaction_summary = interactions.apply(
        lambda x: extract_factor_summary(x, factor_names)
    )
    path = f"{out_path}/full_analysis.png"
    create_folder(path)
    plot_interaction_summary(interaction_summary, output_path=path)

    # Fixing the gray levels
    factor_names = ["Model", "Wavelength", "Reconstruction"]
    for bins in factor_data["GLbins"].unique():
        mask = factor_data["GLbins"] == bins

        interactions = features[mask].apply(
            lambda col: manual_anova(
                col, factor_data[mask][factor_names].values.T, factor_names
            ),
            axis=0,
        )
        interaction_summary = interactions.apply(
            lambda x: extract_factor_summary(x, factor_names)
        )
        path = f"{out_path}/fixed_glbins/analysis_{bins}.png"
        create_folder(path)
        plot_interaction_summary(interaction_summary, output_path=path)

    # Fixing the reconstruction
    factor_names = [
        "Model",
        "GLbins",
        "Wavelength",
    ]
    for recon in factor_data["Reconstruction"].unique():
        mask = factor_data["Reconstruction"] == recon

        interactions = features[mask].apply(
            lambda col: manual_anova(
                col, factor_data[mask][factor_names].values.T, factor_names
            ),
            axis=0,
        )
        interaction_summary = interactions.apply(
            lambda x: extract_factor_summary(x, factor_names)
        )
        path = f"{out_path}/fixed_reconstruction/analysis_{recon}.png"
        create_folder(path)
        plot_interaction_summary(interaction_summary, output_path=path)
