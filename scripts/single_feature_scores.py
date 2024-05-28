from src.classification import single_feature_classification
from src.config_parser import parse_config
import numpy as np
import pandas as pd
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Feature Scoring",
        usage="python -m scripts.single_features_scores <data_path> --output <out_path> --cfg <config_path>",
        description="Reduces the features by observing the distributions and correlations",
    )
    parser.add_argument("data_path", help="Location of the preprocessed data")
    parser.add_argument(
        "-o",
        "--output",
        default="./out",
        help="Where to store the summary tables.",
    )
    parser.add_argument(
        "--cfg",
        default="./config.json",
        help="Where the classifier configurations can be found.",
    )
    parser.add_argument("--validation", action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    data_path = args.data_path
    out_path = args.output
    try:
        data = pd.read_csv(f"{data_path}/All_features_corrected_final_patient.csv")
        models = pd.read_csv(f"{data_path}/All_model.csv")
    except FileNotFoundError:
        print(f"Could not find the data in {data_path}.")
        exit(1)

    try:
        config = parse_config(args.cfg)
    except FileNotFoundError:
        print(f"Could not find configuration file {args.cfg}.")
        exit(1)
    if args.validation:
        # Generate seeds for random selection
        np.random.seed(0)
        patient_to_drop = np.random.choice(
            data[models["Model"] == "luminal"]["PatientName"].unique()
        )
        balanced_data = data[data["PatientName"] != patient_to_drop]
        balanced_models = models[data["PatientName"] != patient_to_drop]

        test_patients_basal = np.random.choice(
            balanced_data[balanced_models["Model"] == "basal"]["PatientName"].unique(),
            size=2,
        )
        test_patients_luminal = np.random.choice(
            balanced_data[balanced_models["Model"] == "luminal"][
                "PatientName"
            ].unique(),
            size=2,
        )
        test_patients = np.concatenate((test_patients_basal, test_patients_luminal))
        train_X = balanced_data[~balanced_data["PatientName"].isin(test_patients)]
        test_X = balanced_data[balanced_data["PatientName"].isin(test_patients)]
        train_y = balanced_models[~balanced_data["PatientName"].isin(test_patients)]
        test_y = balanced_models[balanced_data["PatientName"].isin(test_patients)]
    else:
        train_X = data
        test_X = data
        train_y = models
        test_y = models
    single_feature_classification(
        train_X=train_X,
        train_y=train_y,
        val_X=test_X,
        val_y=test_y,
        config=config,
        out_path=f"{out_path}/feature_scores.csv",
    )
