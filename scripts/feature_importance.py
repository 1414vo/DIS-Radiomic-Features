from src.classification import reduced_feature_classification
from src.config_parser import parse_config
from src.utils import k_fold_split_by_patient
import pandas as pd
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Feature Importance",
        usage="python -m scripts.feature_imporance <data_path> --output <out_path> --cfg <config_path>",
        description="Determines the feature importance from SHAP plots",
    )
    parser.add_argument("data_path", help="Location of the raw data")
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
        data = pd.read_csv(f"{data_path}/Reduced_features.csv")
        patient_names = pd.read_csv(
            f"{data_path}/All_features_corrected_final_patient.csv"
        )["PatientName"]
        models = pd.read_csv(f"{data_path}/All_model.csv")["Model"]
    except FileNotFoundError:
        print(f"Could not find the data in {data_path}.")
        exit(1)

    try:
        config = parse_config(args.cfg)
    except FileNotFoundError:
        print(f"Could not find configuration file {args.cfg}.")
        exit(1)

    if args.validation:
        train_Xs, train_ys, test_Xs, test_ys, _ = k_fold_split_by_patient(
            data, models, patient_names
        )
    else:
        train_Xs = [data]
        test_Xs = [data.copy()]
        train_ys = [models]
        test_ys = [models.copy()]

    reduced_feature_classification(
        train_Xs=train_Xs,
        train_ys=train_ys,
        test_Xs=test_Xs,
        test_ys=test_ys,
        config=config,
        out_path=f"{out_path}/importance",
    )
