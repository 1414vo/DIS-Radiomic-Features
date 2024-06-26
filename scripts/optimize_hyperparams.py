from src.utils import k_fold_split_by_patient, optimize_hyperparams
import pandas as pd
import argparse
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Hyperparameter optimization",
        usage="python -m scripts.optimize_hyperparams <data_path> --output <out_path> --grids <grids_path>",
        description="Determines the optimal hyperparameters for various classifiers.",
    )
    parser.add_argument("data_path", help="Location of the raw data")
    parser.add_argument(
        "-o",
        "--output",
        default="./out",
        help="Where to store the summary tables.",
    )
    parser.add_argument(
        "--grids",
        default="./grids.json",
        help="Where the grids of classifier configurations can be found.",
    )
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
        with open(args.grids, "r") as f:
            grids = json.load(f)
    except FileNotFoundError:
        print(f"Could not find grid configuration file {args.cfg}.")
        exit(1)

    _, _, _, _, folds = k_fold_split_by_patient(data, models, patient_names)
    optimize_hyperparams(
        data, models, folds, grids, out_path=f"{args.output}/config_cv.json"
    )
