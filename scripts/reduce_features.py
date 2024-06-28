"""A script for determining the most discriminative features within a reduced feature set.

Uses statistical methods for determining the discriminative power of features, or forward selection for the
initial reduction, followed by a correlation-based filter.

Usage:

.. code:: bash

    $ python -m scripts.feature_importance <data_path> -o <output_path> --cfg <config_path> \
        --scores <scores> --method <method>

- *data_path*: The location of the data.
- *output_path*: Where to store the outputs.
- *config_path*: The location of the classifier configuration files.
- *scores*: Where the individual feature scores are found.
- *method*: Which method to use for the initial feature reduction. Supports 'kw' (Kruskal-Wallist), \
    'ks' (Kolmogorov-Smirnov), 'fs' (Forward selection)
"""

from src.feature_reduction import compute_all_correlations, reduce_features
from src.config_parser import parse_config
import numpy as np
import pandas as pd
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Feature Reduction",
        usage="python -m scripts.reduce_features <data_path> --output <out_path>"
        + " --scores <score_path> --cfg <config_path>",
        description="Reduces the features by observing the distributions and correlations",
    )
    parser.add_argument("data_path", help="Location of the raw data")
    parser.add_argument(
        "-o",
        "--output",
        default="./out",
        help="Where to store the summary tables.",
    )
    parser.add_argument(
        "-s",
        "--scores",
        default="./out/feature_scores.csv",
        help="Where the individual feature scores are found.",
    )
    parser.add_argument(
        "--cfg",
        default="./config.json",
        help="Where the classifier configurations can be found.",
    )
    parser.add_argument(
        "--method",
        default="kw",
        choices=["kw", "ks", "fs"],
        help="Method for the feature reduction. Supports Kruskal-Wallis (kw), "
        + "Kolmogorov-Smirnov (ks), and Forward Selection (fs)",
    )
    args = parser.parse_args()
    data_path = args.data_path
    out_path = args.output
    score_path = args.scores
    try:
        data = pd.read_csv(f"{data_path}/All_features_corrected_final_patient.csv")
        models = pd.read_csv(f"{data_path}/All_model.csv")["Model"]
        feature_scores = pd.read_csv(args.scores, index_col=0)
    except FileNotFoundError:
        print(f"Could not find the data in {data_path}.")
        exit(1)

    try:
        config = parse_config(args.cfg)
    except FileNotFoundError:
        print(f"Could not find configuration file {args.cfg}.")
        exit(1)

    # Generate seeds for random selection
    np.random.seed(0)

    reduce_features(
        data,
        models,
        feature_scores,
        benjamini_alpha=0.25,
        out_path=args.output,
        method=args.method,
        config=config,
    )

    # Compute all correlations for main contribution
    if args.method == "kw":
        compute_all_correlations(data, out_path=f"{args.output}/corrs.png")
