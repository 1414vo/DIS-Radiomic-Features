""" Utilities for reducing features from a large set.

This module contains statistical methods for determining the discriminative power of features,
correlation-based reduction and forward selection.
"""
import scipy.stats as stats
import pandas as pd
import numpy as np
import pingouin as pg
import operator
from tqdm import tqdm
from numpy.typing import NDArray
from src.plotting import plot_rm_corr_statistics
from src.utils import k_fold_split_by_patient
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import PredefinedSplit
from sklearn.feature_selection import SequentialFeatureSelector


def reduce_features(
    features: pd.DataFrame,
    models: NDArray[np.float64],
    feature_scores: NDArray[np.float64],
    method: str = "kw",
    benjamini_alpha: float = 0.25,
    corr_threshold: float = 0.9,
    out_path: str = None,
    config=None,
):
    """
    Applies feature reduction based on a statistical method or using forward selection.
    Further removes highly correlated features.

    Parameters
    ----------
    features: pd.DataFrame
        A Pandas DataFrame containing the radiomic features.

    models: NDArray[np.float64]
        An array containing the ground truth model for each observation.

    feature_scores: NDArray[np.float64]
        An array/series containing the individual predictive scores for each feature.

    method: Literal['kw', 'ks', 'fs']
        The feature reduction method to be applied. Supports "kw" (Kruskal-Wallis), \
        "ks" (Kolmogorov-Smirnov) and "fs" (Forward Selection)'

    benjamini_alpha: float
        The expected false discovery rate for the Benjamini-Hochberg correction. Defaults to 25%

    corr_threshold: float
        The threshold above which features are considered highly correlated. Defaults to 0.9

    out_path: str
        The folder where to store any intermediate results.

    config: dict[str, dict[str, float]]
        A dictionary containing the hyperparameters for classifiers.

    Returns
    -------
    pd.DataFrame
        The feature-reduced data.
    """
    assert method in [
        "kw",
        "ks",
        "fs",
    ], 'Method should be among "kw" (Kruskal-Wallis), "ks" (Kolmogorov-Smirnov) or "fs" (Forward Selection)'
    patient_names = features["PatientName"]

    if method in ["kw", "ks"]:
        # Remove Patient names from features
        feature_data = features.drop("PatientName", axis=1).to_numpy()

        if out_path is not None:
            out_path_kw = f"{out_path}/{method}_{benjamini_alpha}.csv"
        else:
            out_path_kw = None

        # Reduce using statistical test

        reduced_feature_names = reduce_features_statistical(
            features=feature_data,
            models=models,
            method=method,
            feature_names=list(features.columns[1:]),
            benjamini_alpha=benjamini_alpha,
            out_path=out_path_kw,
        )

    elif method == "fs":
        # Remove patient names from features
        feature_data = features.drop("PatientName", axis=1)

        if out_path is not None:
            out_path_fs = f"{out_path}/fs.csv"
        else:
            out_path_fs = None

        # Reduce using forward selection

        reduced_feature_names = reduce_feature_forward_selection(
            features=feature_data,
            config=config,
            patient_names=patient_names,
            models=models,
            out_path=out_path_fs,
        )

    # Perform first step of reduction
    reduced_features = features[["PatientName"] + reduced_feature_names]

    if out_path is not None:
        out_path_rm_corr = f"{out_path}/Reduced_features_{method}.csv"
    else:
        out_path_rm_corr = None

    # Remove highly correlated features

    reduced_features = reduce_features_rm_corr(
        features=reduced_features,
        feature_scores=feature_scores,
        corr_threshold=corr_threshold,
        out_path=out_path_rm_corr,
    )

    return reduced_features


def reduce_features_statistical(
    features: pd.DataFrame,
    models: NDArray[np.float64],
    feature_names,
    method="kw",
    benjamini_alpha=0.25,
    out_path=None,
):
    """
    Applies feature reduction based on a statistical method for determining
    feature discriminative power.

    Parameters
    ----------
    features: pd.DataFrame
        A Pandas DataFrame containing the radiomic features.

    models: NDArray[np.float64]
        An array containing the ground truth model for each observation.

    feature_scores: NDArray[np.float64]
        An array/series containing the individual predictive scores for each feature.

    method: Literal['kw', 'ks']
        The feature reduction method to be applied. Supports "kw" (Kruskal-Wallis) and \
        "ks" (Kolmogorov-Smirnov)

    benjamini_alpha: float
        The expected false discovery rate for the Benjamini-Hochberg correction. Defaults to 25%

    out_path: str
        The file where to store any intermediate results.

    Returns
    -------
    pd.DataFrame
        The feature-reduced data.
    """
    assert method in [
        "kw",
        "ks",
    ], 'Method should be among "kw" (Kruskal-Wallis) or "ks" (Kolmogorov-Smirnov)'
    p_vals = []

    # Compute p-values for each feature
    for i, feature in enumerate(features.T):
        mask = models == "basal"
        basal = feature[mask]
        luminal = feature[~mask]

        if method == "kw":
            p_vals.append(stats.kruskal(basal, luminal).pvalue)
        elif method == "ks":
            p_vals.append(stats.kstest(basal, luminal).pvalue)

    sorted_features = sorted(
        list(zip(p_vals, feature_names)), key=operator.itemgetter(0)
    )
    # Sort by p-values
    sorted_p_vals = np.array(list(zip(*sorted_features))[0])
    sorted_feature_names = list(zip(*sorted_features))[1]
    # Compute adjusted rank
    q_r = (np.arange(1, len(p_vals) + 1) * benjamini_alpha) / len(p_vals)

    # Save results if needed
    if out_path is not None:
        df = pd.DataFrame(
            {
                "Feature Name": sorted_feature_names,
                "p-value": sorted_p_vals,
                "rank": np.arange(1, len(p_vals) + 1),
                "corrected value": q_r,
            }
        )
        df.set_index("Feature Name")
        df.to_csv(out_path)

    # Return every sample where the adjusted rank is higher than the p-value
    last_viable = np.argmin(q_r > sorted_p_vals)

    return list(sorted_feature_names[:last_viable])


def reduce_features_rm_corr(
    features: pd.DataFrame,
    feature_scores: NDArray[np.float64],
    corr_threshold: float = 0.9,
    out_path: str = None,
):
    """
    Applies feature reduction on highly-correlated feature using the RMCorr measure.

    Parameters
    ----------
    features: pd.DataFrame
        A Pandas DataFrame containing the radiomic features.

    feature_scores: NDArray[np.float64]
        An array/series containing the individual predictive scores for each feature.

    corr_threshold: float
        The threshold above which features are considered highly correlated. Defaults to 0.9

    out_path: str
        The file where to store any intermediate results.

    Returns
    -------
    pd.DataFrame
        The feature-reduced data.
    """
    removed_features = set()
    norm_features = features.apply(lambda x: x / x.std(), axis=0)

    for i, f1 in enumerate(list(norm_features.columns)):
        for f2 in list(norm_features.columns)[i + 1 :]:
            # Ignore patient name column or already removed columns
            if f1 == "PatientName" or f2 == "PatientName" or f1 in removed_features:
                continue

            # Calculate RMCorr measure
            rm_corr = pg.rm_corr(
                norm_features, x=f1, y=f2, subject="PatientName"
            ).r.values[0]

            # Remove based on individual scores if threshold is met
            if rm_corr > corr_threshold:
                if (
                    feature_scores.loc[f1, "Overall Score (Train)"]
                    > feature_scores.loc[f2, "Overall Score (Train)"]
                ):
                    removed_features.add(f2)
                    print(f"Removed feature {f2}")
                else:
                    removed_features.add(f1)
                    print(f"Removed feature {f1}")
    reduced_features = features.drop(columns=list(removed_features) + ["PatientName"])

    if out_path is not None:
        reduced_features.to_csv(out_path, index=False)

    return reduced_features


def reduce_feature_forward_selection(
    features: pd.DataFrame,
    config,
    patient_names: NDArray[np.float64],
    models: NDArray[np.float64],
    out_path: str = None,
):
    """
    Applies feature reduction using sklearn's forward selection.

    Parameters
    ----------
    features: pd.DataFrame
        A Pandas DataFrame containing the radiomic features.

    config: dict[str, dict[str, float]]
        A dictionary containing the hyperparameters for classifiers.

    patient_names: NDArray[np.float64]
        An array/series containing the patient ID for each feature.

    models: NDArray[np.float64]
        An array containing the ground truth model for each observation.

    out_path: str
        The folder where to store any intermediate results.

    Returns
    -------
    pd.DataFrame
        The feature-reduced data.
    """
    model = RandomForestClassifier(**config["rfc"], random_state=0)
    _, _, _, _, folds = k_fold_split_by_patient(features, models, patient_names)
    cv = PredefinedSplit(folds)
    forward_selector = SequentialFeatureSelector(model, cv=cv, direction="forward")
    forward_selector.fit(features, models)
    reduced_features = features.copy()
    reduced_features = reduced_features[
        reduced_features.columns[forward_selector.get_support()]
    ]

    if out_path is not None:
        reduced_features.to_csv(out_path, index=False)

    return list(reduced_features.columns)


def compute_all_correlations(features: pd.DataFrame, out_path: str = None):
    """
    Computes the RMCorr measures and p-values for each pair of features.

    Parameters
    ----------
    features: pd.DataFrame
        A Pandas DataFrame containing the radiomic features.

    out_path: str
        The folder where to store any intermediate results.
    """
    n_features = len(features.columns) - 1
    norm_features = features.apply(lambda x: x / x.std(), axis=0)
    corrs = np.ones((n_features, n_features))
    p_vals = np.zeros((n_features, n_features))

    print("Computing correlations: ")

    with tqdm(total=n_features * (n_features - 1) // 2) as pbar:
        for i, f1 in enumerate(list(norm_features.columns)[1:]):
            for j, f2 in enumerate(list(norm_features.columns)[1:]):
                if i >= j:
                    continue
                rm_corr = pg.rm_corr(norm_features, x=f1, y=f2, subject="PatientName")
                corrs[i, j] = rm_corr.r.values[0]
                corrs[j, i] = rm_corr.r.values[0]
                p_vals[i, j] = rm_corr.pval.values[0]
                p_vals[j, i] = rm_corr.pval.values[0]
                pbar.update(1)
    with open("./out/corrs.npy", "wb") as f:
        np.save(f, corrs)
    with open("./out/pvals.npy", "wb") as f:
        np.save(f, p_vals)
    plot_rm_corr_statistics(corrs, p_vals, out_path)
