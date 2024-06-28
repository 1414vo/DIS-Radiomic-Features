""" Miscalleneous utility functions.

The module contains utility functions for purposes not covered by other modules.
"""
import colorsys
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from numpy.typing import NDArray
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.preprocessing import StandardScaler
import json


def get_feature_groups(feature_names: List[str]) -> Dict[str, List[str]]:
    """
    Assuming a feature name of "original_GROUP_NAME", recovers
    the features split by group.

    Parameters
    ----------
    feature_names: list[str]
        A list of feature names

    Returns
    -------
    dict[str, list[str]]
        A dictionary containing the list of features per group.
    """
    feature_groups = {}
    for feature in feature_names:
        feature_split = feature.split("_")
        # Capitalize group name
        group_name = feature_split[1].upper()

        if group_name not in feature_groups:
            feature_groups[group_name] = []
        feature_groups[group_name].append(feature)

    return feature_groups


def generate_colors(n_colors: int, colorblind_friendly: bool = False):
    """Generates a number of distinct colors to be used for plotting.

    Parameters
    ----------
    n_colors: int
        The relevant number of colors.

    colorblind_friendly: bool
        Ensures that the colors are colorblind-friendly.
        Currently they are hardcoded and up to 8 are allowed. Defaults to False

    Returns
    -------
    List[str]
        A list of visually distinct colors given in Hexcode.
    """
    if colorblind_friendly:
        if n_colors > 8:
            raise ValueError("Cannot ensure more than 8 colorblind-friendly colors")
        return [
            "#000000",
            "#e69f00",
            "#56b3e9",
            "#009e73",
            "#f0e442",
            "#0072b2",
            "#d55e00",
            "#cc79a7",
        ][:n_colors]
    else:
        hues = np.linspace(0, 1, n_colors, endpoint=False)
        # Conversion function for HSV to RGB
        colors = [colorsys.hsv_to_rgb(h, 1.0, 1.0) for h in hues]
        # colors = [(int(255*c[0]), int(255*c[1]), int(255*c[2])) for c in colors]
        return colors


def extract_factor_summary(data: pd.Series, factor_names: List[str]):
    """
    Summarizes the factor analysis. We consider higher-order interactions as error.
    The resulting Series is then normalized to sum up to 1, as to compute the sensitivity.
    Assumes that the 1st order interactions comprise the first rows in the table.

    Parameters
    ----------
    data: pd.Series[float]
        A column of the calculated SSE values for a feature.
    factor_names: list[str]
        The names of the investigated factors.

    Returns
    -------
    pd.Series
        A pd.Series object containing a summary of first-order sensitivity and an associated\
        error component.
    """
    new_col = pd.Series(dtype=float)
    for i, name in enumerate(factor_names):
        new_col[name] = data.iloc[i]

    # Denote higher-order interactions as error
    new_col["Error"] = data.iloc[len(factor_names) :].sum()

    # Normalize and return
    return new_col / new_col.sum()


class NoVoxelsScaler(BaseEstimator):
    """
    A class for normalizing the features to a given factor (in this use case,
    the number of voxels)

    Arguments
    ----------
    transform_options: List[str]
        The options for the functional transform of the number of voxels. Should be among:
        'lin': :math:`f(x) = x`, 'sq': :math:`f(x) = x^2`, 'cub': :math:`f(x) = x^3`,
        'inv': :math:`f(x) = x^{-1}`, 'inv_sq': :math:`f(x) = x^{-2}`, 'inv_cub': :math:`f(x) = x^{-3}`,
        'log': :math:`f(x) = log(x)`,'inv_log': :math:`f(x) = log^{-1}(x)`
    """

    def __init__(self, transform_options: List[str]):
        """
        Creates an object for normalizing the features to a given factor (in this use case,
        the number of voxels)

        Parameters
        ----------
        transform_options: List[str]
            The options for the functional transform of the number of voxels. Should be among:
            'lin': :math:`f(x) = x`, 'sq': :math:`f(x) = x^2`, 'cub': :math:`f(x) = x^3`,
            'inv': :math:`f(x) = x^{-1}`, 'inv_sq': :math:`f(x) = x^{-2}`, 'inv_cub': :math:`f(x) = x^{-3}`,
            'log': :math:`f(x) = log(x)`,'inv_log': :math:`f(x) = log^{-1}(x)`
        """
        super(NoVoxelsScaler, self).__init__()
        for transform in transform_options:
            assert transform in [
                "lin",
                "sq",
                "cub",
                "inv",
                "inv_sq",
                "inv_cub",
                "log",
                "inv_log",
            ], f"Unexpected transformation {transform} received."
        self.transform_options = transform_options
        self.fits = []

    def fit(self, X: NDArray[np.float64], no_voxels: NDArray[np.float64]):
        """Fits the set of scalers for each feature.

        Parameters
        ----------
        X: NDArray[np.float64]
            The set of features.

        no_voxels: NDArray[np.float64]
            The number of voxels for each observation.
        """
        # Fit a scaler for each feature separately
        for i in range(X.shape[1]):
            X_col = X[:, i]
            df = pd.DataFrame({"feature": X_col, "voxels": no_voxels})

            # Bin data
            X_mean = df.groupby("voxels")["feature"].mean().to_numpy()
            X_err = df.groupby("voxels")["feature"].std().to_numpy()
            voxels = df.groupby("voxels")["voxels"].mean().to_numpy()

            # Fit scaler
            col_scaler = SingleNoVoxelsScaler(self.transform_options)
            col_scaler.fit(X_mean, X_err, voxels)
            self.fits.append(col_scaler)

    def transform(
        self, X: NDArray[np.float64], no_voxels: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Transforms the given features.

        Parameters
        ----------
        X: NDArray[np.float64]
            The set of features.

        no_voxels: NDArray[np.float64]
            The number of voxels for each observation.

        Returns
        -------
        NDArray[np.float64]
            The normalized features.
        """
        assert X.shape[1] == len(
            self.fits
        ), f"Number of columns different from fit ({len(self.fits)})"
        X_norm = np.zeros_like(X)
        for i in range(X.shape[1]):
            X_col = X[:, i]
            X_norm[:, i] = self.fits[i].transform(X_col, no_voxels)

        return X_norm

    def fit_transform(
        self, X: NDArray[np.float64], no_voxels: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Fits a scaler and applies the transformation for each feature.

        Parameters
        ----------
        X: NDArray[np.float64]
            The set of features.

        no_voxels: NDArray[np.float64]
            The number of voxels for each observation.

        Returns
        -------
        NDArray[np.float64]
            The normalized features.
        """
        self.fit(X, no_voxels)
        return self.transform(X, no_voxels)


class SingleNoVoxelsScaler(BaseEstimator):
    """
    A class for normalizing a single feature to a given factor (in this use case,
    the number of voxels)

    Arguments
    ----------
    transform_options: List[str]
        The options for the functional transform of the number of voxels. Should be among:
        'lin': :math:`f(x) = x`, 'sq': :math:`f(x) = x^2`, 'cub': :math:`f(x) = x^3`,
        'inv': :math:`f(x) = x^{-1}`, 'inv_sq': :math:`f(x) = x^{-2}`, 'inv_cub': :math:`f(x) = x^{-3}`,
        'log': :math:`f(x) = log(x)`,'inv_log': :math:`f(x) = log^{-1}(x)`
    """

    def __init__(self, transform_options: List[str], weighted: bool = True):
        """
        Creates an object for normalizing the features to a given factor (in this use case,
        the number of voxels)

        Parameters
        ----------
        transform_options: List[str]
            The options for the functional transform of the number of voxels. Should be among:
            'lin': :math:`f(x) = x`, 'sq': :math:`f(x) = x^2`, 'cub': :math:`f(x) = x^3`,
            'inv': :math:`f(x) = x^{-1}`, 'inv_sq': :math:`f(x) = x^{-2}`, 'inv_cub': :math:`f(x) = x^{-3}`,
            'log': :math:`f(x) = log(x)`,'inv_log': :math:`f(x) = log^{-1}(x)`
        weighted: bool
            Whether to perform the fit in a weighted least squares manner.
        """
        super(SingleNoVoxelsScaler, self).__init__()
        for transform in transform_options:
            assert transform in [
                "lin",
                "sq",
                "cub",
                "inv",
                "inv_sq",
                "inv_cub",
                "log",
                "inv_log",
            ], f"Unexpected transformation {transform} received."
        self.transform_options = transform_options
        self.model_fit = None
        self.weighted = weighted

    def __chisqr(
        self,
        obs: NDArray[np.float64],
        exp: NDArray[np.float64],
        err: NDArray[np.float64],
    ) -> float:
        """A helper function for computing the chi square statistic.

        Parameters
        ----------
        obs: NDArray[np.float64]
            The observed feature values.

        exp: NDArray[np.float64]
            The predicted feature values.

        err: NDArray[np.float64]
            The associated errors per observation.

        Returns
        -------
        The chi-squared statistic
        """
        return np.sum((obs - exp) ** 2 / err**2)

    def apply_transform(
        self, X: NDArray[np.float64], transform: str
    ) -> NDArray[np.float64]:
        """A helper function for applying the transformation on the number of voxels

        Parameters
        ----------
        X: NDArray[np.float64]
            The number of voxels.

        transform: str
            The type of transformation to be applied.

        Returns
        -------
            The transformed number of voxels.
        """
        if transform == "lin":
            return X
        elif transform == "sq":
            return X**2
        elif transform == "cub":
            return X**3
        elif transform == "inv":
            return 1 / X
        elif transform == "inv_sq":
            return 1 / X**2
        elif transform == "inv_cub":
            return 1 / X**3
        elif transform == "log":
            return np.log(X)
        elif transform == "inv_log":
            return 1 / np.log(X)
        else:
            return X

    def fit(
        self,
        X: NDArray[np.float64],
        X_err: NDArray[np.float64],
        no_voxels: NDArray[np.float64],
    ):
        """Fits the scaler on a single feature.

        Parameters
        ----------
        X: NDArray[np.float64]
            The feature data.

        X_err: NDArray[np.float64]
            The standard deviations measured for each binned observation.

        no_voxels: NDArray[np.float64]
            The number of voxels for each observation.
        """
        best_score = np.inf
        best_regressor = None
        best_regressor_transform = None

        # Iterate over transformation options
        for transform in self.transform_options:
            regressor = LinearRegression(fit_intercept=True)
            transformed_voxels = self.apply_transform(no_voxels, transform).reshape(
                -1, 1
            )

            # Fit using WLS/LS
            if self.weighted:
                regressor.fit(transformed_voxels, X, sample_weight=1 / X_err)
            else:
                regressor.fit(transformed_voxels, X)

            # Chi squared as scoring metric
            score = self.__chisqr(X, regressor.predict(transformed_voxels), X_err)
            if score < best_score:
                best_score = score
                best_regressor = regressor
                best_regressor_transform = transform

        self.model_fit = best_regressor
        self.model_transform = best_regressor_transform

    def transform(
        self, X: NDArray[np.float64], no_voxels: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Applies the feature transformation.

        Parameters
        ----------
        X: NDArray[np.float64]
            The feature data.

        no_voxels: NDArray[np.float64]
            The number of voxels for each observation.

        Returns
        -------
        NDArray[np.float64]
            The normalized features.
        """
        p0 = self.model_fit.intercept_
        return (X - p0) / self.apply_transform(no_voxels, self.model_transform)

    def fit_transform(
        self, X: NDArray[np.float64], X_err, no_voxels: NDArray[np.float64]
    ):
        """Fits the scaler and applies the transformation

        Parameters
        ----------
        X: NDArray[np.float64]
            The feature data.

        no_voxels: NDArray[np.float64]
            The number of voxels for each observation.

        Returns
        -------
        NDArray[np.float64]
            The normalized features.
        """
        self.fit(X, X_err, no_voxels)
        return self.transform(X, no_voxels)


def k_fold_split_by_patient(
    data: NDArray[np.float64],
    models: NDArray[np.float64],
    patient_names: NDArray[np.float64],
    k: int = 5,
) -> Tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Performs a k-fold split based on the patient tumour model in a stratified manner. Each patient's
    data will be present in either the training set or the test set for each fold (but not both simultaneously).

    Parameters
    ----------
    data: NDArray[np.float64]
        The input features.

    models: NDArray[np.float64]
        The cancer models (targets).

    patient_names: NDArray[np.float64]
        The patient identifiers for each observation.

    k: int
        The number of folds. Must be greater than the number of unique patient identifiers

    Returns
    -------
    NDArray[np.float64]
        The training set features.

    NDArray[np.float64]
        The training set observations.

    NDArray[np.float64]
        The test set features.

    NDArray[np.float64]
        The test set observations.

    NDArray[np.float64]
        An array of which observation belongs to each fold (given by an index).
    """
    # Generate seeds for random selection
    np.random.seed(0)

    basal_patients = np.unique(patient_names[models == "basal"])
    luminal_patients = np.unique(patient_names[models == "luminal"])

    assert (
        len(basal_patients) > k
    ), f"Data should contain at least k={k} patients from the basal model"

    assert (
        len(luminal_patients) > k
    ), f"Data should contain at least k={k} patients from the basal model"

    test_len = min(len(basal_patients) // k, len(luminal_patients) // k)
    np.random.shuffle(basal_patients)
    np.random.shuffle(luminal_patients)

    train_Xs = []
    train_ys = []
    test_Xs = []
    test_ys = []
    folds = -np.ones(len(models))

    for i in range(k):
        test_patients = np.concatenate(
            (
                basal_patients[i * test_len : (i + 1) * test_len],
                luminal_patients[i * test_len : (i + 1) * test_len],
            )
        )
        train_Xs.append(data[~np.isin(patient_names, test_patients)].copy())
        test_Xs.append(data[np.isin(patient_names, test_patients)].copy())
        train_ys.append(models[~np.isin(patient_names, test_patients)].copy())
        test_ys.append(models[np.isin(patient_names, test_patients)].copy())
        folds[np.isin(patient_names, test_patients)] = i

    return train_Xs, train_ys, test_Xs, test_ys, folds


def optimize_hyperparams(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    folds: NDArray[np.int64],
    grids: Dict[str, Dict[str, List[float]]],
    out_path: str = None,
):
    """Performs hyperparameter optimization using grid search

    Parameters
    ----------
    X: NDArray[np.float64]
        The input features.

    y: NDArray[np.float64]
        The target values.

    folds: NDArray[np.int64]
        To what cross-validation fold each observation belongs to.

    grids: Dict[str, Dict[str, List[float]]]
        A dictionary containing the guesses for each parameter for each model.

    out_path: str
        Where to store the optimized configurations
    """
    cv = PredefinedSplit(folds)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    new_params = {k: {} for k in grids.keys()}

    for classifier in grids.keys():
        if classifier == "rfc":
            classifier_model = RandomForestClassifier(random_state=0)
        elif classifier == "gbc":
            classifier_model = GradientBoostingClassifier(random_state=0)
        elif classifier == "svc":
            classifier_model = SVC(degree=2, gamma=0.05, random_state=0)

        classifier_model = GridSearchCV(
            classifier_model, param_grid=grids[classifier], cv=cv, verbose=10
        )

        print(f"Fitting {classifier}")
        classifier_model.fit(X, y)

        print(f"Best parameters for {classifier}:")

        # Store best fit parameters
        for k in grids[classifier].keys():
            print(f"{k}: {classifier_model.best_params_[k]}")
            new_params[classifier][k] = classifier_model.best_params_[k]

    if out_path is not None:
        with open(out_path, "w") as f:
            json.dump(new_params, f)
