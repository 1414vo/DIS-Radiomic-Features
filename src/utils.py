import colorsys
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression

# from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
# from sklearn.svm import SVC
# from sklearn.model_selection import GridSearchCV, PredefinedSplit


def get_feature_groups(feature_names):
    """
    Assuming a feature name of "original_GROUP_NAME", recovers
    the features split by group.

    Parameters
    ----------
    feature_names: list[str]
        A list of feature names

    Returns
    -------
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


def generate_colors(n_colors, colorblind_friendly=False):
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


def extract_factor_summary(data, factor_names):
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
    A pd.Series object containing a summary of first-order sensitivity and an associated
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
    def __init__(self, transform_options):
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

    def fit(self, X, no_voxels):
        for i in range(X.shape[1]):
            X_col = X[:, i]
            col_scaler = SingleNoVoxelsScaler(self.transform_options)
            col_scaler.fit(X_col, no_voxels)
            self.fits.append(col_scaler)

    def transform(self, X, no_voxels):
        assert X.shape[1] == len(
            self.fits
        ), f"Number of columns different from fit ({len(self.fits)})"
        X_norm = np.zeros_like(X)
        for i in range(X.shape[1]):
            X_col = X[:, i]
            X_norm[:, i] = self.fits[i].transform(X_col, no_voxels)

        return X_norm

    def fit_transform(self, X, no_voxels):
        self.fit(X, no_voxels)
        return self.transform(X, no_voxels)


class SingleNoVoxelsScaler(BaseEstimator):
    def __init__(self, transform_options):
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

    def _apply_transform(self, X, transform):
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

    def fit(self, X, no_voxels):
        best_score = 0
        best_regressor = None
        best_regressor_transform = None

        for transform in self.transform_options:
            regressor = LinearRegression(fit_intercept=True)
            transformed_voxels = self._apply_transform(no_voxels, transform).reshape(
                -1, 1
            )
            regressor.fit(transformed_voxels, X)

            if regressor.score(transformed_voxels, X) > best_score:
                best_score = regressor.score(transformed_voxels, X)
                best_regressor = regressor
                best_regressor_transform = transform

        self.model_fit = best_regressor
        self.model_transform = best_regressor_transform

    def transform(self, X, no_voxels):
        p0 = self.model_fit.intercept_
        return (X - p0) / self._apply_transform(no_voxels, self.model_transform)

    def fit_transform(self, X, no_voxels):
        self.fit(X, no_voxels)
        return self.transform(X, no_voxels)


def k_fold_split_by_patient(data, models, patient_names, k=5):
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
