import colorsys
import numpy as np
import pandas as pd
from functools import partial


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
        generate_color = np.vectorize(partial(colorsys.hsv_to_rgb, s=1.0, v=1.0))
        return generate_color(hues)


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
