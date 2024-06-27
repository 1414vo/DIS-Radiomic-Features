import numpy as np
import pandas as pd
from itertools import combinations


def manual_anova(data, factors, factor_names, unbalanced=True):
    """
    Produces a per-factor ANalysis Of VAriance (ANOVA) for a 1D feature. Different factor interactions
    are also considered. The evaluation requires that the factor distributions are balanced,
    and the factors can only be categorical.

    This function is a Python reproduction of the 'manualAnova2.m' file, found in the original
    GitHub repository https://github.com/loressa/Photoacoustic_radiomics_xenografts/tree/main.

    The 'sensitivity' is calculated by measuring the fraction of variance explained by the given interaction,
    excluding any sub-interactions.

    Parameters
    ----------
    data: ndarray[float] (N)
        The feature the analysis is performed on.
    factors: ndarray (k, N)
        A set of k factors. Each factor should be balanced (have the same number of entries per unique value).
    factor_names: list[str]
        The names of the factors - used to construct a final Pandas Series

    Returns
    -------
    A Pandas Series (pd.Series) object, containing the contribution of each interaction in terms
    of sensitivity.
    """
    num_measurements = len(data)
    num_factors = len(factors)

    data_mean = np.mean(data)
    sse = sum((data - data_mean) ** 2)

    # Create matrix of factor categorization
    unique_indeces = np.zeros((num_measurements, num_factors))
    for i in range(num_factors):
        _, unique_indeces[:, i] = np.unique(factors[i], return_inverse=True)
    max_classes = int(unique_indeces.max()) + 1

    explained_sse = {}

    for int_order in range(1, num_factors + 1):
        explained_sse[int_order] = {}
        # Generate combinations of the given interaction order
        comb_list = list(combinations(range(num_factors), int_order))

        for i, comb in enumerate(comb_list):
            int_factor_ind = unique_indeces[:, comb]
            # Represent combination index as a single unique index
            comb_indeces = np.sum(
                int_factor_ind * max_classes ** np.arange(len(comb)), axis=1
            )
            unique_combs = np.unique(comb_indeces)
            if unbalanced:
                comb_sse = 0
                for comb_idx in unique_combs:
                    mean = np.mean(data[comb_indeces == comb_idx])
                    comb_sse += (
                        np.sum(comb_indeces == comb_idx) * (mean - data_mean) ** 2
                    )
            else:
                scaling = num_measurements // len(unique_combs)
                means = np.array([])
                for comb_idx in unique_combs:
                    means = np.append(means, np.mean(data[comb_indeces == comb_idx]))

                # Compute the total sse contribution (without removing lower-level interactions)
                comb_sse = scaling * sum((means - data_mean) ** 2)
            register_factor_interaction_contribution(explained_sse, comb, comb_sse)

    return get_interaction_series(factor_names, explained_sse, sse)


def register_factor_interaction_contribution(explained_sse, comb, comb_sse):
    """
    Compute the sensitivity for a combination of factors given pre-computed SSE contributions.
    Updates the provided explained_sse dictionary in place.

    Parameters
    ----------
    explained_sse: dict[int, dict[int, float]]
        The calculated Explained SSE for each combination of factors.
    comb: list[int]
        The combination of factors.
    comb_sse: float
        The computed SSE for the given combination.
    """
    # First order interactions have no dependencies of lower order.
    if len(comb) == 1:
        explained_sse[len(comb)][comb] = comb_sse
    else:
        ss_to_remove = 0
        # Iterate over combinations
        for j in range(1, len(comb)):
            for lower_order_comb in explained_sse[j]:
                # If combination is subset of the given one, remove the contribution
                if set(lower_order_comb) <= set(comb):
                    ss_to_remove += explained_sse[j][lower_order_comb]

        explained_sse[len(comb)][comb] = comb_sse - ss_to_remove


def get_interaction_series(factor_names, explained_sse, total_sse):
    """
    Produces a Pandas Series from the extracted Sum of Squared Errors (SSE) showing the factor sensitivity.
    Sensitivity is measured as the fraction explained compared to the total SSE.

    Parameters
    ----------
    factor_names: list[str]
        The names of the factors - used to construct a final Pandas Series
    explained_sse: dict[int, dict[int, float]]
        The calculated Explained SSE for each combination of factors.
    total_sse: float
        The total Sum of Squared Errors.

    Returns
    -------
    A Pandas Series (pd.Series) object, containing the contribution of each interaction in terms
    of sensitivity.
    """
    interaction_dict = {}

    for int_order in range(1, len(factor_names) + 1):
        # Genarete the combinations and the corresponding names of the factors
        comb_list = list(combinations(range(len(factor_names)), int_order))
        comb_name_list = list(combinations(factor_names, int_order))

        for i, comb in enumerate(comb_list):
            # Compute the fraction of explained SSE
            interaction_dict[str(comb_name_list[i])] = (
                explained_sse[int_order][comb] / total_sse
            )

            # Limit every small contribution to 0 for numerical precision
            if abs(interaction_dict[str(comb_name_list[i])]) < 1e-10:
                interaction_dict[str(comb_name_list[i])] = 0
    return pd.Series(interaction_dict)
