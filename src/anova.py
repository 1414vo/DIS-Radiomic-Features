import numpy as np
import pandas as pd
from itertools import combinations


def manual_anova(data, factors, factor_names):
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
    unique_indeces = np.zeros((num_measurements, num_factors))
    for i in range(num_factors):
        _, unique_indeces[:, i] = np.unique(factors[i], return_inverse=True)
    max_classes = int(unique_indeces.max()) + 1

    explained_sse = {}

    for int_order in range(1, num_factors + 1):
        explained_sse[int_order] = {}
        comb_list = list(combinations(range(num_factors), int_order))

        for i, comb in enumerate(comb_list):
            int_factor_ind = unique_indeces[:, comb]
            comb_indeces = np.sum(
                int_factor_ind * max_classes ** np.arange(len(comb)), axis=1
            )
            unique_combs = np.unique(comb_indeces)
            scaling = num_measurements // len(unique_combs)
            means = np.array([])
            for comb_idx in unique_combs:
                means = np.append(means, np.mean(data[comb_indeces == comb_idx]))
            cell_factor_sse = scaling * sum((means - data_mean) ** 2)
            # Simply deal with 1st order interactions
            if int_order == 1:
                explained_sse[int_order][comb] = cell_factor_sse
            else:
                ss_to_remove = 0
                for j in range(1, int_order):
                    for lower_order_comb in explained_sse[j]:
                        if set(lower_order_comb) <= set(comb):
                            ss_to_remove += explained_sse[j][lower_order_comb]

                explained_sse[int_order][comb] = cell_factor_sse - ss_to_remove

    interaction_dict = {}
    for int_order in range(1, num_factors + 1):
        comb_list = list(combinations(range(num_factors), int_order))
        comb_name_list = list(combinations(factor_names, int_order))
        for i, comb in enumerate(comb_list):
            interaction_dict[str(comb_name_list[i])] = (
                explained_sse[int_order][comb] / sse
            )
            if abs(interaction_dict[str(comb_name_list[i])]) < 1e-10:
                interaction_dict[str(comb_name_list[i])] = 0
    return pd.Series(interaction_dict)
