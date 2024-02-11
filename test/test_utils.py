from src.utils import get_feature_groups, extract_factor_summary
import pandas as pd
import numpy as np
import math


def test_group_split():
    features = [
        "original_A_1",
        "original_B_1",
        "original_A_2",
        "original_A_3",
        "original_ABCD_2",
        "original_C_3",
        "original_A_4",
        "original_B_2",
    ]
    groups = set(get_feature_groups(feature_names=features))
    assert groups == set(["A", "B", "ABCD", "C"])


def test_factor_summary():
    feature_col = pd.Series(
        {
            "1": 2e-3,
            "2": 1e-3,
            "3": 4e-3,
            "1,2": 5e-4,
            "1,3": 1e-3,
            "2,3": 2e-3,
            "1,2,3": 5e-4,
        }
    )
    ans = extract_factor_summary(feature_col, ["1", "2", "3"])
    true_ans = pd.Series({"1": 0.2, "2": 0.1, "3": 0.4, "Error": 0.3})
    assert np.all([math.isclose(ans[id], true_ans[id])] for id in true_ans.index)
