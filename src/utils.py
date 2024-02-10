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
        group_name = feature_split[1].capitalize()

        if group_name not in feature_groups:
            feature_groups[group_name] = []
        feature_groups[group_name].append(feature)

    return feature_groups
