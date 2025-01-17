"""
Parses configuration files for model training/evaluation setup.

This module offers functionality to parse .json configuration files, converting them into
a Python dictionary for easy access and manipulation within a training or evaluation script.
"""
import json


def parse_config(cfg_path: str) -> dict:
    """Parses a .json configuration file for the use of setting up the training/evaluation script.

    Parameters
    ----------
    cfg_path: str or None
        The location of the config file. If None, will use a default configuration.

    Returns
    -------
    dict[str, dict[str, float]]
        A dictionary containing the default configuration parameters.
    """

    # Extract configuration
    cfg = {}

    # Use default if no path is given
    if cfg_path is None:
        cfg = {
            "gbc": {
                "lr": 1.0,
                "n_estimators": 100,
                "max_depth": 3,
            },
            "rfc": {
                "n_estimators": 100,
                "max_depth": 3,
            },
            "svc": {
                "c": 1.0,
                "kernel": "rbf",
            },
        }

    # Otherwise, load from path
    else:
        with open(cfg_path, "r") as cfg_file:
            cfg = json.load(cfg_file)
    return cfg
