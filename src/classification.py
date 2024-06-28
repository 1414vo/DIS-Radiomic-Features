"""Utilities for classification using a single or many features.

This module contains utilities for evaluating invidiual feature scores under different models,
as well as for providing explanations when evaluating on a reduced feature set. The latter includes
SHAP values for tree-based models, as well as a coefficient plot for the Logistic Regression.
"""

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from .plotting import plot_logistic_regression_coef, plot_explanation
from sklearn.preprocessing import StandardScaler
import pandas as pd
import shap
import numpy as np
from numpy.typing import NDArray
from typing import List


def single_feature_classification(
    train_X: NDArray[np.float64],
    train_y: NDArray[np.float64],
    val_X: NDArray[np.float64],
    val_y: NDArray[np.float64],
    config: dict,
    out_path: str = None,
) -> pd.DataFrame:
    """
    Computes the classification accuracy for individual features across a Gradient Boosting Classifier,
    Random Forest and Support Vector Machine

    Parameters
    ----------
    train_X: NDArray[np.float64]
        The features to train the models on.

    train_y: NDArray[np.float64]
        The ground truth models for the training set.

    val_X: NDArray[np.float64]
        The features to evaluate the models on.

    val_y: NDArray[np.float64]
        The ground truth models for the test set.

    config: dict[str, dict[str, float]]
        A configuration dictionary for the classifier models.

    out_path: str
        A file path where to store the immediate result.

    Returns
    -------
    pd.DataFrame
        A dataframe containing the scores for each individual feature.
    """
    gbc = GradientBoostingClassifier(
        learning_rate=config["gbc"]["learning_rate"],
        n_estimators=config["gbc"]["n_estimators"],
        max_depth=config["gbc"]["max_depth"],
        random_state=0,
    )
    rfc = RandomForestClassifier(
        n_estimators=config["rfc"]["n_estimators"],
        max_depth=config["rfc"]["max_depth"],
        random_state=0,
    )
    svc = SVC(
        C=config["svc"]["c"],
        kernel=config["svc"]["kernel"],
        degree=2,
        gamma=0.05,
        random_state=0,
    )
    results = {
        "SVM Score (Train)": [],
        "SVM Score (Val)": [],
        "Random Forest Score (Train)": [],
        "Random Forest Score (Val)": [],
        "Gradient Boosting Score (Train)": [],
        "Gradient Boosting Score (Val)": [],
        "Overall Score (Train)": [],
        "Overall Score (Val)": [],
    }
    for f in train_X.columns:
        # Fit models
        svc.fit(train_X[[f]], train_y)
        rfc.fit(train_X[[f]], train_y)
        gbc.fit(train_X[[f]], train_y)

        # Register scores
        results["SVM Score (Train)"].append(svc.score(train_X[[f]], train_y))
        results["SVM Score (Val)"].append(svc.score(val_X[[f]], val_y))
        results["Random Forest Score (Train)"].append(rfc.score(train_X[[f]], train_y))
        results["Random Forest Score (Val)"].append(rfc.score(val_X[[f]], val_y))
        results["Gradient Boosting Score (Train)"].append(
            gbc.score(train_X[[f]], train_y)
        )
        results["Gradient Boosting Score (Val)"].append(gbc.score(val_X[[f]], val_y))
        results["Overall Score (Train)"].append(
            (
                results["SVM Score (Train)"][-1]
                + results["Random Forest Score (Train)"][-1]
                + results["Gradient Boosting Score (Train)"][-1]
            )
            / 3
        )
        results["Overall Score (Val)"].append(
            (
                results["SVM Score (Val)"][-1]
                + results["Random Forest Score (Val)"][-1]
                + results["Gradient Boosting Score (Val)"][-1]
            )
            / 3
        )

    results_df = pd.DataFrame(results, index=train_X.columns)
    if out_path is not None:
        results_df.to_csv(out_path)

    return results_df


def reduced_feature_classification(
    train_Xs: List[NDArray[np.float64]],
    train_ys: List[NDArray[np.float64]],
    test_Xs: List[NDArray[np.float64]],
    test_ys: List[NDArray[np.float64]],
    config,
    out_path: str = None,
):
    """
    Computes the classification accuracy for individual features across a Gradient Boosting Classifier,
    Random Forest and Support Vector Machine

    Parameters
    ----------
    train_Xs: List[NDArray[np.float64]]
        The features to train the models on.

    train_ys: List[NDArray[np.float64]]
        The ground truth models for the training set.

    val_Xs: List[NDArray[np.float64]]
        The features to evaluate the models on.

    val_ys: List[NDArray[np.float64]]
        The ground truth models for the test set.

    config: dict[str, dict[str, float]]
        A configuration dictionary for the classifier models.

    out_path: str
        A file path where to store the immediate result.
    """
    gbc = GradientBoostingClassifier(
        learning_rate=config["gbc"]["learning_rate"],
        n_estimators=config["gbc"]["n_estimators"],
        max_depth=config["gbc"]["max_depth"],
        random_state=0,
    )
    rfc = RandomForestClassifier(
        n_estimators=config["rfc"]["n_estimators"],
        max_depth=config["rfc"]["max_depth"],
        random_state=0,
    )
    lr = LogisticRegression()

    for classifier, name in zip(
        [gbc, rfc, lr], ["Gradient Boosting", "Random Forest", "Logistic Regression"]
    ):
        shap_value_list = []
        base_values = []
        train_acc = 0
        test_acc = 0
        for i in range(len(train_Xs)):
            train_X, train_y, test_X, test_y = (
                train_Xs[i],
                train_ys[i],
                test_Xs[i],
                test_ys[i],
            )
            # Scale features
            scaler = StandardScaler()
            train_X[train_X.columns] = scaler.fit_transform(train_X)
            test_X[test_X.columns] = scaler.transform(test_X)
            classifier.fit(train_X, train_y)

            # Accumulate scores
            train_acc += classifier.score(train_X, train_y) / len(train_Xs)
            test_acc += classifier.score(test_X, test_y) / len(train_Xs)

            # Generate explanations for tree-based model
            if name in ["Gradient Boosting", "Random Forest"]:
                explainer = shap.TreeExplainer(classifier)
                explanation = explainer(test_X)
                if name == "Random Forest":
                    shap_values = explanation.values[:, :, 1]
                else:
                    shap_values = explanation.values
                shap_value_list.append(shap_values)
                base_values.append(explainer.expected_value)

            else:
                # Logistic regression coefficient plot
                plot_logistic_regression_coef(
                    classifier, train_X, out_path=f"{out_path}_{name}.png"
                )
                print(f"{name} train accuracy: {classifier.score(train_X, train_y)}")
                print(f"{name} test accuracy: {classifier.score(test_X, test_y)}")
                return

        # Combine explanations and plot
        shap_values = np.vstack(shap_value_list)
        base_value = np.mean(base_values)
        test_X = np.vstack([test_X.values for test_X in test_Xs])

        print(f"{name} train accuracy: {train_acc}")
        print(f"{name} test accuracy: {test_acc}")

        plot_explanation(
            shap_values,
            test_X,
            base_value,
            train_Xs[0].columns,
            out_path=f"{out_path}_{name}.png",
        )
