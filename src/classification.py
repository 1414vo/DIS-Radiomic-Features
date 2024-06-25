from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from .plotting import plot_logistic_regression_coef, plot_explanation
from sklearn.preprocessing import StandardScaler
import pandas as pd
import shap
import numpy as np


def single_feature_classification(
    train_X, train_y, val_X, val_y, config, out_path=None
):
    gbc = GradientBoostingClassifier(
        learning_rate=config["gbc"]["lr"],
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
    train_Xs, train_ys, test_Xs, test_ys, config, out_path=None
):
    gbc = GradientBoostingClassifier(
        learning_rate=config["gbc"]["lr"],
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
            scaler = StandardScaler()
            train_X[train_X.columns] = scaler.fit_transform(train_X)
            test_X[test_X.columns] = scaler.transform(test_X)
            classifier.fit(train_X, train_y)

            train_acc += classifier.score(train_X, train_y) / len(train_Xs)
            test_acc += classifier.score(test_X, test_y) / len(train_Xs)

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
                plot_logistic_regression_coef(
                    classifier, train_X, out_path=f"{out_path}_{name}.png"
                )
                print(f"{name} train accuracy: {classifier.score(train_X, train_y)}")
                print(f"{name} test accuracy: {classifier.score(test_X, test_y)}")
                return
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
