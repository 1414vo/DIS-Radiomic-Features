from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from .plotting import plot_logistic_regression_coef
import matplotlib.pyplot as plt
import pandas as pd
import shap


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
    train_X, train_y, test_X, test_y, config, out_path=None
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
        classifier.fit(train_X, train_y)
        print(f"{name} train accuracy: {classifier.score(train_X, train_y)}")
        print(f"{name} test accuracy: {classifier.score(test_X, test_y)}")

        if name in ["Gradient Boosting", "Random Forest"]:
            explainer = shap.TreeExplainer(classifier)
            explanation = explainer(test_X)
            if name == "Random Forest":
                explanation = explanation[:, :, 1]
            plt.figure(figsize=(40, 40))
            plt.subplot(1, 2, 1)
            shap.plots.beeswarm(explanation, show=False)
            plt.subplot(1, 2, 2)
            shap.plots.bar(explanation, show=False)
        else:
            plot_logistic_regression_coef(classifier, train_X)

        # Save or show dependent on the output path
        if out_path is None:
            plt.show()
        else:
            plt.savefig(f"{out_path}_{name}.png")
