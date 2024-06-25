import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .utils import get_feature_groups, generate_colors
from matplotlib import rcParams
import re
import shap


def __modify_params():
    rcParams["pdf.fonttype"] = 42
    rcParams["ps.fonttype"] = 42
    rcParams["font.family"] = "serif"
    rcParams["font.sans-serif"] = ["Palatio"]
    rcParams["text.usetex"] = True
    rcParams["text.latex.preamble"] = r"\usepackage{amsfonts}"


def __shorten_name(name, max_len=23, abbreviate=True):
    type, feature = name.split(" ")
    feature_words = re.findall("[A-Z][^A-Z]*", feature)

    if abbreviate:
        if len(name) > max_len:
            return type + " " + "".join([word[0] for word in feature_words])
        return name
    else:
        shorter_name = type + " "
        for word in feature_words:
            shorter_name += word
            if len(shorter_name) > max_len:
                return shorter_name + "..."
        return shorter_name


def plot_interaction_summary(
    interaction_summary: pd.DataFrame,
    output_path: str = None,
    title: str = None,
    colors=None,
) -> None:
    """
    Creates a bar chart of the sensitivity analysis.

    Parameters
    ----------
    interaction_summary: pd.DataFrame
        The summarized sensitivity analysis for the factors.
    output_path: str
        Where to save the plot. If None, displays it directly.
    """
    __modify_params()
    plt.figure(dpi=300)
    feature_groups = get_feature_groups(interaction_summary.columns)
    # Dictionary keeping track of stack height
    bottom = {group: np.zeros(len(feature_groups[group])) for group in feature_groups}
    first_group = list(feature_groups.keys())[0]

    # Get bar colors per factor
    if colors is None:
        colors = generate_colors(len(interaction_summary.index))
    for group in feature_groups:
        for i, factor in enumerate(interaction_summary.index):
            bar = plt.bar(
                feature_groups[group],
                interaction_summary.loc[factor, feature_groups[group]].values,
                bottom=bottom[group],
                label=factor if group == first_group else None,
                color=colors[i],
            )
            bottom[group] += interaction_summary.loc[
                factor, feature_groups[group]
            ].values
        # Plot group label and separation bar
        plt.text(bar[0].get_corners()[0, 0] + 1, 1.02, group, fontsize=8)
        plt.bar(f"empty{group}", 0)

    plt.legend()
    plt.xticks([])
    plt.ylabel(r"Factor Sensitivity ($\eta^2$)")
    if title is not None:
        plt.title(title, y=1.06)

    # Save or show dependent on the output path
    if output_path is None:
        plt.show()
    else:
        plt.savefig(output_path)


def plot_rm_corr_statistics(corrs, p_vals, out_path):
    __modify_params()
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=True, dpi=300)

    im1 = axs[0].imshow(corrs, cmap="viridis", aspect="auto", vmin=-1, vmax=1)
    axs[0].set_title("Feature Correlations")
    axs[0].set_xlabel("Feature ID", loc="right")
    axs[0].set_ylabel("Feature ID", loc="top")
    axs[0].set_xticks(range(0, 100, 10))
    axs[0].set_yticks(range(0, 100, 10))
    axs[0].invert_yaxis()

    im2 = axs[1].imshow(p_vals, cmap="viridis", aspect="auto", vmin=0, vmax=1)
    axs[1].set_title("RM-Corr p-values")
    axs[1].set_xlabel("Feature ID", loc="right")
    axs[1].set_xticks(range(0, 100, 10))
    axs[1].invert_yaxis()

    cbar = fig.colorbar(im1, ax=axs[0], shrink=0.95, pad=0.01)
    cbar.set_ticks(np.arange(-1, 1.2, 0.2))
    fig.colorbar(im2, ax=axs[1], shrink=0.95, pad=0.01)

    plt.tight_layout()

    # Save or show dependent on the output path
    if out_path is None:
        plt.show()
    else:
        plt.savefig(out_path)


def plot_logistic_regression_coef(model, train_X, out_path=None):
    __modify_params()
    coefficients = model.coef_[0]
    feature_names = train_X.columns

    X = np.hstack([np.ones((train_X.shape[0], 1)), train_X])
    probs = model.predict_proba(train_X)
    probs = np.diag(probs[:, 0] * probs[:, 1])
    covariance_matrix = np.linalg.inv(X.T @ probs @ X)

    standard_errors = np.sqrt(np.diag(covariance_matrix))[1:]

    sorted_indices = np.argsort(abs(coefficients))[::-1][:15]
    sorted_coefficients = abs(coefficients[sorted_indices])
    sorted_errors = 2 * standard_errors[sorted_indices]
    sorted_feature_names = [__shorten_name(feature_names[i]) for i in sorted_indices]

    _, ax = plt.subplots(figsize=(7, 5), dpi=300)

    cmap = plt.get_cmap("viridis")
    colors = cmap(np.linspace(0.1, 0.9, len(sorted_coefficients)))[::-1]
    for i in range(len(sorted_coefficients)):
        point = ax.errorbar(
            i,
            sorted_coefficients[i],
            yerr=sorted_errors[i],
            fmt="o",
            color=colors[i],
            ecolor=colors[i],
            capsize=5,
        )
        if i == len(sorted_coefficients) // 2:
            point.set_label(r"Coefficient $\pm 2\sigma$")
    ax.set_xticks(np.arange(len(sorted_coefficients)))
    ax.set_xticklabels(sorted_feature_names, rotation=35, ha="right")
    ax.set_ylabel("Absolute Value of Coefficient", fontsize=14)
    ax.set_ylim(0)
    ax.legend(fontsize=12)
    ax.set_facecolor((0.92, 0.92, 0.92))

    plt.tight_layout()

    # Save or show dependent on the output path
    if out_path is None:
        plt.show()
    else:
        plt.savefig(out_path)


def plot_explanation(shap_values, X_test, base_value, column_names, out_path=None):
    columns = [__shorten_name(col) for col in column_names]
    explanation = shap.Explanation(
        shap_values, data=X_test, base_values=base_value, feature_names=columns
    )

    __modify_params()
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 5))
    plt.sca(ax1)
    shap.plots.beeswarm(explanation, show=False, plot_size=None)
    shap.plots.bar(explanation, show=False, ax=ax2)
    plt.tight_layout()

    # Save or show dependent on the output path
    if out_path is None:
        plt.show()
    else:
        plt.savefig(out_path)
