import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .utils import get_feature_groups, generate_colors


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
    fig, axs = plt.subplots(1, 2, figsize=(15, 5), sharey=True)

    im1 = axs[0].imshow(corrs, cmap="viridis", aspect="auto", vmin=-1, vmax=1)
    axs[0].set_title("Feature Correlations")
    axs[0].set_xlabel("Feature ID")
    axs[0].set_ylabel("Feature ID")

    im2 = axs[1].imshow(p_vals, cmap="viridis", aspect="auto", vmin=0, vmax=1)
    axs[1].set_title("RM-Corr p-values")
    axs[1].set_xlabel("Feature ID")

    cbar = fig.colorbar(im1, ax=axs.ravel().tolist(), shrink=0.95)
    cbar.set_label("Value")

    cbar = fig.colorbar(im2, ax=axs.ravel().tolist(), shrink=0.95)
    cbar.set_label("Value")

    plt.tight_layout()

    # Save or show dependent on the output path
    if out_path is None:
        plt.show()
    else:
        plt.savefig(out_path)
