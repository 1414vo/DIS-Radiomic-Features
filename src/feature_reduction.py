import scipy.stats as stats
import pandas as pd
import numpy as np
import pingouin as pg
import operator
from tqdm import tqdm
from src.plotting import plot_rm_corr_statistics


def reduce_features(
    features, models, feature_scores, benjamini_alpha=0.25, out_path=None
):
    feature_data = features.drop("PatientName", axis=1).to_numpy()

    if out_path is not None:
        out_path_kw = f"{out_path}/kw_{benjamini_alpha}.csv"
    else:
        out_path_kw = None

    reduced_feature_names = reduce_features_kw(
        features=feature_data,
        models=models,
        feature_names=list(features.columns[1:]),
        benjamini_alpha=benjamini_alpha,
        out_path=out_path_kw,
    )
    reduced_features = features[["PatientName"] + reduced_feature_names]

    if out_path is not None:
        out_path_rm_corr = f"{out_path}/rm_corr.csv"
    else:
        out_path_rm_corr = None

    reduced_features = reduce_features_rm_corr(
        features=reduced_features,
        feature_scores=feature_scores,
        out_path=out_path_rm_corr,
    )

    return reduced_features


def reduce_features_kw(
    features, models, feature_names, benjamini_alpha=0.25, out_path=None
):
    p_vals = []
    for i, feature in enumerate(features.T):
        mask = models == "basal"
        basal = feature[mask]
        luminal = feature[~mask]
        p_vals.append(stats.kruskal(basal, luminal).pvalue)

    sorted_features = sorted(
        list(zip(p_vals, feature_names)), key=operator.itemgetter(0)
    )

    sorted_p_vals = np.array(list(zip(*sorted_features))[0])
    sorted_feature_names = list(zip(*sorted_features))[1]
    q_r = (np.arange(1, len(p_vals) + 1) * benjamini_alpha) / len(p_vals)

    # Save results if needed
    if out_path is not None:
        df = pd.DataFrame(
            {
                "Feature Name": sorted_feature_names,
                "p-value": sorted_p_vals,
                "rank": np.arange(1, len(p_vals) + 1),
                "corrected value": q_r,
            }
        )
        df.set_index("Feature Name")
        df.to_csv(out_path)

    last_viable = np.argmin(q_r > sorted_p_vals)

    return list(sorted_feature_names[:last_viable])


def reduce_features_rm_corr(features, feature_scores, out_path=None):
    removed_features = set()
    norm_features = features.apply(lambda x: x / x.std(), axis=0)

    for i, f1 in enumerate(list(norm_features.columns)):
        for f2 in list(norm_features.columns)[i + 1 :]:
            if (
                f1 == "PatientName"
                or f2 == "PatientName"
                or f1 in removed_features
                or f2 in removed_features
            ):
                continue
            rm_corr = pg.rm_corr(
                norm_features, x=f1, y=f2, subject="PatientName"
            ).r.values[0]
            if rm_corr > 0.9:
                if (
                    feature_scores.loc[f1, "Overall Score (Train)"]
                    >= feature_scores.loc[f2, "Overall Score (Train)"]
                ):
                    removed_features.add(f2)
                    print(f"Removed feature {f2}")
                else:
                    removed_features.add(f1)
                    print(f"Removed feature {f1}")
    reduced_features = features.drop(columns=list(removed_features))
    if out_path is not None:
        reduced_features.to_csv(out_path)
    return reduce_features


def compute_all_correlations(features, out_path=None):
    n_features = len(features.columns) - 1
    norm_features = features.apply(lambda x: x / x.std(), axis=0)
    corrs = np.ones((n_features, n_features))
    p_vals = np.zeros((n_features, n_features))

    print("Computing correlations: ")

    with tqdm(total=n_features * (n_features - 1) // 2) as pbar:
        for i, f1 in enumerate(list(norm_features.columns)[1:]):
            for j, f2 in enumerate(list(norm_features.columns)[1:]):
                if i >= j:
                    continue
                rm_corr = pg.rm_corr(norm_features, x=f1, y=f2, subject="PatientName")
                corrs[i, j] = rm_corr.r.values[0]
                corrs[j, i] = rm_corr.r.values[0]
                p_vals[i, j] = rm_corr.pval.values[0]
                p_vals[j, i] = rm_corr.pval.values[0]
                pbar.update(1)
    with open("./out/corrs.npy", "wb") as f:
        np.save(f, corrs)
    with open("./out/pvals.npy", "wb") as f:
        np.save(f, p_vals)
    plot_rm_corr_statistics(corrs, p_vals, out_path)
