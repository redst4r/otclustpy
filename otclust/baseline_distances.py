import pandas as pd


def _jac(a, b):
    return len(set(a) & set(b)) / len(set(a) | set(b))


def jaccard_matrix(bs1, bs2):
    """
    for each pair of clusters across bootstraps:
    what the jaccard overlap, i.e frcation of samples that overlap
    """
    df_distance = []
    for c1 in bs1.km.unique():
        for c2 in bs2.km.unique():
            j = _jac(
                bs1[bs1.km == c1].sampleix,
                bs2[bs2.km == c2].sampleix,
            )
            df_distance.append({"source": c1, "target": c2, "base_distance": 1 - j})
    return pd.DataFrame(df_distance)
