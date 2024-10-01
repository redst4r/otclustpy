import pandas as pd


def _jac(a, b):
    return len(set(a) & set(b)) / len(set(a) | set(b))


def jaccard_matrix(bs1: pd.Series, bs2: pd.Series):
    """
    for each pair of clusters across bootstraps:
    what the jaccard overlap, i.e frcation of samples that overlap
    """
    assert isinstance(bs1, pd.Series)
    assert isinstance(bs2, pd.Series)

    df_distance = []
    for c1 in bs1.unique():
        for c2 in bs2.unique():
            j = _jac(
                bs1[bs1 == c1].index,
                bs2[bs2 == c2].index,
            )
            df_distance.append({"source": c1, "target": c2, "base_distance": 1 - j})
    return pd.DataFrame(df_distance)
