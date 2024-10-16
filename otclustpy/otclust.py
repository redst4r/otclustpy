from .baseline_distances import jaccard_matrix
from .otproblem import OTProblem
import ot
import numpy as np
import itertools
import tqdm
import pandas as pd
from typing import List


def ot_distance(bs1: pd.Series, bs2: pd.Series):
    """
    Optimal transport distance between the two clusterings.
    Base distance (between two clusters) is the Jaccard-index, i.e.
    how many samples are shared by the two clusters.
    """
    assert isinstance(bs1, pd.Series)
    assert isinstance(bs2, pd.Series)
    """
    Compare the similarity of the two clusterings using OT.

    :param bs1: Clustering of first bootstrapped sample
    :param bs2: Clustering of second bootstrapped sample

    :return:
    - ot_d: distance
    - ot_matrix: optimal transport plan/matrix
    """
    # need to restrict to sasmpels that got clustered in both bs
    shared = set(bs1.index) & set(bs2.index)

    bs1_tmp = bs1[bs1.index.isin(shared)]
    bs2_tmp = bs2[bs2.index.isin(shared)]

    df_jac_flat = jaccard_matrix(bs1_tmp, bs2_tmp)

    weight1 = bs1_tmp.value_counts().to_dict()
    weight2 = bs2_tmp.value_counts().to_dict()

    ot_instance = OTProblem(df_jac_flat, weight1, weight2)
    ot_matrix, ot_d = ot_instance.solve()

    return ot_d, ot_matrix


def similarity_distance(ot_instance: OTProblem):
    """
    Compares the OT solution to a naive, "uncooperative" solution

    :param ot_instance: instance of an OT problem
    """

    dmatrix, w1, w2 = ot_instance._get_canonical_inputs()

    # ot distance
    d = ot.emd2(w1, w2, dmatrix.values)

    # compare that to the non-cooperative, naive distance
    # /sum pi * q_j d_ij
    d_NT = np.sum((w1.reshape(-1, 1) @ w2.reshape(1, -1)) * dmatrix.values)

    return d / d_NT


class OTClust:
    def __init__(self, bootstrapped_clusterings: List[pd.Series]):
        for bs in bootstrapped_clusterings:
            assert isinstance(bs, pd.Series)

        self.bs_results = bootstrapped_clusterings

    def calculate_bs_distances(self) -> pd.DataFrame:
        """
        calcualtes the distance between all bootstraps, returning a #bs x #bs matrix
        """

        D = []
        n_bootstraps = len(self.bs_results)
        for i, j in tqdm.tqdm(
            itertools.combinations(range(n_bootstraps), 2),
            total=0.5 * n_bootstraps * (n_bootstraps - 1),
        ):
            dot, M = ot_distance(self.bs_results[i], self.bs_results[j])
            D.append({"i": i, "j": j, "distance": dot})
            # reverse
            D.append({"i": j, "j": i, "distance": dot})

        # diagnoal
        for i in range(n_bootstraps):
            D.append({"i": i, "j": i, "distance": 0})

        return pd.DataFrame(D)

    def raincloud_jaccard(self, reference_clustering: pd.Series) -> pd.DataFrame:
        """For each cluster C in the reference, compare
        it against each bootstrap i and each cluster c_ij in that bootstrap
        via Jaccard and find the best (most optimistic) match:  max_j Jac(C, C_ij)

        If a cluster is highly stable it'll have a good match (Jac) across the majority of bootstraps

        simply plot by
        `pn.ggplot(df_rain) + pn.aes(x='ref_cluster', y='max_jaccard_similarity') + pn.geom_jitter()`
        """
        df_rain = []
        for bs in tqdm.tqdm(self.bs_results):
            # note: for the jaccard to make sense we need to restrict to common samples
            shared = set(reference_clustering.index) & set(bs.index)

            bs1_tmp = reference_clustering[reference_clustering.index.isin(shared)]
            bs2_tmp = bs[bs.index.isin(shared)]

            J_distance = jaccard_matrix(bs1_tmp, bs2_tmp)
            J_distance["jaccard_similarity"] = 1 - J_distance["base_distance"]

            # get the maximum jac similarity of each reference cluster (the most optimistic/best mapping)
            tmp = J_distance.groupby("source")["jaccard_similarity"].max()
            for ref_clust, jj in tmp.to_dict().items():
                df_rain.append(
                    {
                        "ref_cluster": ref_clust,
                        "max_jaccard_similarity": jj,
                    }
                )
        return pd.DataFrame(df_rain)
