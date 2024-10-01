from sklearn.preprocessing import OneHotEncoder
from .otclust import ot_distance
import pandas as pd
import numpy as np


def normalize_each_row(df):
    "weird pandas stuff"
    return df.div(df.sum(1), axis=0)


def cluster_assignment_matrix(bs):
    """
    turn the vector into a 1hot matrix
    """
    enc = OneHotEncoder()
    posterior_cluster_matrix = enc.fit_transform(bs.km.values.reshape(-1, 1)).toarray()
    posterior_cluster_matrix = pd.DataFrame(
        posterior_cluster_matrix, columns=enc.categories_[0]
    )  # weird categories is a list of array
    return posterior_cluster_matrix


def get_posterior_clustermatrix(reference_partition, bs_results):
    """
    given a bunch of bootstrapped clusterings and a reference partition,
    how stable is that reference partition?
    """

    assignments_in_ref = np.zeros(
        (
            reference_partition.shape[0],
            len(reference_partition.km.unique()),
        )  # #data x #clusters in ref
    )
    assignments_in_ref = pd.DataFrame(
        assignments_in_ref, index=np.arange(reference_partition.shape[0])
    )

    for bs in bs_results:
        if id(bs) == id(reference_partition):
            continue

        # mapping from bs to reference!
        # cols of M are the reference clusters
        dot, M = ot_distance(
            bs,
            reference_partition,
        )

        # each row sums to one now, i.e. of the cluster_i how much of it maps to which reference-cluster
        Gamma_row = normalize_each_row(M)
        Gamma_col = normalize_each_row(M.T).T

        # 1-hot membership matrix
        assignment_matrix = cluster_assignment_matrix(bs)
        assignment_matrix.index = (
            bs.sampleix
        )  # so we can keep track of individual datapoints
        assert all(assignment_matrix.columns.values == Gamma_row.index.values)

        # where do those cells map in the reference
        assignment_in_ref_tmp = assignment_matrix @ Gamma_row
        assignment_in_ref_tmp.index = (
            bs.sampleix
        )  # so we can keep track of individual datapoints
        assignments_in_ref = assignments_in_ref.add(assignment_in_ref_tmp, fill_value=0)

    # normlize the final matrix of data x ref cluster
    # essentially the frequency of a datapoitn belonging to reference cluster i
    assignments_in_ref = assignments_in_ref.div(assignments_in_ref.sum(1), axis=0)
    assignments_in_ref["ground_truth"] = reference_partition.km

    return assignments_in_ref
