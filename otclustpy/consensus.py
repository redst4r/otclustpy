from sklearn.preprocessing import OneHotEncoder
from .otclust import ot_distance
import pandas as pd
import numpy as np


def normalize_each_row(df: pd.DataFrame):
    "weird pandas stuff"
    return df.div(df.sum(1), axis=0)


def cluster_assignment_matrix(bs: pd.Series):
    """
    turn the vector into a 1hot matrix
    """
    enc = OneHotEncoder()
    posterior_cluster_matrix = enc.fit_transform(bs.values.reshape(-1, 1)).toarray()
    posterior_cluster_matrix = pd.DataFrame(
        posterior_cluster_matrix, columns=enc.categories_[0]
    )  # weird categories is a list of array
    return posterior_cluster_matrix


def get_posterior_clustermatrix(reference_partition: pd.Series, bs_results):
    """
    given a bunch of bootstrapped clusterings and a reference partition,
    how stable is that reference partition?
    """

    # data x #clusters in ref
    assignments_in_ref = pd.DataFrame(
        np.zeros((reference_partition.shape[0], len(reference_partition.unique()))),
        index=reference_partition.index,
        columns=reference_partition.unique(),
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
        # Gamma_col = normalize_each_row(M.T).T

        # 1-hot membership matrix
        assignment_matrix = cluster_assignment_matrix(bs)
        assignment_matrix.index = (
            bs.index
        )  # so we can keep track of individual datapoints
        assert all(assignment_matrix.columns.values == Gamma_row.index.values)

        # where do those cells map in the reference
        assignment_in_ref_tmp = assignment_matrix @ Gamma_row
        assignment_in_ref_tmp.index = (
            bs.index
        )  # so we can keep track of individual datapoints

        # need to have the same column odrder!!
        assignment_in_ref_tmp = assignment_in_ref_tmp[assignments_in_ref.columns]
        assert all(assignments_in_ref.columns == assignment_in_ref_tmp.columns)

        # add to the total
        assignments_in_ref = assignments_in_ref.add(assignment_in_ref_tmp, fill_value=0)

    # normlize the final matrix of data x ref cluster
    # essentially the frequency of a datapoitn belonging to reference cluster i
    assignments_in_ref = assignments_in_ref.div(assignments_in_ref.sum(1), axis=0)
    assignments_in_ref["ground_truth"] = reference_partition

    return assignments_in_ref
