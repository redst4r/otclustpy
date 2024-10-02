from sklearn.preprocessing import OneHotEncoder
from .otclust import ot_distance
import pandas as pd
import numpy as np
import tqdm
from typing import List


def normalize_each_row(df: pd.DataFrame) -> pd.DataFrame:
    "weird pandas stuff"
    return df.div(df.sum(1), axis=0)


def cluster_assignment_matrix(bs: pd.Series) -> pd.DataFrame:
    """
    turn the vector into a 1hot matrix
    """
    enc = OneHotEncoder()
    posterior_cluster_matrix = enc.fit_transform(bs.values.reshape(-1, 1)).toarray()
    posterior_cluster_matrix = pd.DataFrame(
        posterior_cluster_matrix, columns=enc.categories_[0]
    )  # weird categories is a list of array
    return posterior_cluster_matrix


def get_posterior_clustermatrix(
    reference_partition: pd.Series, bs_results: List[pd.Series]
):
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

    for bs in tqdm.tqdm(bs_results):
        if id(bs) == id(reference_partition):
            continue

        # map the bootstrap clustering to the reference
        assignment_in_ref_tmp = _move_clustering_to_reference(reference_partition, bs)

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


def _move_clustering_to_reference(
    reference_partition: pd.Series, query: pd.Series
) -> pd.DataFrame:
    """
    given two clusterings `reference` and `query`
    establish a mapping between them using OT
    and move the query clustering onto the reference!

    essentially: uses the OT matrix M to map the 1hot encoded query clustering to the reference
    via matrix multiplication

    returns: the soft assignment vector for each datapoint (soft because datapoints might be split between reference clusters)
    """

    # mapping from bs to reference!
    # cols of M are the reference clusters
    dot, M = ot_distance(
        query,
        reference_partition,
    )

    # each row sums to one now, i.e. of the cluster_i how much of it maps to which reference-cluster
    Gamma_row = normalize_each_row(M)
    # Gamma_col = normalize_each_row(M.T).T

    # 1-hot membership matrix
    assignment_matrix = cluster_assignment_matrix(query)
    assignment_matrix.index = (
        query.index
    )  # so we can keep track of individual datapoints
    assert all(assignment_matrix.columns.values == Gamma_row.index.values)

    # where do those cells map in the reference
    assignment_in_ref_tmp = assignment_matrix @ Gamma_row
    assignment_in_ref_tmp.index = (
        query.index
    )  # so we can keep track of individual datapoints
    return assignment_in_ref_tmp
