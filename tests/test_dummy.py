import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from otclust.otclust import OTClust
from otclust.consensus import get_posterior_clustermatrix


def test_full():
    # some toy data
    d = 2
    k = 3
    _weights = np.array([1] * k)
    data_gmm = GaussianMixture(n_components=k)
    data_gmm.weights_ = _weights / _weights.sum()
    # data_gmm.means_ = np.random.random((k, d)) * 10
    # data_gmm.covariances_ = [np.diag(np.random.random(d)) for _ in range(k)]

    data_gmm.means_ = np.array([[-1.5, 0], [1.5, 0], [5, 5]])
    data_gmm.covariances_ = [np.diag([1] * d) for _ in range(k)]

    x, y = data_gmm.sample(100)

    def do_cluster(x, nclust):
        km = KMeans(n_clusters=nclust).fit(x)
        yhat = km.labels_
        return pd.DataFrame({"km": yhat, "x0": x[:, 0], "x1": x[:, 1]})

    # create bootstraps
    percent = 0.9
    bs_results = []
    n_bs = 10
    for i in range(n_bs):
        ix_sub = np.random.choice(len(x), int(len(x) * percent), replace=False)
        x_bs = x[ix_sub]
        # y_bs = y[ix_sub]

        # nclust = 2 if i % 2 == 0 else 3
        nclust = 3
        df_cluster = do_cluster(x_bs, nclust)
        df_cluster["sampleix"] = ix_sub
        bs_results.append(df_cluster.set_index("sampleix").km)

    c = OTClust(bs_results)
    D = c.calculate_bs_distances()

    assert D.shape[0] == n_bs * n_bs
    assert D.shape[1] == 3

    get_posterior_clustermatrix(
        reference_partition=bs_results[0], bs_results=bs_results[1:]
    )
