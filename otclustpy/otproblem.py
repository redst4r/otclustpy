import ot
import pandas as pd
import numpy as np
from typing import Tuple


class OTProblem:
    """
    class representing a standard OT problem to be solved
    1. a baseline distance between each pair of entities
    2. freq of each entity in source and target
    """

    def __init__(
        self, baseline_dist_flat: pd.DataFrame, row_freq: dict, col_freq: dict
    ):
        assert "source" in baseline_dist_flat.columns
        assert "target" in baseline_dist_flat.columns
        assert "base_distance" in baseline_dist_flat.columns

        # make it a wide matrix, i.e. entity x entity
        d_base = pd.crosstab(
            baseline_dist_flat.source,
            baseline_dist_flat.target,
            values=baseline_dist_flat.base_distance,
            aggfunc="mean",
        )

        self.baseline_dist = d_base
        self.row_freq = row_freq
        self.col_freq = col_freq

    def _get_canonical_inputs(self):
        # get the order correct, normalize
        weight1 = np.array([self.row_freq[_] for _ in self.baseline_dist.index.values])
        weight1 = weight1 / weight1.sum()
        weight2 = np.array(
            [self.col_freq[_] for _ in self.baseline_dist.columns.values]
        )
        weight2 = weight2 / weight2.sum()

        return (self.baseline_dist, weight1, weight2)

    def solve(self) -> Tuple[pd.DataFrame, float]:
        """
        Solve the OT problem at hand

        :returns:
        - the optimal transport matrix `M` (M.sum()==1)
        - the optimal transport distance `ot_distance`
        """
        d, w1, w2 = self._get_canonical_inputs()
        M = pd.DataFrame(
            ot.emd(w1, w2, d.values),
            index=d.index,
            columns=d.columns,
        )
        ot_distance = (M * d).values.sum()
        return M, ot_distance
