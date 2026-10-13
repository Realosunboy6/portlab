"""Hierarchical Risk Parity (Lopez de Prado 2016) — cluster-based allocation
that needs no matrix inversion and no solver, prized for robustness."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform


def _quasi_diag_order(corr: pd.DataFrame, method: str) -> list[int]:
    dist = np.sqrt(0.5 * (1 - corr.values).clip(0, 2))
    condensed = squareform(dist, checks=False)
    link = linkage(condensed, method=method)
    return list(leaves_list(link))


def _cluster_var(cov: np.ndarray, idx: list[int]) -> float:
    sub = cov[np.ix_(idx, idx)]
    ivp = 1.0 / np.diag(sub)
    w = ivp / ivp.sum()
    return float(w @ sub @ w)


def hrp(rets_or_cov: pd.DataFrame, method: str = "single",
        is_cov: bool | None = None) -> pd.Series:
    """Hierarchical Risk Parity weights.

    rets_or_cov: return panel (rows=time) or a covariance matrix (square,
    symmetric); pass is_cov=True to force interpretation.
    method: scipy linkage method ('single' per the original paper, 'ward'
    is a popular robust variant).
    """
    df = rets_or_cov
    square = df.shape[0] == df.shape[1] and list(df.index) == list(df.columns)
    if is_cov is None:
        is_cov = square
    if is_cov:
        cov = df.copy()
        sd = np.sqrt(np.diag(cov.values))
        corr = pd.DataFrame(cov.values / np.outer(sd, sd),
                            index=cov.index, columns=cov.columns)
    else:
        cov = df.cov()
        corr = df.corr()

    order = _quasi_diag_order(corr, method)
    assets = cov.columns[order]
    C = cov.loc[assets, assets].values

    w = pd.Series(1.0, index=assets)
    clusters = [list(range(len(assets)))]
    while clusters:
        clusters = [c[j:k] for c in clusters
                    for j, k in ((0, len(c) // 2), (len(c) // 2, len(c)))
                    if len(c) > 1]
        for a, b in zip(clusters[::2], clusters[1::2]):
            var_a, var_b = _cluster_var(C, a), _cluster_var(C, b)
            alloc_a = 1 - var_a / (var_a + var_b)
            w.iloc[a] *= alloc_a
            w.iloc[b] *= 1 - alloc_a
    w = w / w.sum()
    return w.reindex(rets_or_cov.columns).rename("weight")
