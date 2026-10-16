"""Deterministic synthetic market data — no network required."""

import numpy as np
import pandas as pd
import pytest

N_ASSETS = 8
N_DAYS = 756  # ~3 years


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(12345)


@pytest.fixture(scope="session")
def returns(rng) -> pd.DataFrame:
    """Correlated daily simple returns for 8 synthetic assets (GBM-ish)."""
    mu = np.linspace(0.02, 0.12, N_ASSETS) / 252
    vols = np.linspace(0.08, 0.30, N_ASSETS) / np.sqrt(252)
    corr = 0.3 + 0.7 * np.eye(N_ASSETS)
    L = np.linalg.cholesky(corr)
    z = rng.standard_normal((N_DAYS, N_ASSETS)) @ L.T
    rets = mu + z * vols
    idx = pd.bdate_range("2021-01-04", periods=N_DAYS)
    return pd.DataFrame(rets, index=idx,
                        columns=[f"A{i}" for i in range(N_ASSETS)])


@pytest.fixture(scope="session")
def prices(returns) -> pd.DataFrame:
    return 100 * (1 + returns).cumprod()


@pytest.fixture(scope="session")
def mu_cov(returns):
    from portlab.optimize import mean_cov
    return mean_cov(returns)

# ---- public-release declaration ---------------------------------------
# test_tbill_download fetches a live T-bill series from FRED via
# pandas_datareader. It is the suite's only network test. It is
# skipped by default so the shipped suite runs fully offline, and
# runs when the user opts in:
#     pip install "portlab[data]"
#     PORTLAB_LIVE=1 pytest tests/test_riskfree.py
import os as _os

import pytest as _pytest


def pytest_collection_modifyitems(items):
    if _os.environ.get("PORTLAB_LIVE") == "1":
        return
    for item in items:
        if item.name == "test_tbill_download":
            item.add_marker(_pytest.mark.skip(
                reason="live-network test (FRED download); "
                       "set PORTLAB_LIVE=1 to run it"))
