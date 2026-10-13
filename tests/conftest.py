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


# --- public-core extraction note -------------------------------------
# Two upstream tests exercise modules held back from the public
# core (portlab.mpo, portlab.report). They are skipped here with
# the reasons visible in the test report.
import pytest as _pytest


_HELD_BACK_TESTS = {
    "test_generic_interface_with_mpo":
        "requires portlab.mpo (held back from the public core)",
    "test_tear_sheet_monthly_window_sensible":
        "requires portlab.report (held back from the public core)",
}


def pytest_collection_modifyitems(items):
    for item in items:
        if item.name in _HELD_BACK_TESTS:
            item.add_marker(_pytest.mark.skip(
                reason=_HELD_BACK_TESTS[item.name]))
