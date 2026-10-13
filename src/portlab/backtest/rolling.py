"""Walk-forward (rolling-window) strategy backtest.

Refactor of the RollingBacktest class from Portfolio_Optimization_COLAB:
the CFG global is gone (explicit parameters), plotting lives in plots.py,
and the strategy is any callable `weights_fn(train_returns) -> pd.Series`,
so every optimizer in portlab.optimize plugs in directly.
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from ..config import DEFAULT_RF, TRADING_DAYS
from .. import metrics as M


def equal_weight(train: pd.DataFrame) -> pd.Series:
    return pd.Series(1.0 / train.shape[1], index=train.columns)


class RollingBacktest:
    """Retrain on a lookback window, hold weights for a rebalance block.

    train_window: lookback length in periods (e.g. 252 daily).
    rebalance_every: holding block length in periods (e.g. 21 daily).
    min_history: drop assets with fewer valid observations in the window.
    tc_bps: one-way transaction cost charged on turnover at each rebalance.
    """

    def __init__(self, returns: pd.DataFrame, train_window: int = 252,
                 rebalance_every: int = 21, tc_bps: float = 0.0,
                 min_history: int | None = None, rf: float | pd.Series = DEFAULT_RF,
                 periods: int = TRADING_DAYS):
        self.returns = returns
        self.train_window = train_window
        self.rebalance_every = rebalance_every
        self.tc_bps = tc_bps
        self.min_history = min_history or train_window // 2
        self.rf = rf
        self.periods = periods

    def run(self, weights_fn: Callable[[pd.DataFrame], pd.Series],
            name: str = "strategy") -> pd.DataFrame:
        """Returns a DataFrame with per-period out-of-sample strategy returns
        and a `weights` attribute (DataFrame indexed by rebalance date)."""
        R = self.returns
        oos_rets, weight_rows, weight_dates = [], [], []
        w_prev: pd.Series | None = None

        for start in range(self.train_window, len(R), self.rebalance_every):
            train = R.iloc[start - self.train_window:start]
            # investable filter: enough valid history in the window
            valid = train.columns[train.notna().sum() >= self.min_history]
            train = train[valid].fillna(0.0)
            if train.shape[1] == 0:
                continue
            try:
                w = weights_fn(train)
            except Exception:
                w = equal_weight(train)
            w = w.clip(lower=0)
            w = w / w.sum() if w.sum() > 0 else equal_weight(train)

            test = R.iloc[start:start + self.rebalance_every][w.index].fillna(0.0)
            block = (test * w.values).sum(axis=1)

            if self.tc_bps > 0:
                prev = w_prev.reindex(w.index).fillna(0.0) if w_prev is not None \
                    else pd.Series(0.0, index=w.index)
                turnover = float((w - prev).abs().sum())
                if not block.empty:
                    block.iloc[0] -= turnover * self.tc_bps / 1e4
            w_prev = w

            oos_rets.append(block)
            weight_rows.append(w)
            weight_dates.append(R.index[start])

        rets = pd.concat(oos_rets) if oos_rets else pd.Series(dtype=float)
        rets.name = name
        out = rets.to_frame()
        out.attrs["weights"] = pd.DataFrame(weight_rows, index=weight_dates).fillna(0.0)
        return out

    def run_many(self, strategies: dict[str, Callable[[pd.DataFrame], pd.Series]]
                 ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Run several strategies; returns (per-period returns, summary table)."""
        frames = [self.run(fn, name) for name, fn in strategies.items()]
        for f in frames:
            f.attrs = {}   # DataFrame-valued attrs break pd.concat's attr merge
        rets = pd.concat(frames, axis=1)
        summary = pd.concat(
            [M.summary(rets[c].dropna(), rf=self.rf, periods=self.periods, name=c)
             for c in rets.columns], axis=1)
        return rets, summary
