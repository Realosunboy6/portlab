"""Performance attribution: Brinson-Hood-Beebower (1986).

Decomposes a portfolio's excess return over a benchmark into *why* it
happened — allocation bets vs. stock selection — per segment (sector,
asset class, region, ...). For segment *i* with portfolio weight ``w_p``,
benchmark weight ``w_b``, portfolio return ``r_p`` and benchmark return
``r_b``::

    allocation_i  = (w_p,i - w_b,i) * r_b,i
    selection_i   = w_b,i * (r_p,i - r_b,i)
    interaction_i = (w_p,i - w_b,i) * (r_p,i - r_b,i)

The identity ``sum(allocation + selection + interaction) == R_p - R_b`` holds
algebraically whenever both weight vectors sum to 1; :func:`brinson` checks
it and raises if it fails to close (a non-closing identity means an input
bug, never a rounding choice).

``method="fachler"`` switches allocation to the Brinson-Fachler variant,
``(w_p,i - w_b,i) * (r_b,i - R_b)``, which scores an overweight only when the
segment beat the overall benchmark. The identity is unaffected.

Reference: Brinson, Hood & Beebower (1986), "Determinants of Portfolio
Performance," *Financial Analysts Journal* 42(4), 39-44.
"""

from __future__ import annotations

import pandas as pd

__all__ = ["brinson"]

_METHODS = ("bhb", "fachler")
_IDENTITY_TOL = 1e-9


def _as_frame(segments) -> pd.DataFrame:
    if isinstance(segments, pd.DataFrame):
        df = segments.copy()
    else:
        df = pd.DataFrame(list(segments))
    need = {"segment", "w_p", "w_b", "r_p", "r_b"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"segments need columns {sorted(need)}; missing {sorted(missing)}")
    return df


def brinson(segments, method: str = "bhb") -> dict:
    """Single-period Brinson attribution.

    ``segments``: DataFrame (or list of dicts) with columns
    segment/w_p/w_b/r_p/r_b. Weights must each sum to 1 (within 1e-8);
    all values must be finite.

    Returns dict with ``table`` (per-segment allocation/selection/
    interaction/total), ``portfolio_return``, ``benchmark_return``,
    ``excess_return``, and the three effect totals.
    """
    method = str(method).lower()
    if method not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS}, got {method!r}")
    df = _as_frame(segments)
    if df.empty:
        raise ValueError("brinson() needs at least one segment")
    for col in ("w_p", "w_b", "r_p", "r_b"):
        vals = pd.to_numeric(df[col], errors="raise").to_numpy(dtype=float)
        if not all(pd.notna(vals)) or not all(abs(v) != float("inf") for v in vals):
            raise ValueError(f"column {col!r} must be finite")
        df[col] = vals
    for col, name in (("w_p", "portfolio"), ("w_b", "benchmark")):
        s = float(df[col].sum())
        if abs(s - 1.0) > 1e-8:
            raise ValueError(f"{name} weights sum to {s}, not 1")

    w_p = df["w_p"].to_numpy()
    w_b = df["w_b"].to_numpy()
    r_p = df["r_p"].to_numpy()
    r_b = df["r_b"].to_numpy()

    r_p_total = float((w_p * r_p).sum())
    r_b_total = float((w_b * r_b).sum())
    excess = r_p_total - r_b_total

    active_w = w_p - w_b
    active_r = r_p - r_b
    if method == "fachler":
        allocation = active_w * (r_b - r_b_total)
    else:
        allocation = active_w * r_b
    selection = w_b * active_r
    interaction = active_w * active_r

    a, s, i = float(allocation.sum()), float(selection.sum()), float(interaction.sum())
    if abs((a + s + i) - excess) > _IDENTITY_TOL:
        raise ValueError(
            f"attribution identity failed to close: {a + s + i} != {excess} "
            "(check the weight/return inputs)")

    table = pd.DataFrame({
        "segment": df["segment"].astype(str).to_numpy(),
        "allocation": allocation,
        "selection": selection,
        "interaction": interaction,
    })
    table["total"] = table[["allocation", "selection", "interaction"]].sum(axis=1)
    return {"table": table.set_index("segment"),
            "portfolio_return": r_p_total,
            "benchmark_return": r_b_total,
            "excess_return": excess,
            "allocation": a, "selection": s, "interaction": i,
            "method": method}
