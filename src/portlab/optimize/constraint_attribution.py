"""Constraint attribution (Axioma-style).

Answers "what is each constraint costing me?" by re-solving the optimization
with each constraint relaxed to its 'off' value and measuring the improvement
in the *true* objective (higher = better). For example::

    "your 5% position cap costs 0.42%/yr of risk-adjusted return;
     the 20% turnover limit costs another 0.11%/yr"

The interface is deliberately generic: ``solve_fn`` is any callable taking
the constraint parameters as keywords and returning weights, so this works
with :func:`portlab.optimize.constrained.opt_constrained`,
:func:`portlab.mpo.multiperiod_weights`, :func:`portlab.taxopt.tax_aware_weights`,
or any user-supplied optimizer.

Leave-one-out deltas do not sum exactly to the total constrained-vs-free gap
(constraints interact); the table reports each marginal cost honestly and the
fully-relaxed optimum as the reference. With a single constraint the delta
*is* the total gap.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["attribute_constraints"]


def attribute_constraints(solve_fn, params: dict, relaxed: dict,
                          objective_fn, annualize: float = 1.0) -> pd.DataFrame:
    """Attribute objective shortfall to individual constraints.

    ``solve_fn``: callable(**params) -> weights (Series or array).
    ``params``: {constraint_name: current value} — the constrained setup.
    ``relaxed``: {constraint_name: 'off' value} — same keys; relaxing a
        constraint must never worsen the true objective.
    ``objective_fn``: weights -> float, the TRUE objective in return units
        (higher = better), e.g. ``w @ mu - 0.5 * gamma * w @ cov @ w``.
    ``annualize``: multiplies deltas (e.g. 252 for daily objectives).

    Returns a DataFrame indexed by constraint with ``objective_base``,
    ``objective_relaxed``, ``delta`` (relaxed − base, ≥ 0) and
    ``delta_annualized``. A final ``TOTAL (all relaxed)`` row gives the
    fully unconstrained reference. Rows for constraints whose relaxed
    solve fails report NaN and the error.
    """
    if set(params) != set(relaxed):
        raise ValueError("params and relaxed must have the same keys")
    try:
        w_base = solve_fn(**params)
    except Exception as e:
        raise ValueError(f"base solve failed: {e}") from e
    obj_base = float(objective_fn(np.asarray(w_base, dtype=float)))

    rows = []
    for name in params:
        trial = dict(params)
        trial[name] = relaxed[name]
        try:
            w_rel = solve_fn(**trial)
            obj_rel = float(objective_fn(np.asarray(w_rel, dtype=float)))
            delta = obj_rel - obj_base
            # relaxing a constraint cannot worsen the true optimum;
            # a negative delta is solver noise — clip with a note
            clipped = delta < -1e-9
            rows.append({"constraint": name, "objective_base": obj_base,
                         "objective_relaxed": obj_rel,
                         "delta": max(delta, 0.0),
                         "delta_annualized": max(delta, 0.0) * annualize,
                         "note": "clipped solver noise" if clipped else ""})
        except Exception as e:  # noqa: BLE001 — one bad constraint
            rows.append({"constraint": name, "objective_base": obj_base,  # noqa
                         "objective_relaxed": float("nan"),
                         "delta": float("nan"), "delta_annualized": float("nan"),
                         "note": f"solve failed: {type(e).__name__}"})
    # fully relaxed reference
    try:
        w_free = solve_fn(**relaxed)
        obj_free = float(objective_fn(np.asarray(w_free, dtype=float)))
        rows.append({"constraint": "TOTAL (all relaxed)",
                     "objective_base": obj_base,
                     "objective_relaxed": obj_free,
                     "delta": max(obj_free - obj_base, 0.0),
                     "delta_annualized": max(obj_free - obj_base, 0.0)
                     * annualize, "note": ""})
    except Exception as e:  # noqa: BLE001
        rows.append({"constraint": "TOTAL (all relaxed)",
                     "objective_base": obj_base,
                     "objective_relaxed": float("nan"), "delta": float("nan"),
                     "delta_annualized": float("nan"),
                     "note": f"solve failed: {type(e).__name__}"})
    return pd.DataFrame(rows).set_index("constraint")
