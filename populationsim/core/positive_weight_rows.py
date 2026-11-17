# PopulationSim
# See full license in LICENSE.txt.
"""Positive-row storage of the sub-geography weight tables.

Used only with the fork setting ``STORE_POSITIVE_WEIGHT_ROWS_ONLY``
(:mod:`populationsim.core.bs_run_options`). Upstream builds one row per seed
household (group) for every sub zone, almost all of them with zero weight. With
the setting on, the integerizer wrappers keep per zone only the rows whose
``balanced_weight`` or ``integer_weight`` is positive, and the next geography's
``sub_balancing`` rebuilds the full-length parent weight vector with zeros.

Why the synthetic population does not change:

* ``expand_households`` reads only the ``_sparse`` table, which is
  ``dense[integer_weight > 0]``. The kept rows are a superset of those rows in
  the same order with the same index labels and dtypes, so the ``_sparse``
  table is the same frame.
* The balancers and integerizers drop zero-weight rows on entry; the rebuilt
  parent vector has the dense path's values, order, index and dtype (proof in
  ``tests/test_bs_positive_weight_rows.py``).

The seed-geography weight table is never filtered: several steps add columns to
it by position.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

WEIGHT_COLUMNS = ("balanced_weight", "integer_weight")

# (kept, total) rows per zone id since the last reset (one sub_balancing step).
# Keyed by zone: when do_simul_integerizing retries, a zone's sequential frame is
# replaced by its simultaneous one, and the later count replaces the earlier.
_TALLY = {}


def keep_positive_rows(zone_weights_df, zone_id):
    """Return the rows of one zone's canonical weight frame with a positive weight.

    A row is kept if any of ``balanced_weight`` and ``integer_weight`` (the ones
    present) is positive. Index labels, column order and dtypes are those of the
    input. A NaN weight raises before any row is dropped. The kept and total row
    counts of ``zone_id`` go into the tally that :func:`log_row_tally` reports;
    nothing is logged here (this runs once per zone).
    """
    columns = [c for c in WEIGHT_COLUMNS if c in zone_weights_df.columns]
    if not columns:
        raise ValueError(
            "weight frame has none of the columns %s" % (list(WEIGHT_COLUMNS),)
        )
    keep = np.zeros(len(zone_weights_df.index), dtype=bool)
    for column in columns:
        values = zone_weights_df[column].to_numpy()
        nan_count = int(pd.isna(values).sum())
        if nan_count:
            raise ValueError(
                "zone %s has %d NaN weights in column %s; refusing to drop rows"
                % (zone_id, nan_count, column)
            )
        keep |= values > 0
    _TALLY[zone_id] = (int(keep.sum()), len(keep))
    return zone_weights_df[keep]


def concat_kept(frames):
    """Concatenate filtered frames, leaving out the empty ones.

    If every frame is empty, the first one is returned, so the result keeps the
    columns and dtypes of the dense path (an all-zero zone or parent changes
    neither).
    """
    non_empty = [f for f in frames if len(f.index) > 0]
    if not non_empty:
        return frames[0].copy()
    return pd.concat(non_empty)


def rebuild_parent_weights(parent_rows, household_id_col, column, order):
    """Full-length parent weight vector from the stored positive rows.

    Parameters
    ----------
    parent_rows : pandas.DataFrame
        the stored rows of one parent zone (positive rows only)
    household_id_col : str
        household (or group) id column; becomes the index name
    column : str
        weight column to return
    order : pandas.Index
        the row order the dense table had for this parent: the seed incidence
        index of the parent's seed zone

    Returns
    -------
    pandas.Series
        ``column`` indexed by ``order`` (named ``household_id_col``), zero
        where no row was stored; the dtype is the stored column's dtype, as in
        the dense path.
    """
    weights = parent_rows.set_index(household_id_col)[column]
    if not weights.index.is_unique:
        raise ValueError(
            "parent weight rows of %s are not unique by %s" % (column, household_id_col)
        )
    order = pd.Index(order, name=household_id_col)
    # same check as upstream's length assertion: rows of another seed zone
    assert weights.index.isin(order).all(), (
        "seed table and initial weights table do not match, "
        "possibly due to overlapping zones in crosswalk."
    )
    return weights.reindex(order, fill_value=0)


def reset_row_tally():
    """Start counting kept and total rows for one geography."""
    _TALLY.clear()


def log_row_tally(geography):
    """Log the kept and total rows of ``<geography>_weights`` once (INFO).

    ``total`` is the row count the dense table would have had.
    """
    kept = sum(k for k, _ in _TALLY.values())
    total = sum(t for _, t in _TALLY.values())
    share = 100.0 * kept / total if total else 0.0
    logger.info(
        "[positive weight rows] %s_weights: kept %d of %d rows (%.1f %%)"
        % (geography, kept, total, share)
    )
    return kept, total
