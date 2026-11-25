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
* A parent zone that ``sub_balancing`` balances but that has no stored row (all
  its weights are zero) is refused by name: with the setting off its all-zero
  parent vector fails inside the balancer, so no run with the setting off
  completes past it either.

Which rows the dense table would hold is known without it
(:func:`dense_zone_keys`): one row per seed household of the zone's seed zone,
for every (seed zone, parent zone, zone) of the crosswalk whose zone and parent
zone have controls. ``sub_balancing`` checks this count against the rows the
integerizers built, ``summarize`` uses it to fill zeros only where the dense
table had a row, and the multiprocess coalescing step uses it for the rate line
of the coalesced table.

The seed-geography weight table is never filtered: several steps add columns to
it by position.
"""

from __future__ import annotations

import contextlib
import logging

import numpy as np
import pandas as pd

from populationsim.core import inject

logger = logging.getLogger(__name__)

WEIGHT_COLUMNS = ("balanced_weight", "integer_weight")

# (kept, total) rows per (geography, parent zone id, zone id) since the last reset
# (one sub_balancing step). When do_simul_integerizing retries, a zone's
# sequential frame is replaced by its simultaneous one, and the later count
# replaces the earlier; the same zone id under two parents counts twice, as the
# dense table holds it twice.
_TALLY = {}

# (geography, parent zone id) of the parent being integerized; see tally_scope
_SCOPE = (None, None)


@contextlib.contextmanager
def tally_scope(geography, parent_id):
    """Count the zone frames built inside the block under ``(geography, parent_id)``."""
    global _SCOPE
    previous = _SCOPE
    _SCOPE = (geography, parent_id)
    try:
        yield
    finally:
        _SCOPE = previous


def keep_positive_rows(zone_weights_df, zone_id):
    """Return the rows of one zone's canonical weight frame with a positive weight.

    A row is kept if any of ``balanced_weight`` and ``integer_weight`` (the ones
    present) is positive. Index labels, column order and dtypes are those of the
    input. A NaN or negative weight raises before any row is dropped (a negative
    weight is no zero weight: dropping it would hide it). The kept and total row
    counts of ``zone_id`` go into the tally that :func:`log_row_tally` reports,
    keyed by the current :func:`tally_scope`; nothing is logged here (this runs
    once per zone).
    """
    columns = [c for c in WEIGHT_COLUMNS if c in zone_weights_df.columns]
    if not columns:
        raise ValueError(
            "weight frame has none of the columns %s" % (list(WEIGHT_COLUMNS),)
        )
    keep = np.zeros(len(zone_weights_df.index), dtype=bool)
    for column in columns:
        values = zone_weights_df[column].to_numpy()
        _check_weights(
            zone_id,
            column,
            int(pd.isna(values).sum()),
            int((values < 0).sum()),
        )
        keep |= values > 0
    _TALLY[_SCOPE + (zone_id,)] = (int(keep.sum()), len(keep))
    return zone_weights_df[keep]


def _check_weights(zone_id, column, nan_count, negative_count):
    """Raise the ``ValueError`` for a NaN or negative weight (NaN first)."""
    if nan_count:
        raise ValueError(
            "zone %s has %d NaN weights in column %s; refusing to drop rows"
            % (zone_id, nan_count, column)
        )
    if negative_count:
        raise ValueError(
            "zone %s has %d negative weights in column %s; refusing to drop rows"
            % (zone_id, negative_count, column)
        )


def positive_row_mask(zone_ids, weights_by_column):
    """:func:`keep_positive_rows` for several zones at once, on weight arrays.

    Parameters
    ----------
    zone_ids : sequence
        the zone ids, in the order the zones are stored
    weights_by_column : dict
        for each of :data:`WEIGHT_COLUMNS` to check, in that order, a 2-D array
        with one row per zone and one column per household

    Returns
    -------
    numpy.ndarray
        boolean array of the same shape: the rows
        :func:`keep_positive_rows` would keep.

    The result and the side effects are those of calling :func:`keep_positive_rows`
    zone by zone: the kept and total counts of every zone go into the tally, and
    the first offending zone in zone order raises the same ``ValueError`` (checking
    the columns in order, NaN before negative), after the zones before it have
    been counted.
    """
    keep = None
    nan_counts = {}
    negative_counts = {}
    for column, values in weights_by_column.items():
        nan_counts[column] = pd.isna(values).sum(axis=1)
        negative_counts[column] = (values < 0).sum(axis=1)
        positive = values > 0
        keep = positive if keep is None else keep | positive
    if keep is None:
        raise ValueError(
            "weight frame has none of the columns %s" % (list(WEIGHT_COLUMNS),)
        )
    kept_counts = keep.sum(axis=1)
    household_count = keep.shape[1]
    invalid = np.zeros(len(zone_ids), dtype=bool)
    for column in weights_by_column:
        invalid |= (nan_counts[column] > 0) | (negative_counts[column] > 0)
    first_invalid = int(invalid.argmax()) if invalid.any() else len(zone_ids)
    for position in range(first_invalid):
        _TALLY[_SCOPE + (zone_ids[position],)] = (
            int(kept_counts[position]),
            household_count,
        )
    if first_invalid < len(zone_ids):
        for column in weights_by_column:
            _check_weights(
                zone_ids[first_invalid],
                column,
                int(nan_counts[column][first_invalid]),
                int(negative_counts[column][first_invalid]),
            )
    return keep


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


def rebuild_parent_weights(
    parent_rows,
    household_id_col,
    column,
    order,
    geography=None,
    parent_geography=None,
    parent_id=None,
):
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
    geography, parent_geography, parent_id
        the geography being balanced and the parent zone, for the error message

    Returns
    -------
    pandas.Series
        ``column`` indexed by ``order`` (named ``household_id_col``), zero
        where no row was stored; the dtype is the stored column's dtype, as in
        the dense path.

    Raises
    ------
    RuntimeError
        if the parent has no stored row at all. ``sub_balancing`` balances every
        parent zone with controls; with the setting off such a parent's rows are
        all zero and the balancer fails on them (it has no positive row to
        balance), and otherwise the stored table does not match the parent ids
        (for example a key or dtype mismatch). An all-zero vector would hide
        either.
    ValueError
        if the stored rows are not unique by ``household_id_col``, or a stored
        household is not in ``order`` (it does not belong to the parent's seed
        zone, possibly overlapping zones in the crosswalk). A ``ValueError``
        and not an assertion, so that it also holds under ``python -O``.
    """
    if len(parent_rows.index) == 0:
        raise RuntimeError(
            "[positive weight rows] %s: parent %s %s has no stored row in %s_weights "
            "(no positive balanced or integer weight). sub_balancing balances every "
            "parent zone with controls; with the setting off its all-zero parent "
            "weights fail in the balancer, so this is refused instead of balancing "
            "against zeros."
            % (geography, parent_geography, parent_id, parent_geography)
        )
    weights = parent_rows.set_index(household_id_col)[column]
    if not weights.index.is_unique:
        raise ValueError(
            "parent weight rows of %s are not unique by %s" % (column, household_id_col)
        )
    order = pd.Index(order, name=household_id_col)
    # every stored household must belong to the parent's seed zone; this takes the
    # place of upstream's length assertion, which a positive-row slice cannot meet
    # (the message is upstream's); a ValueError, not an assert, so that the
    # check also holds under ``python -O``
    if not weights.index.isin(order).all():
        raise ValueError(
            "seed table and initial weights table do not match, "
            "possibly due to overlapping zones in crosswalk."
        )
    return weights.reindex(order, fill_value=0)


def dense_zone_keys(
    crosswalk_df, geography, parent_geography, seed_geography, zone_ids, parent_ids
):
    """The (seed zone, parent zone, zone) triples the dense weight table holds.

    ``sub_balancing`` balances, per seed zone of the crosswalk, every parent zone
    with controls (``parent_ids``) and in it every zone of the crosswalk with
    controls (``zone_ids``; the control tables hold only zones with households).
    The dense table has one row per seed household for each such triple.

    Returns
    -------
    pandas.DataFrame
        distinct rows with the columns ``seed_geography``, ``parent_geography`` and
        ``geography`` (fewer if they coincide), in crosswalk order
    """
    columns = list(dict.fromkeys([seed_geography, parent_geography, geography]))
    keys = crosswalk_df[columns].drop_duplicates()
    keep = keys[geography].isin(zone_ids) & keys[parent_geography].isin(parent_ids)
    return keys[keep]


def dense_row_count(keys, incidence_df, seed_geography):
    """Rows of the dense table of ``keys`` (:func:`dense_zone_keys`)."""
    households = incidence_df[seed_geography].value_counts()
    return int(keys[seed_geography].map(households).fillna(0).sum())


def reset_row_tally():
    """Start counting kept and total rows for one geography."""
    _TALLY.clear()


def _is_slice_process():
    """True in a multiprocess sub-process that writes its own pipeline (a slice)."""
    return inject.get_injectable("pipeline_file_prefix", None) is not None


def log_row_tally(geography, stored_rows=None, dense_rows=None):
    """Log the kept and total rows of ``<geography>_weights`` once (INFO).

    ``total`` is the row count the dense table would have had. ``stored_rows``
    (the length of the stored table) and ``dense_rows`` (the count of
    :func:`dense_row_count`) are checked against the tally when given: a
    difference means the tally or the dense-table premise is wrong, and raises.

    In a multiprocess sub-process the counts are those of its slice and the line
    says so; :func:`log_coalesced_table` reports the coalesced table.
    """
    kept = sum(k for k, _ in _TALLY.values())
    total = sum(t for _, t in _TALLY.values())
    if stored_rows is not None and stored_rows != kept:
        raise RuntimeError(
            "[positive weight rows] %s_weights: %d rows stored, but the zone frames "
            "kept %d" % (geography, stored_rows, kept)
        )
    if dense_rows is not None and dense_rows != total:
        raise RuntimeError(
            "[positive weight rows] %s_weights: the zone frames had %d rows, but the "
            "crosswalk and controls give %d dense rows" % (geography, total, dense_rows)
        )
    share = 100.0 * kept / total if total else 0.0
    label = "%s_weights" % geography
    if _is_slice_process():
        label += " (slice of this sub-process)"
    logger.info(
        "[positive weight rows] %s: kept %d of %d rows (%.1f %%)"
        % (label, kept, total, share)
    )
    return kept, total


def log_coalesced_table(table_name, df, sub_process_count):
    """Log the rate line of a coalesced multiprocess weight table (INFO).

    Called by the multiprocess coalescing step for each coalesced table, with the
    coalesced pipeline open. Only with ``STORE_POSITIVE_WEIGHT_ROWS_ONLY`` and only
    for a sub-geography weight table: kept rows are the coalesced table's length,
    the total is the dense row count (:func:`dense_row_count`) from the coalesced
    crosswalk, the control tables and the incidence table.
    """
    from populationsim.core import config, pipeline
    from populationsim.core.bs_run_options import store_positive_weight_rows_only
    from populationsim.core.helper import get_control_table, weight_table_name

    if not store_positive_weight_rows_only():
        return None
    geographies = config.setting("geographies")
    seed_geography = config.setting("seed_geography")
    sub_geographies = geographies[geographies.index(seed_geography) + 1 :]
    names = {weight_table_name(g): g for g in sub_geographies}
    if table_name not in names:
        return None
    geography = names[table_name]
    parent_geography = geographies[geographies.index(geography) - 1]
    keys = dense_zone_keys(
        pipeline.get_table("crosswalk"),
        geography,
        parent_geography,
        seed_geography,
        get_control_table(geography).index,
        get_control_table(parent_geography).index,
    )
    total = dense_row_count(keys, pipeline.get_table("incidence_table"), seed_geography)
    kept = len(df.index)
    share = 100.0 * kept / total if total else 0.0
    logger.info(
        "[positive weight rows] %s (coalesced from %d sub-processes): "
        "kept %d of %d rows (%.1f %%)"
        % (table_name, sub_process_count, kept, total, share)
    )
    return kept, total
