# PopulationSim
# See full license in LICENSE.txt.

import logging

import pandas as pd
import numpy as np

from populationsim.core import bs_run_options, pipeline, inject, config
from populationsim.core.helper import get_weight_table

logger = logging.getLogger(__name__)


def draw_household_ids(group_ids, household_groups, household_id_col, rng):
    """Draw one household id per row of ``group_ids`` in one batch.

    Fork fast path of ``expand_households`` (setting ``FAST_PATHS``). Upstream
    makes one ``rng.choice(hh_ids, p=probs)`` call per expanded household and
    builds ``[hh_ids, probs]`` for every incidence group, although the expanded
    weights of a run use only a small share of the groups. This draws all rows
    at once and works only on the groups that occur, with whole-array numpy
    operations on one stable sort of ``household_groups`` (no pandas object per
    group, which made the first version several times slower than upstream's
    list build).

    Why the draw equals upstream's. ``RandomState.choice(a, p=p)`` with no size
    takes one ``random_sample()`` uniform, builds ``cdf = p.cumsum()``,
    ``cdf /= cdf[-1]`` and returns ``a[cdf.searchsorted(u, side="right")]``. The
    uniforms are consumed in row order, one per row, whatever the group (a group
    of one household takes one too). ``random_sample(n)`` gives the same ``n``
    numbers as ``n`` calls of ``random_sample()`` and leaves the generator in the
    same state; each row's uniform is then mapped through the ``cdf`` of its own
    group by the same ``cumsum``, division and ``searchsorted``. The
    probabilities are ``w / w.sum()`` on a contiguous float or integer array of
    the group's weights in the row order of ``household_groups``, the numpy
    operations that pandas' ``Series.sum`` and division run on the same values,
    so the sum, the division and the cumulative sum are the same operations.
    ``tests/test_bs_fast_paths.py`` compares draws and the next random number
    with upstream's per-row choice for float64, int64, int32 and float32 weights.

    Id dtype. Upstream builds a Python list of ids per group, so ``choice`` turns
    it into the platform's default integer array (int32 on Windows with NumPy
    1.x) and the ``hh_id`` column of the expanded table has that dtype, not the
    dtype of the id column. The result here has the dtype of
    ``np.asarray(list(ids of a group))`` (the common dtype if groups differ),
    derived the same way, so the table is equal in dtype on every platform.

    No rows. An empty ``group_ids`` returns an empty array of the dtype of
    ``group_ids`` (what upstream's ``apply`` on an empty Series gives) and draws
    nothing.

    Differences in failure. Upstream fails inside ``choice`` on probabilities
    that are negative or do not sum to 1, and not at all on NaN; a group whose
    weights are not finite, are negative or sum to 0 raises a ``ValueError`` that
    names the group here, and a group id without households raises a
    ``ValueError`` where upstream raises a ``KeyError`` or ``IndexError``. Every
    check runs before the first number is drawn.

    Parameters
    ----------
    group_ids : array-like
        one group id per expanded household, in the order of the draws. The step
        hands over an object array (the expanded table is built from values of
        mixed dtype); a lookup of such an array against the integer group column
        compares element by element, so the distinct ids are cast to the dtype of
        the ``group_id`` column first. A ``ValueError`` is raised if the cast
        changes a value (a non-integral, missing or non-numeric id).
    household_groups : pandas.DataFrame
        columns ``household_id_col``, ``group_id``, ``sample_weight``
    household_id_col : str
    rng : numpy.random.RandomState
        generator of the step; advanced by ``len(group_ids)`` uniforms

    Returns
    -------
    numpy.ndarray
        household id of each row of ``group_ids``
    """
    group_col = household_groups["group_id"].to_numpy()
    group_ids = np.asarray(group_ids)
    codes, used = pd.factorize(group_ids.astype(object))
    if (codes < 0).any():
        raise ValueError(
            "expand_households: a group id is missing (NaN) in the expanded weights"
        )
    try:
        converted = np.asarray(used).astype(group_col.dtype)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            "expand_households: group ids cannot be cast to the dtype %s of the "
            "group_id column (%s)" % (group_col.dtype, error)
        ) from error
    if not np.array_equal(converted, np.asarray(used)):
        raise ValueError(
            "expand_households: the cast of the used group ids to the dtype %s of "
            "the group_id column changes a group id value" % group_col.dtype
        )

    # household rows of every used group, in their order in household_groups (as
    # groupby keeps it): one stable sort, then a slice of the sorted arrays per
    # group (a contiguous copy of the group's values, like upstream's group Series)
    order = np.argsort(group_col, kind="stable")
    sorted_groups = group_col[order]
    lower = np.searchsorted(sorted_groups, converted, side="left")
    upper = np.searchsorted(sorted_groups, converted, side="right")
    sorted_weights = household_groups["sample_weight"].to_numpy()[order]
    ids = household_groups[household_id_col].to_numpy()

    # Whole-array validity per used group (exact counts, no summation): no
    # household, a non-finite or negative weight, or only zero weights.
    def _count_in_groups(flags):
        cumulative = np.concatenate([[0], np.cumsum(flags, dtype=np.int64)])
        return cumulative[upper] - cumulative[lower]

    missing = upper == lower
    nonfinite = _count_in_groups(~np.isfinite(sorted_weights)) > 0
    negative = _count_in_groups(sorted_weights < 0) > 0
    all_zero = _count_in_groups(sorted_weights > 0) == 0
    invalid = nonfinite | negative | all_zero

    # Groups are checked in the order of their first row, so the group named in
    # an error is the one upstream's per-row choice reaches first and fails on.
    # A total that overflows to inf is only seen with the sum, in the same loop.
    cdfs = []
    for index, (group_id, lo, hi) in enumerate(
        zip(converted.tolist(), lower.tolist(), upper.tolist())
    ):
        if missing[index]:
            raise ValueError(
                "expand_households: group %s has no households in household_groups"
                % group_id
            )
        w = sorted_weights[lo:hi]
        total = w.sum()
        if invalid[index] or not np.isfinite(total):
            raise ValueError(
                "expand_households: invalid sample weights in household group %s"
                " (non-finite, negative or summing to 0)" % group_id
            )
        # upstream: list(w / w.sum()) handed to choice as a float64 array
        cdf = np.asarray(w / total, dtype=np.float64).cumsum()
        cdf /= cdf[-1]
        cdfs.append(cdf)

    # dtype of the result: that of upstream's np.asarray(list of the group's ids)
    # (the common dtype of the used groups); with no row it is the dtype of the
    # group id Series, as upstream's apply on an empty Series returns
    if len(codes) == 0:
        result_dtype = group_ids.dtype
        id_dtype_path = "empty"
    else:
        result_dtype, id_dtype_path = _list_id_dtype(
            ids, order, lower, upper, household_groups, household_id_col
        )

    uniforms = rng.random_sample(len(codes))
    by_group = np.argsort(codes, kind="stable")
    bounds = np.concatenate([[0], np.cumsum(np.bincount(codes, minlength=len(used)))])
    sorted_uniforms = uniforms[by_group]
    donor_rows = np.empty(len(codes), dtype=np.int64)
    for index, cdf in enumerate(cdfs):
        start, stop = bounds[index], bounds[index + 1]
        donor_rows[start:stop] = lower[index] + np.searchsorted(
            cdf, sorted_uniforms[start:stop], side="right"
        )
    chosen = np.empty(len(codes), dtype=result_dtype)
    chosen[by_group] = ids[order[donor_rows]]

    logger.info(
        "expand_households: %s households drawn in one batch from %s of %s incidence"
        " groups (FAST_PATHS; the others are not drawn from; id dtype %s from %s)"
        % (
            len(codes),
            len(cdfs),
            len(np.flatnonzero(np.diff(sorted_groups)) + 1) + 1 if len(order) else 0,
            result_dtype,
            id_dtype_path,
        )
    )
    return chosen


def _list_id_dtype(ids, order, lower, upper, household_groups, household_id_col):
    """Dtype of ``np.asarray(list(ids of a group))``, common over the used groups.

    Returns ``(dtype, how)``. For an integer id column whose ids in the used
    groups fit int64 this is one comparison of the smallest and largest id with
    the default integer (``np.asarray([0]).dtype``: int32 on Windows with NumPy
    1.x, int64 elsewhere): a group list of Python ints is that dtype if all its
    ids fit it and int64 otherwise, and the common dtype of such groups is the
    wider one. Any other id column (floats, strings, objects, unsigned ids beyond
    int64) takes the per-group list, as upstream builds it, and is counted as
    ``"group lists"`` in the log line of the draw.
    """
    if ids.dtype.kind in "iu":
        counts = upper - lower
        group_rows = order[
            np.repeat(lower - np.cumsum(counts) + counts, counts)
            + np.arange(counts.sum())
        ]
        used_ids = ids[group_rows]
        int64 = np.iinfo(np.int64)
        if used_ids.min() >= int64.min and used_ids.max() <= int64.max:
            default = np.asarray([0]).dtype
            info = np.iinfo(default)
            fits = used_ids.min() >= info.min and used_ids.max() <= info.max
            return (default if fits else np.dtype(np.int64)), "one range check"
    dtypes = []
    id_series = household_groups[household_id_col]
    for lo, hi in zip(lower.tolist(), upper.tolist()):
        dtypes.append(np.asarray(list(id_series.iloc[order[lo:hi]])).dtype)
    return np.result_type(*dtypes), "group lists"


@inject.step()
def expand_households():
    """
    Create a complete expanded synthetic household list with their assigned geographic zone ids.

    This is the skeleton synthetic household id list with no household or person attributes,
    one row per household, with geography columns and seed household table household_id.

    Creates pipeline table expanded_household_ids
    """

    if config.setting("NO_INTEGERIZATION_EVER", False):
        logger.warning("skipping expand_households: NO_INTEGERIZATION_EVER")
        inject.add_table("expanded_household_ids", pd.DataFrame())
        return

    geographies = config.setting("geographies")
    household_id_col = config.setting("household_id_col")

    low_geography = geographies[-1]

    # only one we really need is low_geography
    seed_geography = config.setting("seed_geography")
    geography_cols = geographies[geographies.index(seed_geography) :]

    weights = get_weight_table(low_geography, sparse=True)
    weights = weights[geography_cols + [household_id_col, "integer_weight"]]

    # - expand weights table by integer_weight, so there is one row per desired hh
    weight_cols = weights.columns.values
    weights_np = np.repeat(weights.values, weights.integer_weight.values, axis=0)
    expanded_weights = pd.DataFrame(data=weights_np, columns=weight_cols)

    if config.setting("GROUP_BY_INCIDENCE_SIGNATURE"):

        # get these in a repeatable order so np.random.choice behaves the same regardless of weight table order
        # i.e. which could vary depending on whether we ran single or multi process due to apportioned/coalesce
        expanded_weights = expanded_weights.sort_values(
            geography_cols + [household_id_col]
        )

        # the household_id_col is really the group_id
        expanded_weights.rename(columns={household_id_col: "group_id"}, inplace=True)

        # the original incidence table with one row per hh, with index hh_id
        household_groups = pipeline.get_table("household_groups")
        household_groups = household_groups[
            [household_id_col, "group_id", "sample_weight"]
        ]

        if bs_run_options.fast_paths():
            # fork: FAST_PATHS draws all households in one batch, from the same
            # random numbers in the same order as the per-row choice below
            prng = pipeline.get_rn_generator().get_external_rng("expand_households")
            expanded_weights[household_id_col] = draw_household_ids(
                expanded_weights.group_id.to_numpy(),
                household_groups,
                household_id_col,
                prng,
            )
        else:
            # for each group, lists of hh_ids and their sample_weights (as relative probabiliities)
            # [ [ [<group_0_hh_id_list>], [<group_0_hh_prob_list>] ],
            #   [ [<group_1_hh_id_list>], [<group_1_hh_prob_list>] ], ... ]
            HH_IDS = 0
            HH_PROBS = 1
            grouper = household_groups.groupby("group_id")
            group_hh_probs = [0] * len(grouper)
            for group_id, df in grouper:
                hh_ids = list(df[household_id_col])
                probs = list(df.sample_weight / df.sample_weight.sum())
                group_hh_probs[group_id] = [hh_ids, probs]

            # get a repeatable random number sequence generator for consistent choice results
            prng = pipeline.get_rn_generator().get_external_rng("expand_households")

            # now make a hh_id choice for each group_id in expanded_weights
            def chooser(group_id):
                hh_ids = group_hh_probs[group_id][HH_IDS]
                hh_probs = group_hh_probs[group_id][HH_PROBS]
                return prng.choice(hh_ids, p=hh_probs)

            expanded_weights[household_id_col] = expanded_weights.group_id.apply(
                chooser
            )

        # FIXME - omit in production?
        del expanded_weights["group_id"]
        del expanded_weights["integer_weight"]

    append = inject.get_step_arg("append", False)
    replace = inject.get_step_arg("replace", False)
    assert not (
        append and replace
    ), "can't specify both append and replace for expand_households"

    if append or replace:
        t = inject.get_table("expanded_household_ids").to_frame()
        prev_hhs = len(t.index)
        added_hhs = len(expanded_weights.index)

        if replace:
            # FIXME - should really get from crosswalk table?
            low_ids_to_replace = expanded_weights[low_geography].unique()
            t = t[~t[low_geography].isin(low_ids_to_replace)]

        expanded_weights = pd.concat([t, expanded_weights], ignore_index=True)

        dropped_hhs = prev_hhs - len(t.index)
        final_hhs = len(expanded_weights.index)
        op = "append" if append else "replace"
        logger.info(
            "expand_households op: %s prev hh count %s dropped %s added %s final %s"
            % (op, prev_hhs, dropped_hhs, added_hhs, final_hhs)
        )

    # sort this so results will be consistent whether single or multiprocessing, GROUP_BY_INCIDENCE_SIGNATURE, etc...
    expanded_weights = expanded_weights.sort_values(geography_cols + [household_id_col])

    repop = inject.get_step_arg("repop", default=False)
    inject.add_table("expanded_household_ids", expanded_weights, replace=repop)
