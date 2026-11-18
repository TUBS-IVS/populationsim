# PopulationSim
# See full license in LICENSE.txt.

import logging

import pandas as pd
import numpy as np

from populationsim.core import bs_run_options, pipeline, inject, config
from populationsim.core.helper import get_weight_table

logger = logging.getLogger(__name__)


def household_lists_for_groups(household_groups, group_ids, household_id_col):
    """Household ids and draw probabilities of the groups in ``group_ids``.

    Fork fast path of ``expand_households`` (setting ``FAST_PATHS``). Upstream
    builds ``[hh_ids, probs]`` for every incidence group by iterating a
    ``groupby`` over the whole ``household_groups`` table, although the expanded
    weights of a run use only a small share of the groups. This builds the same
    two lists for the used groups only and returns them in a dict keyed by group
    id, which the step's chooser indexes exactly as upstream's list.

    The lists are equal element for element and in type to upstream's: the rows
    of a group keep their order in ``household_groups`` (a stable sort by group
    id, as ``groupby`` keeps it), and each list is made by the same pandas
    expressions on a Series of the same dtype and values
    (``list(ids)``, ``list(w / w.sum())``), so the sum and the division are the
    same operations. Building the lists draws no random number; upstream's loop
    does not either, so skipping the unused groups leaves the draws unchanged.

    Parameters
    ----------
    household_groups : pandas.DataFrame
        columns ``household_id_col``, ``group_id``, ``sample_weight``
    group_ids : array-like
        group ids that occur in the expanded weights (any order, repeats allowed).
        The step hands over an object array (the expanded table is built from
        values of mixed dtype); ``np.isin`` of such an array against the integer
        group column compares element by element (slower by orders of magnitude
        than the sorted comparison of two integer arrays), so the used ids are cast to
        the dtype of the ``group_id`` column first. A ``ValueError`` is raised if
        the cast changes a value (a non-integral or non-numeric id).
    household_id_col : str

    Returns
    -------
    dict
        ``{group_id: [hh_ids, probs]}`` for every group id in ``group_ids``
    """
    group_col = household_groups["group_id"].to_numpy()
    used = pd.unique(np.asarray(group_ids))
    try:
        converted = used.astype(group_col.dtype)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            "expand_households: group ids cannot be cast to the dtype %s of the "
            "group_id column (%s)" % (group_col.dtype, error)
        ) from error
    if not np.array_equal(converted, used):
        raise ValueError(
            "expand_households: the cast of the used group ids to the dtype %s of "
            "the group_id column changes a group id value" % group_col.dtype
        )
    used = converted
    rows = np.flatnonzero(np.isin(group_col, used))
    order = rows[np.argsort(group_col[rows], kind="stable")]
    sorted_groups = group_col[order]
    ids = household_groups[household_id_col].iloc[order].reset_index(drop=True)
    weights = household_groups["sample_weight"].iloc[order].reset_index(drop=True)

    lists = {}
    if len(order):
        starts = np.flatnonzero(
            np.concatenate([[True], sorted_groups[1:] != sorted_groups[:-1]])
        )
        ends = np.concatenate([starts[1:], [len(order)]])
        for group_id, start, end in zip(
            sorted_groups[starts].tolist(), starts.tolist(), ends.tolist()
        ):
            w = weights.iloc[start:end]
            lists[group_id] = [list(ids.iloc[start:end]), list(w / w.sum())]

    logger.info(
        "expand_households: household lists built for %s of %s incidence groups"
        " (FAST_PATHS; the others are not drawn from)"
        % (len(lists), len(pd.unique(group_col)))
    )
    return lists


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

        # for each group, lists of hh_ids and their sample_weights (as relative probabiliities)
        # [ [ [<group_0_hh_id_list>], [<group_0_hh_prob_list>] ],
        #   [ [<group_1_hh_id_list>], [<group_1_hh_prob_list>] ], ... ]
        HH_IDS = 0
        HH_PROBS = 1
        if bs_run_options.fast_paths():
            # fork: FAST_PATHS builds the lists only for the groups the expanded
            # weights use, as a dict by group id; the draws below are unchanged
            group_hh_probs = household_lists_for_groups(
                household_groups, expanded_weights.group_id.to_numpy(), household_id_col
            )
        else:
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

        expanded_weights[household_id_col] = expanded_weights.group_id.apply(chooser)

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
