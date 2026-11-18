# PopulationSim
# See full license in LICENSE.txt.

import logging
import os
import pandas as pd
import numpy as np

from populationsim.core import inject, config

from populationsim.core import positive_weight_rows
from populationsim.core.bs_run_options import store_positive_weight_rows_only
from populationsim.core.helper import get_control_table, get_weight_table

logger = logging.getLogger(__name__)

AS_CSV = False


class _DenseLayout:
    """Fork: which rows the dense ``<geography>_weights`` table would hold.

    With ``STORE_POSITIVE_WEIGHT_ROWS_ONLY`` a sub-geography weight table holds
    only its positive rows. Zeros are filled in only where the dense table had a
    row: the (seed, parent, zone) keys of
    :func:`populationsim.core.positive_weight_rows.dense_zone_keys`, and the
    households of the seed zones among them. A household or zone the dense
    table did not hold (never balanced at this geography) stays missing, as with
    the setting off. Stored rows outside the dense keys raise.
    """

    def __init__(self, geography, weights_df, crosswalk_df, incidence_df):
        geographies = config.setting("geographies")
        self.seed_geography = config.setting("seed_geography")
        self.geography = geography
        self.parent_geography = geographies[geographies.index(geography) - 1]
        self.keys = positive_weight_rows.dense_zone_keys(
            crosswalk_df,
            geography,
            self.parent_geography,
            self.seed_geography,
            get_control_table(geography).index,
            get_control_table(self.parent_geography).index,
        )
        self.crosswalk_df = crosswalk_df
        columns = list(self.keys.columns)
        stored = pd.MultiIndex.from_frame(weights_df[columns].drop_duplicates())
        outside = ~stored.isin(pd.MultiIndex.from_frame(self.keys))
        if outside.any():
            raise RuntimeError(
                "[positive weight rows] %s_weights: %d stored %s keys are not in the "
                "dense table of the crosswalk and controls, e.g. %s"
                % (geography, outside.sum(), tuple(columns), stored[outside][0])
            )
        seeds = self.keys[self.seed_geography].unique()
        self.households = incidence_df.index[
            incidence_df[self.seed_geography].isin(seeds)
        ]

    def zone_ids(self, target):
        """Ids of ``target`` the dense table's rows carry (``target`` is this
        geography or one above it; sub_balancing writes the ids above the parent
        as the crosswalk maximum per parent zone)."""
        if target == self.geography:
            values = self.keys[self.geography]
        elif target == self.parent_geography:
            values = self.keys[self.parent_geography]
        else:
            per_parent = self.crosswalk_df.groupby(self.parent_geography)[target].max()
            values = self.keys[self.parent_geography].map(per_parent)
        return pd.unique(values)


def _zero_filled_by_household(hh_weights, households):
    """Fork: one row per household of ``households`` (sorted, as ``groupby`` gives
    them), zero where none was stored; ``households`` are those the dense table
    held (:class:`_DenseLayout`)."""
    full = pd.Index(np.sort(np.asarray(households)), name=hh_weights.index.name)
    return hh_weights.reindex(full, fill_value=0)


def _zero_filled_aggregate(aggregate, incidence_df, layout, hh_id_col):
    """Fork: the seed-level aggregate with one row per (seed zone, household) the
    dense table held (sorted by seed zone and household, as ``groupby`` gives
    them; zeros where no row was stored).

    ``aggregate`` is ``weights_df.groupby([seed, hh], as_index=False).sum()`` of a
    positive-row table.
    """
    seed_geography = layout.seed_geography
    keys = pd.DataFrame(
        {
            seed_geography: incidence_df.loc[
                layout.households, seed_geography
            ].to_numpy(),
            hh_id_col: layout.households.to_numpy(),
        }
    ).sort_values([seed_geography, hh_id_col])
    full = pd.MultiIndex.from_frame(keys)
    return (
        aggregate.set_index([seed_geography, hh_id_col])
        .reindex(full, fill_value=0)
        .reset_index()
    )


def _zero_filled_zones(summary_df, zone_ids, dense_zone_ids):
    """Fork: one summary row per zone of ``zone_ids`` the dense table held (sorted,
    as ``groupby`` gives them), zero where no row was stored.

    If the dense table held none of them, ``summary_df`` is returned unchanged
    (the step's own all-empty case).
    """
    zones = np.intersect1d(np.asarray(zone_ids), np.asarray(dense_zone_ids))
    if len(zones) == 0:
        return summary_df
    full = pd.Index(zones.astype(summary_df.index.dtype), name=summary_df.index.name)
    return summary_df.reindex(full, fill_value=0)


def out_table(table_name, df):

    table_name = "summary_%s" % table_name

    if AS_CSV:
        file_name = "%s.csv" % table_name
        output_dir = inject.get_injectable("output_dir")
        file_path = os.path.join(output_dir, file_name)
        logger.info("writing output file %s" % file_path)
        write_index = df.index.name is not None
        df.to_csv(file_path, index=write_index)
    else:
        logger.info("saving summary table %s" % table_name)
        repop = inject.get_step_arg("repop", default=False)
        inject.add_table(table_name, df, replace=repop)


def summarize_geography(
    geography,
    weight_col,
    hh_id_col,
    crosswalk_df,
    results_df,
    incidence_df,
    layout=None,
):
    """Summarize ``results_df`` (a sub-geography weight table) by ``geography``.

    Fork: ``layout`` (:class:`_DenseLayout`, only with
    ``STORE_POSITIVE_WEIGHT_ROWS_ONLY``) fills the zones the dense table held.
    """

    # controls_table for current geography level
    controls_table = get_control_table(geography)
    control_names = controls_table.columns.tolist()

    # only want zones from crosswalk for which non-zero control rows exist
    zone_ids = crosswalk_df[geography].unique()
    zone_ids = controls_table.index.intersection(zone_ids).astype(np.int64)

    # Using numpy matrix multiplication for efficient aggregation
    zone_results_df = results_df.loc[
        results_df[geography].isin(zone_ids), [geography, hh_id_col, weight_col]
    ]

    # get the weights and incidence for the zones
    geo_vec = zone_results_df[geography]
    weights = zone_results_df[weight_col].to_numpy()
    incidence = incidence_df.loc[zone_results_df[hh_id_col], control_names]

    # multiply incidence by weights and append the geography vector
    results = np.column_stack(
        [np.transpose(np.transpose(incidence) * weights), geo_vec]
    )

    # If empty, fill with zeros
    if results.size == 0:
        results = np.zeros((1, results.shape[1]))

    logger.info("summarizing %s" % geography)
    controls = [controls_table.loc[x].tolist() for x in zone_ids]

    summary_df = (
        pd.DataFrame(
            data=results, columns=["%s_result" % c for c in control_names] + [geography]
        )
        .groupby(geography)
        .sum()
    )
    if layout is not None:
        summary_df = _zero_filled_zones(
            summary_df, zone_ids, layout.zone_ids(geography)
        )

    controls_df = pd.DataFrame(
        data=np.asanyarray(controls),
        columns=["%s_control" % c for c in control_names],
        index=zone_ids,
    )

    dif_df = pd.DataFrame(
        data=np.array(summary_df) - np.array(controls_df),
        columns=["%s_diff" % c for c in control_names],
        index=zone_ids,
    )

    summary_df = pd.concat(
        [controls_df, summary_df, dif_df], axis=1, ignore_index=False
    )

    summary_cols = summary_df.columns.tolist()

    summary_df["geography"] = geography
    summary_df["id"] = summary_df.index
    summary_df.index = summary_df["geography"] + "_" + summary_df["id"].astype(str)
    summary_df = summary_df[["geography", "id"] + summary_cols]

    return summary_df


def meta_summary(
    incidence_df,
    control_spec,
    top_geography,
    top_id,
    sub_geographies,
    hh_id_col,
    layouts=None,
):
    """Summary of the controls and weights of one ``top_geography`` zone.

    Fork: ``layouts`` (:class:`_DenseLayout` per sub-geography, only with
    ``STORE_POSITIVE_WEIGHT_ROWS_ONLY``) fills the households the dense tables
    held.
    """

    if config.setting("NO_INTEGERIZATION_EVER", False):
        seed_weight_cols = ["preliminary_balanced_weight", "balanced_weight"]
        sub_weight_cols = ["balanced_weight"]
    else:
        seed_weight_cols = [
            "preliminary_balanced_weight",
            "balanced_weight",
            "integer_weight",
        ]
        sub_weight_cols = ["balanced_weight", "integer_weight"]

    incidence_df = incidence_df[incidence_df[top_geography] == top_id]

    control_cols = control_spec.target.values

    controls_df = get_control_table(top_geography)

    # controls for this geography as series
    controls = controls_df[control_cols].loc[top_id]

    incidence = incidence_df[control_cols]

    summary = pd.DataFrame(index=control_cols)

    summary.index.name = "control_name"

    summary["control_value"] = controls

    seed_geography = config.setting("seed_geography")
    seed_weights_df = get_weight_table(seed_geography)

    for c in seed_weight_cols:
        if c in seed_weights_df:
            summary_col_name = "%s_%s" % (top_geography, c)
            summary[summary_col_name] = incidence.multiply(
                seed_weights_df[c], axis="index"
            ).sum(axis=0)

    for g in sub_geographies:

        sub_weights = get_weight_table(g)

        if sub_weights is None:
            continue

        sub_weights = sub_weights[sub_weights[top_geography] == top_id]

        sub_weights = (
            sub_weights[[hh_id_col] + sub_weight_cols].groupby(hh_id_col).sum()
        )
        if layouts is not None and g in layouts:
            sub_weights = _zero_filled_by_household(
                sub_weights, incidence.index.intersection(layouts[g].households)
            )

        for c in sub_weight_cols:
            summary["%s_%s" % (g, c)] = incidence.multiply(
                sub_weights[c], axis="index"
            ).sum(axis=0)

    return summary


@inject.step()
def summarize(crosswalk, incidence_table, control_spec):
    """
    Write aggregate summary files of controls and weights for all geographic levels to output dir

    Fork: with ``STORE_POSITIVE_WEIGHT_ROWS_ONLY`` the sub-geography weight tables
    hold only their positive rows; the step fills zeros for the households and
    zones the dense table held (:class:`_DenseLayout`), so every summary table
    equals the one of a run with the setting off: a zero term adds nothing to a
    sum, and a household or zone that was never balanced stays missing. In repop
    mode no zeros are filled: the lowest table is rebuilt dense by
    ``repop_balancing``, and the tables above it are only summed in the meta
    summary, where pandas sums a missing value as zero.

    Parameters
    ----------
    crosswalk : pipeline table
    incidence_table : pipeline table
    control_spec : pipeline table

    Returns
    -------

    """

    include_integer_colums = not config.setting("NO_INTEGERIZATION_EVER", False)
    positive_only = store_positive_weight_rows_only()

    crosswalk_df = crosswalk.to_frame()
    incidence_df = incidence_table.to_frame()

    geographies = config.setting("geographies")
    seed_geography = config.setting("seed_geography")
    meta_geography = geographies[0]
    sub_geographies = geographies[geographies.index(seed_geography) + 1 :]
    super_geographies = geographies[: geographies.index(seed_geography)]

    hh_id_col = config.setting("household_id_col")

    layouts = None
    if positive_only and not inject.get_step_arg("repop", default=False):
        layouts = {}
        for geography in sub_geographies:
            weights_df = get_weight_table(geography)
            if weights_df is not None:
                layouts[geography] = _DenseLayout(
                    geography, weights_df, crosswalk_df, incidence_df
                )

    meta_ids = crosswalk_df[meta_geography].unique()
    for meta_id in meta_ids:
        meta_summary_df = meta_summary(
            incidence_df,
            control_spec,
            meta_geography,
            meta_id,
            sub_geographies,
            hh_id_col,
            layouts=layouts,
        )
        out_table("%s_%s" % (meta_geography, meta_id), meta_summary_df)

    hh_weights_summary = pd.DataFrame(index=incidence_df.index)

    # add seed level summaries
    seed_weights_df = get_weight_table(seed_geography)
    hh_weights_summary["%s_balanced_weight" % seed_geography] = seed_weights_df[
        "balanced_weight"
    ]
    if include_integer_colums:
        hh_weights_summary["%s_integer_weight" % seed_geography] = seed_weights_df[
            "integer_weight"
        ]

    for geography in sub_geographies:

        weights_df = get_weight_table(geography)

        # If repop mode, skip all other geography levels above the lowest.
        # Repop mode can only control one geography, thus it must be the lowest
        # and all others are now outdated and will throw errors downstream
        if weights_df is None or (
            inject.get_step_arg("repop", default=False) and geography != geographies[-1]
        ):
            continue

        if include_integer_colums:
            hh_weight_cols = [hh_id_col, "balanced_weight", "integer_weight"]
        else:
            hh_weight_cols = [hh_id_col, "balanced_weight"]

        hh_weights = weights_df[hh_weight_cols].groupby([hh_id_col]).sum()
        layout = layouts[geography] if layouts is not None else None
        if layout is not None:
            hh_weights = _zero_filled_by_household(hh_weights, layout.households)
        hh_weights_summary["%s_balanced_weight" % geography] = hh_weights[
            "balanced_weight"
        ]
        if include_integer_colums:
            hh_weights_summary["%s_integer_weight" % geography] = hh_weights[
                "integer_weight"
            ]

        # aggregate to seed level
        aggegrate_weights = weights_df.groupby(
            [seed_geography, hh_id_col], as_index=False
        ).sum()
        if layout is not None:
            aggegrate_weights = _zero_filled_aggregate(
                aggegrate_weights, incidence_df, layout, hh_id_col
            )
        aggegrate_weights.set_index(hh_id_col, inplace=True)

        if include_integer_colums:
            aggegrate_weight_cols = [
                seed_geography,
                "balanced_weight",
                "integer_weight",
            ]
        else:
            aggegrate_weight_cols = [seed_geography, "balanced_weight"]

        aggegrate_weights = aggegrate_weights[aggegrate_weight_cols]
        aggegrate_weights["sample_weight"] = incidence_df["sample_weight"]
        aggegrate_weights["%s_preliminary_balanced_weight" % seed_geography] = (
            seed_weights_df["preliminary_balanced_weight"]
        )
        aggegrate_weights["%s_balanced_weight" % seed_geography] = seed_weights_df[
            "balanced_weight"
        ]
        if include_integer_colums:
            aggegrate_weights["%s_integer_weight" % seed_geography] = seed_weights_df[
                "integer_weight"
            ]

        out_table("%s_aggregate" % (geography,), aggegrate_weights)

        summary_col = "integer_weight" if include_integer_colums else "balanced_weight"
        df_seed = summarize_geography(
            seed_geography,
            summary_col,
            hh_id_col,
            crosswalk_df,
            weights_df,
            incidence_df,
            layout=layout,
        )
        out_table(
            "%s_%s"
            % (
                geography,
                seed_geography,
            ),
            df_seed,
        )

        df_geo = summarize_geography(
            geography,
            summary_col,
            hh_id_col,
            crosswalk_df,
            weights_df,
            incidence_df,
            layout=layout,
        )
        out_table("%s" % (geography,), df_geo)

        # Aggregate super geographies
        for super_geo in super_geographies:
            df_super = summarize_geography(
                super_geo,
                summary_col,
                hh_id_col,
                crosswalk_df,
                weights_df,
                incidence_df,
                layout=layout,
            )
            out_table(
                "%s_%s"
                % (
                    geography,
                    super_geo,
                ),
                df_super,
            )

    out_table("hh_weights", hh_weights_summary)
