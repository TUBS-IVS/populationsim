# PopulationSim
# See full license in LICENSE.txt.

import logging

import pytest

from populationsim.core import config
from populationsim.integerizing import bs_options


@pytest.fixture
def settings(monkeypatch):
    """Inject a settings dict by patching config.setting."""
    values = {}
    monkeypatch.setattr(
        config, "setting", lambda key, default=None: values.get(key, default)
    )
    bs_options.reset_for_tests()
    yield values
    bs_options.reset_for_tests()


def test_timeout_default_is_60(settings):
    assert bs_options.integerizer_timeout_seconds() == 60


def test_timeout_new_name(settings):
    settings["INTEGERIZER_TIMEOUT"] = 120
    assert bs_options.integerizer_timeout_seconds() == 120


def test_timeout_old_name_alias(settings):
    settings["INTEGIZER_TIMEOUT"] = 7200
    assert bs_options.integerizer_timeout_seconds() == 7200


def test_timeout_both_equal_ok(settings):
    settings["INTEGIZER_TIMEOUT"] = 300
    settings["INTEGERIZER_TIMEOUT"] = 300
    assert bs_options.integerizer_timeout_seconds() == 300


def test_timeout_conflict_raises(settings):
    settings["INTEGIZER_TIMEOUT"] = 300
    settings["INTEGERIZER_TIMEOUT"] = 600
    with pytest.raises(ValueError) as exc:
        bs_options.integerizer_timeout_seconds()
    assert "INTEGIZER_TIMEOUT" in str(exc.value)
    assert "INTEGERIZER_TIMEOUT" in str(exc.value)


@pytest.mark.parametrize("bad", [0, -5, 1.5, 60.0, "60", True])
def test_timeout_rejects_zero_and_float(settings, bad):
    settings["INTEGERIZER_TIMEOUT"] = bad
    with pytest.raises(ValueError):
        bs_options.integerizer_timeout_seconds()


def test_draw_method_values(settings):
    assert bs_options.draw_method() == "off"
    for ok in ("off", "pareto", "gumbel"):
        settings["INTEGERIZER_DRAW"] = ok
        assert bs_options.draw_method() == ok
    for bad in ("yes", True, False, "Pareto", 1):
        settings["INTEGERIZER_DRAW"] = bad
        with pytest.raises(ValueError):
            bs_options.draw_method()


def test_seed_range(settings):
    assert bs_options.random_seed() == 0
    for ok in (0, 1, 2**32 - 1):
        settings["INTEGERIZER_RANDOM_SEED"] = ok
        assert bs_options.random_seed() == ok
    for bad in (-1, 2**32, 1.5, "7", True):
        settings["INTEGERIZER_RANDOM_SEED"] = bad
        with pytest.raises(ValueError):
            bs_options.random_seed()


def test_flags_default_off(settings):
    assert bs_options.keep_fractional_sample_weights() is False
    assert bs_options.path_record_enabled() is False
    settings["KEEP_FRACTIONAL_SAMPLE_WEIGHTS"] = True
    settings["INTEGERIZER_PATH_RECORD"] = True
    assert bs_options.keep_fractional_sample_weights() is True
    assert bs_options.path_record_enabled() is True


def _first5(seed, kind, label, settings):
    settings["INTEGERIZER_RANDOM_SEED"] = seed
    return tuple(bs_options.call_rng(kind, label).gumbel(size=5))


def test_call_rng_depends_on_seed_kind_and_label(settings):
    base = _first5(1, "single", "TAZ_1", settings)
    assert base == _first5(1, "single", "TAZ_1", settings)
    assert base != _first5(2, "single", "TAZ_1", settings)
    assert base != _first5(1, "simul", "TAZ_1", settings)
    assert base != _first5(1, "single", "TAZ_2", settings)
    with pytest.raises(ValueError):
        bs_options.call_rng("other", "TAZ_1")


def test_options_logged_once(settings, caplog, fork_logging):
    with caplog.at_level(logging.INFO, logger=bs_options.logger.name):
        bs_options.log_options_once()
        bs_options.log_options_once()
    records = [r for r in caplog.records if r.name == bs_options.logger.name]
    assert len(records) == 1
    assert records[0].levelno == logging.INFO
    msg = records[0].getMessage()
    assert "do_simul_integerizing" not in msg
    for token in ("timeout", "draw", "seed", "fractional", "path_record"):
        assert token in msg


def test_rounding_mode_defaults_to_round(settings):
    assert bs_options.rounding_mode() == "round"
    settings["INTEGERIZER_ROUNDING"] = "controlled"
    assert bs_options.rounding_mode() == "controlled"


@pytest.mark.parametrize("bad", ["Controlled", "yes", True, 1])
def test_rounding_mode_rejects_unknown_value(settings, bad):
    settings["INTEGERIZER_ROUNDING"] = bad
    with pytest.raises(ValueError, match="INTEGERIZER_ROUNDING"):
        bs_options.rounding_mode()


def test_controlled_rounding_rejects_cvxpy(settings):
    assert bs_options.resolve_rounding_mode(True) == "round"
    assert bs_options.resolve_rounding_mode(False) == "round"
    settings["INTEGERIZER_ROUNDING"] = "controlled"
    assert bs_options.resolve_rounding_mode(False) == "controlled"
    with pytest.raises(ValueError, match="INTEGERIZER_ROUNDING"):
        bs_options.resolve_rounding_mode(True)


def test_penalty_settings_defaults(settings):
    got = bs_options.penalty_settings()
    assert isinstance(got, bs_options.PenaltySettings)
    assert tuple(got) == (100.0, 100000.0, 10.0, 100000.0)
    settings["INTEGERIZER_PENALTY_FLOOR"] = 50
    assert bs_options.penalty_settings().floor == 50.0


@pytest.mark.parametrize(
    "key,bad",
    [
        ("INTEGERIZER_PENALTY_FLOOR", 0),
        ("INTEGERIZER_PENALTY_CAP", 50),
        ("INTEGERIZER_PARENT_TOTAL_FACTOR", 0.5),
        ("INTEGERIZER_CONSERVE_PENALTY", -1),
        ("INTEGERIZER_PENALTY_FLOOR", float("nan")),
        ("INTEGERIZER_PENALTY_CAP", float("inf")),
        ("INTEGERIZER_CONSERVE_PENALTY", True),
        ("INTEGERIZER_PARENT_TOTAL_FACTOR", "10"),
    ],
)
def test_penalty_settings_reject_invalid(settings, key, bad):
    settings[key] = bad
    with pytest.raises(ValueError, match=key):
        bs_options.penalty_settings()


def test_options_log_controlled_values(settings, caplog):
    settings["INTEGERIZER_ROUNDING"] = "controlled"
    with caplog.at_level(logging.INFO, logger=bs_options.logger.name):
        bs_options.log_options_once()
    msg = " ".join(r.getMessage() for r in caplog.records)
    assert "rounding=controlled" in msg
    assert "100000.0" in msg and "penalty_floor" in msg


def test_controlled_solver_settings_defaults(settings):
    assert bs_options.controlled_solver() == "highs-ipm"
    assert bs_options.controlled_retry_solver() == "clp-dual"
    assert bs_options.controlled_repair() is True


def test_controlled_solver_settings_accept_every_solver(settings):
    from populationsim.integerizing.constants import CONTROLLED_SOLVERS

    for name in CONTROLLED_SOLVERS:
        settings["INTEGERIZER_CONTROLLED_SOLVER"] = name
        settings["INTEGERIZER_CONTROLLED_RETRY_SOLVER"] = "none"
        assert bs_options.controlled_solver() == name
        assert bs_options.controlled_retry_solver() == "none"
    settings["INTEGERIZER_CONTROLLED_SOLVER"] = "clp-dual"
    settings["INTEGERIZER_CONTROLLED_RETRY_SOLVER"] = "highs-ds"
    assert bs_options.controlled_retry_solver() == "highs-ds"
    settings["INTEGERIZER_CONTROLLED_REPAIR"] = False
    assert bs_options.controlled_repair() is False


@pytest.mark.parametrize(
    "key,bad",
    [
        ("INTEGERIZER_CONTROLLED_SOLVER", "highs"),
        ("INTEGERIZER_CONTROLLED_SOLVER", "none"),
        ("INTEGERIZER_CONTROLLED_SOLVER", "HIGHS-IPM"),
        ("INTEGERIZER_CONTROLLED_SOLVER", 1),
        ("INTEGERIZER_CONTROLLED_RETRY_SOLVER", "gurobi"),
        ("INTEGERIZER_CONTROLLED_RETRY_SOLVER", None),
        ("INTEGERIZER_CONTROLLED_RETRY_SOLVER", False),
        ("INTEGERIZER_CONTROLLED_REPAIR", "yes"),
        ("INTEGERIZER_CONTROLLED_REPAIR", 1),
    ],
)
def test_controlled_solver_settings_reject_invalid(settings, key, bad):
    settings[key] = bad
    reader = {
        "INTEGERIZER_CONTROLLED_SOLVER": bs_options.controlled_solver,
        "INTEGERIZER_CONTROLLED_RETRY_SOLVER": bs_options.controlled_retry_solver,
        "INTEGERIZER_CONTROLLED_REPAIR": bs_options.controlled_repair,
    }[key]
    with pytest.raises(ValueError, match=key):
        reader()


def test_controlled_retry_solver_must_differ(settings):
    settings["INTEGERIZER_CONTROLLED_SOLVER"] = "clp-dual"
    with pytest.raises(ValueError) as exc:
        bs_options.controlled_retry_solver()
    assert "INTEGERIZER_CONTROLLED_RETRY_SOLVER" in str(exc.value)
    assert "INTEGERIZER_CONTROLLED_SOLVER" in str(exc.value)


def test_options_log_controlled_solver_values(settings, caplog):
    settings["INTEGERIZER_ROUNDING"] = "controlled"
    settings["INTEGERIZER_CONTROLLED_REPAIR"] = False
    with caplog.at_level(logging.INFO, logger=bs_options.logger.name):
        bs_options.log_options_once()
    msg = " ".join(r.getMessage() for r in caplog.records)
    assert "controlled_solver=highs-ipm" in msg
    assert "controlled_retry_solver=clp-dual" in msg
    assert "controlled_repair=False" in msg


def test_options_log_omits_solver_values_in_round_mode(settings, caplog):
    with caplog.at_level(logging.INFO, logger=bs_options.logger.name):
        bs_options.log_options_once()
    msg = " ".join(r.getMessage() for r in caplog.records)
    assert "controlled_solver" not in msg
    assert "controlled_method" not in msg and "search_" not in msg


def test_controlled_method_defaults_to_search(settings):
    assert bs_options.controlled_method() == "search"
    settings["INTEGERIZER_CONTROLLED_METHOD"] = "lp"
    assert bs_options.controlled_method() == "lp"


@pytest.mark.parametrize("bad", ["LP", "flow", "", None, True, 1])
def test_controlled_method_rejects_unknown_value(settings, bad):
    settings["INTEGERIZER_CONTROLLED_METHOD"] = bad
    with pytest.raises(ValueError, match="INTEGERIZER_CONTROLLED_METHOD"):
        bs_options.controlled_method()


def test_search_settings_defaults_and_values(settings):
    got = bs_options.search_settings()
    assert isinstance(got, bs_options.SearchSettings)
    assert tuple(got) == (200, 24)
    settings["INTEGERIZER_SEARCH_MAX_SWEEPS"] = 1
    settings["INTEGERIZER_SEARCH_SHORTLIST"] = 1
    assert tuple(bs_options.search_settings()) == (1, 1)


@pytest.mark.parametrize(
    "key,bad",
    [
        ("INTEGERIZER_SEARCH_MAX_SWEEPS", 0),
        ("INTEGERIZER_SEARCH_MAX_SWEEPS", -3),
        ("INTEGERIZER_SEARCH_MAX_SWEEPS", 10.0),
        ("INTEGERIZER_SEARCH_MAX_SWEEPS", "200"),
        ("INTEGERIZER_SEARCH_MAX_SWEEPS", True),
        ("INTEGERIZER_SEARCH_SHORTLIST", 0),
        ("INTEGERIZER_SEARCH_SHORTLIST", 2.5),
        ("INTEGERIZER_SEARCH_SHORTLIST", None),
        ("INTEGERIZER_SEARCH_SHORTLIST", False),
    ],
)
def test_search_settings_reject_invalid(settings, key, bad):
    settings[key] = bad
    with pytest.raises(ValueError, match=key):
        bs_options.search_settings()


def test_options_log_search_values(settings, caplog):
    settings["INTEGERIZER_ROUNDING"] = "controlled"
    settings["INTEGERIZER_SEARCH_SHORTLIST"] = 12
    with caplog.at_level(logging.INFO, logger=bs_options.logger.name):
        bs_options.log_options_once()
    msg = " ".join(r.getMessage() for r in caplog.records)
    assert "controlled_method=search" in msg
    assert "search_max_sweeps=200" in msg
    assert "search_shortlist=12" in msg
