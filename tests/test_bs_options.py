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


def test_options_logged_once(settings, caplog):
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
