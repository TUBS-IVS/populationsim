# PopulationSim
# See full license in LICENSE.txt.

import pytest

from populationsim.core import bs_run_options, config


@pytest.fixture
def settings(monkeypatch):
    """Inject a settings dict by patching config.setting."""
    values = {}
    monkeypatch.setattr(
        config, "setting", lambda key, default=None: values.get(key, default)
    )
    return values


@pytest.mark.parametrize(
    "reader, key",
    [
        (
            bs_run_options.store_positive_weight_rows_only,
            "STORE_POSITIVE_WEIGHT_ROWS_ONLY",
        ),
        (bs_run_options.fast_paths, "FAST_PATHS"),
    ],
)
def test_run_options_default_off_and_read_booleans(settings, reader, key):
    assert reader() is False
    settings[key] = True
    assert reader() is True
    settings[key] = False
    assert reader() is False


@pytest.mark.parametrize(
    "reader, key",
    [
        (
            bs_run_options.store_positive_weight_rows_only,
            "STORE_POSITIVE_WEIGHT_ROWS_ONLY",
        ),
        (bs_run_options.fast_paths, "FAST_PATHS"),
    ],
)
@pytest.mark.parametrize("bad", [1, 0, "true", "on", None, 1.0])
def test_run_options_refuse_non_booleans(settings, reader, key, bad):
    settings[key] = bad
    if bad is None:
        # an explicit null is "not set": config.setting returns the default
        settings.pop(key)
        assert reader() is False
        return
    with pytest.raises(ValueError) as error:
        reader()
    assert key in str(error.value)
