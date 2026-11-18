"""Give the step-driving tests their own settings, whatever ran before.

``populationsim.core.config.setting`` reads the orca injectable ``settings``,
which reads ``settings.yaml`` from the injectable ``configs_dir`` (a directory
called ``configs`` in the working directory). A unit test has neither. Replacing
``config.setting`` alone is not enough: ``populationsim.balancing.wrappers`` and
``populationsim.balancing.simul_balancer`` bind the function by name
(``from populationsim.core.config import setting``), so each keeps whichever
function was current when the module was FIRST imported. Both modules are
imported here, at import time of this helper and outside any patch, so that the
binding is always the real function and ``monkeypatch`` restores exactly that.
(Importing them for the first time inside a patched test would bind the stub for
the rest of the process, and a test run after the module was imported elsewhere
would read the real ``settings`` injectable instead.)
"""

from __future__ import annotations

from populationsim.balancing import simul_balancer, wrappers
from populationsim.core import config

# Every module that binds ``setting`` by name; ``config`` itself is patched by attribute.
_MODULES_BINDING_SETTING = (wrappers, simul_balancer)


def patch_settings(monkeypatch, values=None):
    """Make ``setting(key, default)`` read the dict ``values`` everywhere.

    Patches ``config.setting`` and the name ``setting`` of every module that
    imported it by name. All patches are undone by ``monkeypatch`` at teardown.
    Returns the dict, so a test can fill it after the call.
    """
    values = {} if values is None else values

    def fake_setting(key, default=None):
        return values.get(key, default)

    monkeypatch.setattr(config, "setting", fake_setting)
    for module in _MODULES_BINDING_SETTING:
        monkeypatch.setattr(module, "setting", fake_setting)
    return values
