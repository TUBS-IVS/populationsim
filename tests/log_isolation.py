"""Keep the log tests of the fork independent of the test order.

``populationsim.core.tracing.config_logger`` (called by the pipeline and by
several upstream tests) applies the example ``logging.yaml``: the
``populationsim`` logger gets its own handlers and ``propagate: false``, and
``disable_existing_loggers`` may switch off loggers that already exist. After
that, records of the ``populationsim`` loggers never reach the root logger, where
pytest's ``caplog`` handler sits. Worse, the ``root`` section of that file makes
``dictConfig`` remove every handler of the root logger, caplog's included, so a
``caplog``-based assertion sees an empty log. The context manager below sets the
state such a test needs (fork loggers enabled and propagating, caplog's handler
on the root logger) and puts the previous state back afterwards.
"""

from __future__ import annotations

import contextlib
import logging

ROOT_NAME = "populationsim"


def _fork_loggers():
    """The ``populationsim`` logger and all existing descendants."""
    loggers = [logging.getLogger(ROOT_NAME)]
    prefix = ROOT_NAME + "."
    for name, obj in list(logging.root.manager.loggerDict.items()):
        if name.startswith(prefix) and isinstance(obj, logging.Logger):
            loggers.append(obj)
    return loggers


@contextlib.contextmanager
def fork_logging_visible_to_caplog(caplog_handler=None):
    """Enable the fork loggers, let them propagate to the root logger and make
    sure the caplog handler (if given) is attached to the root logger."""
    root = logging.getLogger()
    attached = caplog_handler is not None and caplog_handler not in root.handlers
    if attached:
        root.addHandler(caplog_handler)
    saved = [(lg, lg.disabled, lg.propagate, lg.level) for lg in _fork_loggers()]
    for lg, _, _, _ in saved:
        lg.disabled = False
        lg.propagate = True
        lg.setLevel(logging.NOTSET)
    try:
        yield
    finally:
        if attached:
            root.removeHandler(caplog_handler)
        for lg, disabled, propagate, level in saved:
            lg.disabled = disabled
            lg.propagate = propagate
            lg.setLevel(level)
