# PopulationSim
# See full license in LICENSE.txt.
"""Regression test: caplog must see fork log lines after the pipeline's logging setup."""

from __future__ import annotations

import logging
from pathlib import Path

from populationsim.core import inject, tracing
from tests.log_isolation import fork_logging_visible_to_caplog

EXAMPLE = Path(__file__).parent.parent / "examples" / "example_test"


def test_caplog_sees_fork_logs_after_config_logger(caplog):
    inject.add_injectable("configs_dir", [EXAMPLE / "configs"])
    inject.add_injectable("output_dir", Path(__file__).parent / "output")
    inject.clear_cache()
    # The same call the pipeline makes; the example logging.yaml sets
    # propagate: false on the populationsim logger.
    tracing.config_logger()
    fork_logger = logging.getLogger("populationsim.integerizing.wrappers")
    polluted = logging.getLogger("populationsim").propagate

    with caplog.at_level(logging.INFO):
        fork_logger.info("lost line")
    # premise: the pipeline setup has cut the fork loggers off from caplog
    assert polluted is False and "lost line" not in caplog.text

    with fork_logging_visible_to_caplog(caplog.handler):
        with caplog.at_level(logging.INFO):
            fork_logger.info("kept line")
    assert "kept line" in caplog.text
    # the previous state is back
    assert logging.getLogger("populationsim").propagate is polluted
