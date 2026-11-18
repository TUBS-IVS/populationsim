import pytest

from tests.log_isolation import fork_logging_visible_to_caplog


@pytest.fixture
def fork_logging(caplog):
    """Make ``caplog`` see the fork's log lines whatever ran before (see
    ``tests/log_isolation.py``); the previous logging state is restored."""
    with fork_logging_visible_to_caplog(caplog.handler):
        yield
