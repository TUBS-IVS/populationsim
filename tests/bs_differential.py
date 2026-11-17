# PopulationSim
# See full license in LICENSE.txt.
"""Helpers for differential tests: run the example on two code states and compare.

Not a test module (no ``test_`` prefix). The example runs in a subprocess so
that ``populationsim`` is imported from exactly one code root per run.
"""

from __future__ import annotations

import io
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent

# Same model list and config layering as tests/test_steps.py::test_full_run1.
_RUNNER = """
import sys
from pathlib import Path

code_root, configs_tmp, tests_configs, example_dir, out_dir = sys.argv[1:6]

import populationsim

pkg = Path(populationsim.__file__).resolve()
assert Path(code_root).resolve() in pkg.parents, (
    "populationsim imported from %s, expected under %s" % (pkg, code_root)
)
print("populationsim.__file__ =", pkg)

from populationsim.core import inject, pipeline, tracing

example = Path(example_dir)
inject.reinject_decorated_tables()
inject.add_injectable(
    "configs_dir", [Path(configs_tmp), Path(tests_configs), example / "configs"]
)
inject.add_injectable("output_dir", Path(out_dir))
inject.add_injectable("data_dir", example / "data")
inject.clear_cache()
tracing.config_logger()

models = [
    "input_pre_processor",
    "setup_data_structures",
    "initial_seed_balancing",
    "meta_control_factoring",
    "final_seed_balancing",
    "integerize_final_seed_weights",
    "sub_balancing.geography=TRACT",
    "sub_balancing.geography=TAZ",
    "expand_households",
    "summarize",
    "write_tables",
    "write_synthetic_population",
]
pipeline.run(models=models, resume_after=None)
ids = pipeline.get_table("expanded_household_ids")
ids.to_csv(Path(out_dir) / "bs_expanded_household_ids.csv", index=False)
pipeline.close_pipeline()
"""


def export_tag(tag: str, dest: Path) -> Path:
    """Extract ``populationsim`` of git tag ``tag`` into ``dest``; fail if missing."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        ["git", "archive", "--format=tar", tag, "populationsim"],
        cwd=REPO_ROOT,
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"git tag {tag!r} cannot be exported ({proc.stderr.decode().strip()}); "
            f"run `git fetch --tags` and retry"
        )
    with tarfile.open(fileobj=io.BytesIO(proc.stdout)) as tar:
        # The archive comes from our own git tag, not from untrusted input; the
        # "data" filter is only a hardening where the interpreter offers it
        # (Python 3.9 has no filter argument).
        if hasattr(tarfile, "data_filter"):
            tar.extractall(dest, filter="data")
        else:
            tar.extractall(dest)
    return dest


def run_example(
    code_root: Path,
    out_dir: Path,
    settings_overrides: dict | None = None,
    allow_empty: bool = False,
) -> pd.DataFrame:
    """Run examples/example_test with ``populationsim`` taken from ``code_root``.

    Returns the expanded household ids table written by the run. With
    ``allow_empty`` an empty table (``NO_INTEGERIZATION_EVER`` skips the
    expansion) is returned as an empty frame instead of raising. The run's
    checkpoint store stays in ``out_dir / "pipeline.h5"``.
    """
    code_root = Path(code_root).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    configs_tmp = out_dir / "_configs_override"
    configs_tmp.mkdir(exist_ok=True)
    overrides = {"inherit_settings": True, **(settings_overrides or {})}
    (configs_tmp / "settings.yaml").write_text(yaml.safe_dump(overrides))

    # On purpose, both runs (tag code and working tree code) read tests/configs
    # and examples/example_test from the working tree: only the code differs.
    env = dict(os.environ)
    env["PYTHONPATH"] = str(code_root)
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            _RUNNER,
            str(code_root),
            str(configs_tmp),
            str(REPO_ROOT / "tests" / "configs"),
            str(REPO_ROOT / "examples" / "example_test"),
            str(out_dir),
        ],
        cwd=out_dir,
        env=env,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"example run failed (rc={proc.returncode}) for code_root={code_root}\n"
            f"stdout tail: {proc.stdout[-1500:]}\nstderr tail: {proc.stderr[-3000:]}"
        )
    ids_path = out_dir / "bs_expanded_household_ids.csv"
    if allow_empty and not ids_path.read_text().strip():
        return pd.DataFrame()
    return pd.read_csv(ids_path)
