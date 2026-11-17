# PopulationSim
# See full license in LICENSE.txt.

from tests.bs_differential import REPO_ROOT, export_tag, run_example


def _canonical(df):
    return df.sort_values(list(df.columns)).reset_index(drop=True)


def test_defaults_match_v0_10_0(tmp_path):
    """With no fork setting touched, the output equals the upstream tag's."""
    tag_root = export_tag("v0.10.0", tmp_path / "tag_code")
    base = run_example(tag_root, tmp_path / "out_tag")
    new = run_example(REPO_ROOT, tmp_path / "out_tree")
    assert len(base) > 0
    assert _canonical(base).equals(
        _canonical(new)
    ), f"rows tag={len(base)} tree={len(new)}"
