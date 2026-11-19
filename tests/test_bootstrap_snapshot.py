"""Seeded bootstrap results must not change under resampling refactors."""

import json

from validation.bootstrap_snapshot import SNAPSHOT_PATH, build_snapshot


def test_seeded_results_match_snapshot():
    expected = json.loads(SNAPSHOT_PATH.read_text())
    actual = json.loads(json.dumps(build_snapshot()))
    assert actual.keys() == expected.keys()
    for key in expected:
        assert actual[key] == expected[key], key
