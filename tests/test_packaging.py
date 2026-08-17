from importlib.metadata import distribution

import whales


def test_version_matches_distribution_metadata():
    assert whales.__version__ == distribution("whales").version


def test_console_entry_points_are_registered():
    scripts = {
        entry_point.name: entry_point.value
        for entry_point in distribution("whales").entry_points
        if entry_point.group == "console_scripts"
    }

    assert scripts == {
        "generate-interesting-points": "generate_interesting_points:cli",
    }
