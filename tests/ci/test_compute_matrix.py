from __future__ import annotations

import json
from typing import TYPE_CHECKING

import compute_matrix as cm

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def _read_outputs(path: Path) -> dict[str, str]:
    """Parse a GITHUB_OUTPUT file (``key=value`` lines) into a dict."""
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, _, value = line.partition("=")
        result[key] = value
    return result


class TestIsFull:
    def test_push_is_always_full(self) -> None:
        assert cm.is_full("push", []) is True
        # branch is implied by on.push.branches; labels are irrelevant here
        assert cm.is_full("push", ["something-else"]) is True

    def test_pr_with_full_ci_label_is_full(self) -> None:
        assert cm.is_full("pull_request", ["full-ci"]) is True
        assert cm.is_full("pull_request", ["bug", "full-ci"]) is True

    def test_pr_without_full_ci_label_is_minimal(self) -> None:
        assert cm.is_full("pull_request", []) is False
        assert cm.is_full("pull_request", ["bug", "enhancement"]) is False


class TestParseLabels:
    def test_empty_string(self) -> None:
        assert cm.parse_labels("") == []

    def test_null_from_tojson(self) -> None:
        # toJSON renders a missing PR object as the literal "null"
        assert cm.parse_labels("null") == []

    def test_valid_array(self) -> None:
        assert cm.parse_labels('["full-ci", "bug"]') == ["full-ci", "bug"]

    def test_malformed_json(self) -> None:
        assert cm.parse_labels("[not valid") == []

    def test_non_list_json(self) -> None:
        assert cm.parse_labels('{"name": "full-ci"}') == []

    def test_drops_non_string_items(self) -> None:
        assert cm.parse_labels('["full-ci", 1, null, "bug"]') == ["full-ci", "bug"]


class TestCollectOsList:
    def test_full_matrix(self) -> None:
        assert cm.collect_os_list(cm.FULL_MATRIX) == ["ubuntu-latest", "windows-latest"]

    def test_minimal_matrix(self) -> None:
        assert cm.collect_os_list(cm.MINIMAL_MATRIX) == ["ubuntu-latest"]

    def test_dedupes_across_base_and_includes(self) -> None:
        matrix = {
            "os": ["ubuntu-latest", "ubuntu-latest"],
            "include": [{"os": "ubuntu-latest"}, {"os": "windows-latest"}],
        }
        assert cm.collect_os_list(matrix) == ["ubuntu-latest", "windows-latest"]

    def test_empty_matrix(self) -> None:
        assert cm.collect_os_list({}) == []


class TestMain:
    def test_push_writes_full_matrix(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        out = tmp_path / "out.txt"
        monkeypatch.setenv("EVENT_NAME", "push")
        monkeypatch.delenv("LABELS_JSON", raising=False)
        monkeypatch.setenv("GITHUB_OUTPUT", str(out))

        assert cm.main() == 0

        outputs = _read_outputs(out)
        assert outputs["full"] == "true"
        assert json.loads(outputs["matrix"]) == cm.FULL_MATRIX
        assert json.loads(outputs["warm_os"]) == ["ubuntu-latest", "windows-latest"]

    def test_pr_without_label_writes_minimal_matrix(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        out = tmp_path / "out.txt"
        monkeypatch.setenv("EVENT_NAME", "pull_request")
        monkeypatch.setenv("LABELS_JSON", '["bug"]')
        monkeypatch.setenv("GITHUB_OUTPUT", str(out))

        assert cm.main() == 0

        outputs = _read_outputs(out)
        assert outputs["full"] == "false"
        assert json.loads(outputs["matrix"]) == cm.MINIMAL_MATRIX
        assert json.loads(outputs["warm_os"]) == ["ubuntu-latest"]

    def test_pr_with_full_ci_label_writes_full_matrix(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        out = tmp_path / "out.txt"
        monkeypatch.setenv("EVENT_NAME", "pull_request")
        monkeypatch.setenv("LABELS_JSON", '["full-ci"]')
        monkeypatch.setenv("GITHUB_OUTPUT", str(out))

        assert cm.main() == 0
        assert _read_outputs(out)["full"] == "true"

    def test_missing_github_output_returns_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("EVENT_NAME", "push")
        monkeypatch.delenv("GITHUB_OUTPUT", raising=False)

        assert cm.main() == 1
