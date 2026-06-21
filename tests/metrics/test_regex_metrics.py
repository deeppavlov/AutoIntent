"""Unit tests for regex metrics."""

from __future__ import annotations

from autointent.metrics.regex import regex_partial_accuracy, regex_partial_precision


def test_partial_accuracy_empty_returns_minus_one() -> None:
    assert regex_partial_accuracy([], []) == -1


def test_partial_precision_counts_hits_over_nonempty_predictions() -> None:
    # label 0 is in its prediction set, label 1 is not -> 1 hit over 2 non-empty sets
    assert regex_partial_precision([0, 1], [[0], [2]]) == 0.5


def test_partial_precision_all_hits() -> None:
    assert regex_partial_precision([0, 1], [[0], [1]]) == 1.0


def test_partial_precision_no_nonempty_predictions_returns_minus_one() -> None:
    empty_preds: list[list[int]] = [[], []]
    assert regex_partial_precision([0, 1], empty_preds) == -1
