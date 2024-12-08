import numpy as np
import pytest

from autointent.context.data_handler import Tag
from autointent.modules.prediction._utils import apply_tags


def sample_data():
    labels = np.array([[1, 1, 0], [0, 1, 1]])
    scores = np.array([[0.9, 0.8, 0.1], [0.1, 0.8, 0.7]])
    tags = [Tag(name="mutual_exclusive", intent_ids=[0, 1])]
    expected_labels = np.array([[1, 0, 0], [0, 1, 1]])
    return labels, scores, tags, expected_labels


def no_conflict_data():
    labels = np.array([[1, 0, 0], [0, 1, 0]])
    scores = np.array([[0.9, 0.2, 0.1], [0.1, 0.8, 0.3]])
    tags = [Tag(name="mutual_exclusive", intent_ids=[0, 1])]
    expected_labels = labels.copy()
    return labels, scores, tags, expected_labels


def multiple_tags_data():
    labels = np.array([[1, 1, 1, 0], [1, 0, 1, 1]])
    scores = np.array([[0.9, 0.8, 0.7, 0.6], [0.95, 0.85, 0.9, 0.8]])
    tags = [Tag(name="tag1", intent_ids=[0, 1]), Tag(name="tag2", intent_ids=[2, 3])]
    expected_labels = np.array([[1, 0, 1, 0], [1, 0, 1, 0]])
    return labels, scores, tags, expected_labels


# Parametrized test function
@pytest.mark.parametrize(
    ("labels", "scores", "tags", "expected_labels"),
    [
        # Test case: No tags provided (no conflict)
        (
            np.array([[1, 0, 0], [0, 1, 1]]),
            np.array([[0.9, 0.2, 0.1], [0.1, 0.8, 0.7]]),
            [],
            np.array([[1, 0, 0], [0, 1, 1]]),
        ),
        # Test case: Single tag, no conflict
        no_conflict_data(),
        # Test case: Single tag with conflict
        sample_data(),
        # Test case: Multiple tags with conflicts
        multiple_tags_data(),
        # Test case: All intents conflict
        (
            np.array([[1, 1, 1], [1, 1, 1]]),
            np.array([[0.9, 0.85, 0.8], [0.95, 0.9, 0.88]]),
            [Tag(name="all_conflict", intent_ids=[0, 1, 2])],
            np.array([[1, 0, 0], [1, 0, 0]]),
        ),
        # Test case: No assigned intents
        (
            np.array([[0, 0, 0], [0, 0, 0]]),
            np.array([[0.1, 0.2, 0.3], [0.05, 0.1, 0.15]]),
            [Tag(name="tag1", intent_ids=[0, 1])],
            np.array([[0, 0, 0], [0, 0, 0]]),
        ),
        # Test case: Partial conflict
        (
            np.array([[1, 1, 0], [0, 1, 1]]),
            np.array([[0.7, 0.9, 0.1], [0.1, 0.8, 0.85]]),
            [Tag(name="tag1", intent_ids=[0, 1]), Tag(name="tag2", intent_ids=[1, 2])],
            np.array([[0, 1, 0], [0, 0, 1]]),
        ),
        # Test case: Overlapping tags
        (
            np.array([[1, 1, 1], [1, 1, 0]]),
            np.array([[0.9, 0.85, 0.8], [0.95, 0.9, 0.88]]),
            [Tag(name="tag1", intent_ids=[0, 1]), Tag(name="tag2", intent_ids=[1, 2])],
            np.array([[1, 0, 1], [1, 0, 0]]),
        ),
        # Test case: Conflict with same scores
        (
            np.array([[1, 1], [1, 1]]),
            np.array([[0.8, 0.8], [0.9, 0.9]]),
            [Tag(name="tag1", intent_ids=[0, 1])],
            np.array([[1, 0], [1, 0]]),
        ),
    ],
)
def test_apply_tags(labels, scores, tags, expected_labels):
    adjusted_labels = apply_tags(labels, scores, tags)
    np.testing.assert_array_equal(adjusted_labels, expected_labels)
