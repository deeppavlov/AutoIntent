import pytest

from autointent.metrics.decision import handle_oos


@pytest.mark.parametrize(
    argnames=("y_true", "y_pred", "expected_true", "expected_pred"),
    argvalues=[
        ([0, 1, 2, 3, None, None], [0, 1, 2, 2, None, 1], [0, 1, 2, 3, 4, 4], [0, 1, 2, 2, 4, 1]),
        (
            [[0, 0, 1], [0, 1, 0], None, None],
            [[0, 0, 1], [0, 1, 1], None, [0, 1, 0]],
            [[0, 0, 1, 1], [0, 1, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1]],
            [[0, 0, 1, 1], [0, 1, 1, 1], [0, 0, 0, 1], [0, 1, 0, 1]],
        ),
    ],
)
def test_oos_handling(y_true, y_pred, expected_true, expected_pred):
    handled_true, handled_pred = handle_oos(y_true, y_pred)
    assert handled_true == expected_true
    assert handled_pred == expected_pred
