import numpy as np

from autointent.modules import JinoosPredictor


def detect_oos(scores, labels, thresh):
    """
    `labels`: labels without oos detection
    """
    mask = np.max(scores, axis=1) < thresh
    labels[mask] = -1
    return labels

def test_predict_returns_correct_indices(multiclass_fit_data):
    predictor = JinoosPredictor()
    predictor.fit(*multiclass_fit_data)
    scores = np.array([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])

    # inference
    predictions = predictor.predict(scores)
    desired = detect_oos(scores, np.array([1,0,1]), predictor.thresh)

    np.testing.assert_array_equal(predictions, desired)

