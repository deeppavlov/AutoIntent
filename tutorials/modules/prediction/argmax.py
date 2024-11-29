# %% [markdown]
"""
# Argmax Predictor

One can use simple argmax predictor for multiclass classification problem.
"""

# %%
import numpy as np

from autointent.modules.prediction import ArgmaxPredictor

# %% [markdown]
"""
Example usage:
"""

# %%
predictor = ArgmaxPredictor()
predictor.fit(scores=np.array([[0.1, 0.9]]), labels=[1])
scores = np.array([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
predictions = predictor.predict(scores)
np.testing.assert_array_equal(predictions, np.array([1, 0, 1]))
