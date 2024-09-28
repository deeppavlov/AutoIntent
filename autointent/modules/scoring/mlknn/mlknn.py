import numpy as np

from autointent.modules.scoring.base import Context, ScoringModule


class MLKnnScorer(ScoringModule):
    def __init__(self, k: int, s: float = 1.0, ignore_first_neighbours: int = 0):
        """
        Arguments
        ---
        - `k`: int, number of closest neighbors to consider during inference;
        - `s`: float, smoothing parameter for Bayesian inference;
        - `ignore_first_neighbours`: int, number of nearest neighbors to ignore.
        """
        self.k = k
        self.s = s
        self.ignore_first_neighbours = ignore_first_neighbours

    def fit(self, context: Context):
        self._multilabel = context.multilabel
        self._collection = context.get_best_collection()
        self._n_classes = context.n_classes
        self._prior_prob_true, self._prior_prob_false = self._compute_prior(context.data_handler.labels)
        self._cond_prob_true, self._cond_prob_false = self._compute_cond(
            context.data_handler.features, context.data_handler.labels
        )

    def _compute_prior(self, y):
        prior_prob_true = (self.s + y.sum(axis=0)) / (self.s * 2 + y.shape[0])
        prior_prob_false = 1 - prior_prob_true
        return prior_prob_true, prior_prob_false

    def _compute_cond(self, x, y):
        c = np.zeros((self._n_classes, self.k + 1), dtype=int)
        cn = np.zeros((self._n_classes, self.k + 1), dtype=int)

        neighbors = self._get_neighbors(x)

        for instance in range(x.shape[0]):
            deltas = y[neighbors[instance], :].sum(axis=0)
            for label in range(self._n_classes):
                if y[instance, label] == 1:
                    c[label, deltas[label]] += 1
                else:
                    cn[label, deltas[label]] += 1

        c_sum = c.sum(axis=1)
        cn_sum = cn.sum(axis=1)

        cond_prob_true = (self.s + c) / (self.s * (self.k + 1) + c_sum[:, None])
        cond_prob_false = (self.s + cn) / (self.s * (self.k + 1) + cn_sum[:, None])

        return cond_prob_true, cond_prob_false

    def _get_neighbors(self, x):
        query_res = self._collection.query(
            query_texts=x, n_results=self.k + self.ignore_first_neighbours, include=["distances"]
        )
        return [res[self.ignore_first_neighbours :] for res in query_res["distances"]]

    def predict(self, x):
        result = np.zeros((x.shape[0], self._n_classes), dtype=int)
        neighbors = self._get_neighbors(x)

        for instance in range(x.shape[0]):
            deltas = self._collection.metadata[neighbors[instance], :].sum(axis=0)

            for label in range(self._n_classes):
                p_true = self._prior_prob_true[label] * self._cond_prob_true[label, deltas[label]]
                p_false = self._prior_prob_false[label] * self._cond_prob_false[label, deltas[label]]
                result[instance, label] = int(p_true >= p_false)

        return result

    def predict_proba(self, x):
        result = np.zeros((x.shape[0], self._n_classes), dtype=float)
        neighbors = self._get_neighbors(x)

        for instance in range(x.shape[0]):
            deltas = self._collection.metadata[neighbors[instance], :].sum(axis=0)

            for label in range(self._n_classes):
                p_true = self._prior_prob_true[label] * self._cond_prob_true[label, deltas[label]]
                p_false = self._prior_prob_false[label] * self._cond_prob_false[label, deltas[label]]
                result[instance, label] = p_true / (p_true + p_false)

        return result
