from sklearn.linear_model import LogisticRegressionCV

from .base import DataHandler, ScoringModule


class LinearScorer(ScoringModule):
    """
    TODO:
    - implement different modes (incremental learning with SGD and simple learning with LogisticRegression)
    - control n_jobs
    - adjust cv
    - ensure that embeddings of train set are not recalculated
    """

    def fit(self, data_handler: DataHandler):
        dataset = data_handler.collection.get(include=["embeddings", "metadatas"])
        features = dataset["embeddings"]
        labels = [dct["intent_id"] for dct in dataset["metadatas"]]
        clf = LogisticRegressionCV(cv=3, n_jobs=8, multi_class="multinomial")
        clf.fit(features, labels)

        self._clf = clf
        self._emb_func = data_handler.collection._embedding_function

    def predict(self, utterances: list[str]):
        features = self._emb_func(utterances)
        return self._clf.predict_proba(features)
