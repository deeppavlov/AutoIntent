from sklearn.linear_model import LogisticRegressionCV

from .base import DataHandler, ScoringModule


class LinearScorer(ScoringModule):
    """
    TODO:
    - implement different modes (incremental learning with SGD and simple learning with LogisticRegression)
    - control n_jobs
    - adjust cv
    - separate the sklearn fit() process and transformers tokenizers process (from data_handler embedding function) to avoid the warnings:
    ```
    huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
    To disable this warning, you can either:
        - Avoid using `tokenizers` before the fork if possible
        - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
    ```
    """

    def fit(self, data_handler: DataHandler):
        collection = data_handler.get_best_collection()
        dataset = collection.get(include=["embeddings", "metadatas"])
        features = dataset["embeddings"]
        labels = [dct["intent_id"] for dct in dataset["metadatas"]]
        clf = LogisticRegressionCV(cv=3, n_jobs=8)
        clf.fit(features, labels)

        self._clf = clf
        self._emb_func = collection._embedding_function

    def predict(self, utterances: list[str]):
        features = self._emb_func(utterances)
        return self._clf.predict_proba(features)
