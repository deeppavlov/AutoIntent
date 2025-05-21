"""CatBoostScorer class for CatBoost-based classification with switchable encoding."""

import json
import shutil
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
from catboost import CatBoostClassifier, Pool  # type: ignore[import-untyped]
from catboost.text_processing import Dictionary, Tokenizer  # type: ignore[import-untyped]

from autointent import Context, Embedder
from autointent.configs import EmbedderConfig, TaskTypeEnum
from autointent.custom_types import ListOfLabels
from autointent.modules.base import BaseScorer

MAX_TOKEN_LENGTH = 4096
DEFAULT_TOKEN_LENGTH = 512
BINARY_CLASS_THRESHOLD = 2


class CatBoostScorer(BaseScorer):
    """CatBoost scorer using either external embeddings or CatBoost's own BoW encoding.

    Args:
        embedder_config: Config of the base transformer model (HFModelConfig, str, or dict)
            If None (default) the scorer relies on CatBoost's own Bag-of-Words encoding,
            otherwise the provided embedder is used.
        iterations: Number of boosting iterations.
        learning_rate: Learning rate for each iteration.
        loss_function: CatBoost loss function.  If None, an appropriate loss is
            chosen automatically from the task type.
        random_seed: Random seed for reproducibility.
        verbose: If True, CatBoost prints training progress.
        **catboost_kwargs: Any additional keyword arguments forwarded to
            :class:`catboost.CatBoostClassifier`.

    Example:
    -------
    .. testcode::

    from autointent.modules import CatBoostScorer


    scorer = CatBoostScorer(
        iterations=50,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3,
        eval_metric="Accuracy",
        random_seed=42,
        verbose=False,
    )
    utterances = ["hello", "goodbye", "allo", "sayonara"]
    labels = [0, 1, 0, 1]
    scorer.fit(utterances, labels)
    test_utterances = ["hi", "bye"]
    probabilities = scorer.predict(test_utterances)
    print(probabilities)

    .. testoutput::

        [[0.50525691 0.49474309]
         [0.50525691 0.49474309]]

    """

    name = "catboost"
    supports_multiclass = True
    supports_multilabel = True

    def __init__(
        self,
        embedder_config: EmbedderConfig | str | dict[str, Any] | None = None,
        iterations: int = 100,
        learning_rate: float = 0.1,
        loss_function: str | None = None,
        random_seed: int = 0,
        verbose: bool = False,
        **catboost_kwargs: Any,  # noqa: ANN401
    ) -> None:
        self._use_embedder = embedder_config is not None
        if self._use_embedder:
            self.embedder_config = EmbedderConfig.from_search_config(embedder_config)
            self._embedder = Embedder(self.embedder_config)
        else:
            self._init_catboost_text_tools()
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.loss_function = loss_function
        self.random_seed = random_seed
        self.verbose = verbose
        self.catboost_kwargs = catboost_kwargs
        self._model: CatBoostClassifier

    @classmethod
    def from_context(
        cls,
        context: Context,
        embedder_config: EmbedderConfig | str | dict[str, Any] | None = None,
        iterations: int = 100,
        learning_rate: float = 0.1,
        loss_function: str | None = None,
        random_seed: int = 0,
        verbose: bool = False,
        **catboost_kwargs: Any,  # noqa: ANN401
    ) -> "CatBoostScorer":
        if embedder_config is None:
            embedder_config = context.resolve_embedder()
        return cls(
            embedder_config=embedder_config,
            iterations=iterations,
            learning_rate=learning_rate,
            loss_function=loss_function,
            random_seed=random_seed,
            verbose=verbose,
            **catboost_kwargs,
        )


    def _init_catboost_text_tools(self) -> None:
        if not hasattr(self, "_tokenizer"):
            self._tokenizer = Tokenizer(lowercasing=True, separator_type="BySense", token_types=["Word", "Number"])
        if not hasattr(self, "_dictionary"):
            self._dictionary = Dictionary(occurence_lower_bound=1, gram_order=1)
        if not hasattr(self, "_dictionary_fitted"):
            self._dictionary_fitted = False

    def get_embedder_config(self) -> dict[str, Any]:
        if self._use_embedder:
            return self.embedder_config.model_dump()
        return {}

    def get_implicit_initialization_params(self) -> dict[str, Any]:
        return {
            "embedder_config": self.embedder_config.model_dump(),
        }

    def _encode_utterances(self, utterances: list[str]) -> npt.NDArray[np.float32]:
        if self._use_embedder:
            vecs = self._embedder.embed(utterances, task_type=TaskTypeEnum.classification)
            return np.asarray(vecs, dtype=np.float32)

        tokenized = [self._tokenizer.tokenize(u) for u in utterances]
        if not self._dictionary_fitted:
            self._dictionary.fit(tokenized)
            self._dictionary_fitted = True

        vocab = self._dictionary.size
        x = np.zeros((len(utterances), vocab), dtype=np.float32)
        for row, idxs in enumerate(self._dictionary.apply(tokenized)):
            if idxs:
                counts = np.bincount(idxs, minlength=vocab).astype(np.float32)
                x[row] = counts
        return x

    def fit(
        self,
        utterances: list[str],
        labels: ListOfLabels,
    ) -> None:
        if getattr(self, "_model", None) is not None:
            self.clear_cache()
        self._validate_task(labels)

        x = self._encode_utterances(utterances)
        y = np.asarray(labels, dtype=np.float32)

        default_loss = (
            "MultiLogloss"
            if self._multilabel
            else ("MultiClass" if self._n_classes > BINARY_CLASS_THRESHOLD else "Logloss")
        )

        self._model = CatBoostClassifier(
            iterations=self.iterations,
            learning_rate=self.learning_rate,
            loss_function=self.loss_function or default_loss,
            random_seed=self.random_seed,
            verbose=self.verbose,
            **self.catboost_kwargs,
        )
        self._model.fit(Pool(x, y))

    def predict(self, utterances: list[str]) -> npt.NDArray[np.float64]:
        if getattr(self, "_model", None) is None:
            msg = "Model is not trained. Call fit() first."
            raise RuntimeError(msg)
        x = self._encode_utterances(utterances)
        return cast("npt.NDArray[np.float64]", self._model.predict_proba(x))

    def clear_cache(self) -> None:
        if hasattr(self, "_model"):
            del self._model
        if hasattr(self, "_embedder"):
            del self._embedder
        if hasattr(self, "_tokenizer"):
            del self._tokenizer
        if hasattr(self, "_dictionary"):
            del self._dictionary
        if hasattr(self, "_dictionary_fitted"):
            del self._dictionary_fitted

    def dump(self, path: str) -> None:
        root = Path(path)
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True)

        simple_attrs: dict[str, Any] = {}
        for k, v in vars(self).items():
            if k in {"_model", "_dictionary", "_tokenizer", "_embedder"}:
                continue
            if isinstance(v, EmbedderConfig):
                simple_attrs[k] = v.model_dump()
            elif isinstance(v, type(None) | str | int | float | bool | list | dict):
                simple_attrs[k] = v
        (root / "simple_attrs.json").write_text(
            json.dumps(simple_attrs, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        if hasattr(self, "_model"):
            self._model.save_model(str(root / "model.cbm"))

        if not self._use_embedder:
            if hasattr(self, "_dictionary"):
                dict_dir = root / "dictionary"
                dict_dir.mkdir()
                self._dictionary.save(str(dict_dir / "dictionary.tsv"))

            if hasattr(self, "_tokenizer"):
                tok_params = {
                    "lowercasing": getattr(self._tokenizer, "lowercasing", True),
                    "separator_type": getattr(self._tokenizer, "separator_type", "BySense"),
                    "token_types": getattr(self._tokenizer, "token_types", ["Word", "Number"]),
                }
                (root / "tokenizer_params.json").write_text(json.dumps(tok_params), encoding="utf-8")

    @classmethod
    def load(
        cls,
        path: str,
        embedder_config: EmbedderConfig | None = None,
    ) -> "CatBoostScorer":
        root = Path(path)
        simple_attrs = json.loads((root / "simple_attrs.json").read_text(encoding="utf-8"))

        scorer = cls(
            embedder_config=embedder_config,
            iterations=simple_attrs["iterations"],
            learning_rate=simple_attrs["learning_rate"],
            loss_function=simple_attrs["loss_function"],
            random_seed=simple_attrs["random_seed"],
            verbose=simple_attrs["verbose"],
            **simple_attrs.get("catboost_kwargs", {}),
        )

        scorer._use_embedder = simple_attrs.get("_use_embedder", False)  # noqa: SLF001
        scorer._n_classes = simple_attrs.get("_n_classes")  # noqa: SLF001
        scorer._multilabel = simple_attrs.get("_multilabel")  # noqa: SLF001

        if scorer._use_embedder:
            scorer._embedder = Embedder(scorer.embedder_config)
        else:
            scorer._init_catboost_text_tools()
            dict_file = root / "dictionary" / "dictionary.tsv"
            if dict_file.exists():
                scorer._dictionary.load(str(dict_file))  # noqa: SLF001
                scorer._dictionary_fitted = simple_attrs.get("_dictionary_fitted", True)  # noqa: SLF001

            tok_params_file = root / "tokenizer_params.json"
            if tok_params_file.exists():
                tok_params = json.loads(tok_params_file.read_text(encoding="utf-8"))
                scorer._tokenizer = Tokenizer(**tok_params)  # noqa: SLF001

        model_file = root / "model.cbm"
        if model_file.exists():
            scorer._model = CatBoostClassifier()  # noqa: SLF001
            scorer._model.load_model(str(model_file))  # noqa: SLF001

        return scorer
