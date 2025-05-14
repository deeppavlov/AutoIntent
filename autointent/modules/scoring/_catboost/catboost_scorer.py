"""CatBoostScorer class for CatBoost-based classification with switchable encoding."""

import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import torch
from catboost import CatBoostClassifier, Pool  # type: ignore[import-untyped]
from catboost.text_processing import Dictionary, Tokenizer  # type: ignore[import-untyped]
from transformers import AutoModel, AutoTokenizer

from autointent import Context
from autointent.configs import EmbedderConfig
from autointent.custom_types import ListOfLabels
from autointent.modules.base import BaseScorer

MAX_TOKEN_LENGTH = 4096
DEFAULT_TOKEN_LENGTH = 512
BINARY_CLASS_THRESHOLD = 2


class CatBoostScorer(BaseScorer):
    """CatBoost scorer using either external embeddings or CatBoost's own BoW encoding.

    Args:
        classification_model_config: Config of the base transformer model (HFModelConfig, str, or dict)
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
        classification_model_config: EmbedderConfig | str | dict[str, Any] | None = None,
        iterations: int = 100,
        learning_rate: float = 0.1,
        loss_function: str | None = None,
        random_seed: int = 0,
        verbose: bool = False,
        **catboost_kwargs: Any,  # noqa: ANN401
    ) -> None:
        self.classification_model_config = EmbedderConfig.from_search_config(classification_model_config)
        self._use_embedder = classification_model_config is not None
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.loss_function = loss_function
        self.random_seed = random_seed
        self.verbose = verbose
        self.catboost_kwargs = catboost_kwargs
        self._model: CatBoostClassifier
        self._embedder: Any
        self._tokenizer: Tokenizer
        self._dictionary: Dictionary

    @classmethod
    def from_context(
        cls,
        context: Context,
        classification_model_config: EmbedderConfig | str | dict[str, Any] | None = None,
        iterations: int = 100,
        learning_rate: float = 0.1,
        loss_function: str | None = None,
        random_seed: int = 0,
        verbose: bool = False,
        **catboost_kwargs: Any,  # noqa: ANN401
    ) -> "CatBoostScorer":
        if classification_model_config is None:
            classification_model_config = context.resolve_embedder()
        return cls(
            classification_model_config=classification_model_config,
            iterations=iterations,
            learning_rate=learning_rate,
            loss_function=loss_function,
            random_seed=random_seed,
            verbose=verbose,
            **catboost_kwargs,
        )

    def get_classification_model_config(self) -> dict[str, Any]:
        return self.classification_model_config.model_dump()

    def get_implicit_initialization_params(self) -> dict[str, Any]:
        return {
            "classification_model_config": self.classification_model_config.model_dump(),
            "encoding": self.encoding,  # type: ignore [attr-defined]
        }

    def _load_embedder(self) -> Any:  # noqa: ANN401
        if getattr(self, "_embedder", None) is not None:
            return self._embedder
        cfg = self.classification_model_config
        if hasattr(cfg, "encode"):
            self._embedder = cfg
            return self._embedder

        model_name = getattr(cfg, "model_name", None)
        if model_name is None and hasattr(cfg, "model_dump"):
            model_name = cfg.model_dump().get("model_name")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device).eval()

        raw_max = getattr(tokenizer, "model_max_length", None)
        max_len = (
            DEFAULT_TOKEN_LENGTH
            if not isinstance(raw_max, int) or raw_max <= 0 or raw_max > MAX_TOKEN_LENGTH
            else raw_max
        )

        def encode(texts: list[str]) -> npt.NDArray[np.float32]:
            with torch.no_grad():
                batch = tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=max_len,
                    return_tensors="pt",
                )
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                return np.array(embeddings, dtype=np.float32)

        self._embedder = encode
        return self._embedder

    def _init_text_tools(self) -> None:
        if not hasattr(self, "_tokenizer"):
            self._tokenizer = Tokenizer(lowercasing=True, separator_type="BySense", token_types=["Word", "Number"])
        if not hasattr(self, "_dictionary"):
            self._dictionary = Dictionary(occurence_lower_bound=1, gram_order=1)

    def _encode_utterances(self, utterances: list[str]) -> npt.NDArray[np.float32]:
        if self._use_embedder:
            embedder = self._load_embedder()
            vecs = embedder.encode(utterances) if hasattr(embedder, "encode") else embedder(utterances)
            return np.asarray(vecs, dtype=np.float32)
        self._init_text_tools()
        tokenized = [self._tokenizer.tokenize(u) for u in utterances]
        if not hasattr(self, "_dictionary_fitted"):
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
        y: npt.NDArray[np.float32] | npt.NDArray[np.int64]
        if self._multilabel:
            y_mat = np.zeros((len(labels), self._n_classes), dtype=np.float32)
            for i, lbls in enumerate(cast("Sequence[Sequence[int]]", labels)):
                for lbl in lbls:
                    y_mat[i, lbl] = 1.0
            y = y_mat
        else:
            y = np.asarray(cast("Sequence[int]", labels), dtype=np.int64)

        default_loss = (
            "MultiLabel"
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

    def dump(self, path: str) -> None:  # noqa: C901
        """Save scorer and all artefacts needed for inference to path."""
        root = Path(path)
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True)

        simple_attrs: dict[str, Any] = {}
        for k, v in vars(self).items():
            if k in {"_model", "_dictionary", "_tokenizer"}:
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

        if self._use_embedder and hasattr(self, "_embedder"):
            obj = getattr(self._embedder, "__self__", self._embedder)
            if hasattr(obj, "save_pretrained"):
                obj.save_pretrained(str(root / "hf_model"))
                if hasattr(self, "_tokenizer") and hasattr(self._tokenizer, "save_pretrained"):
                    self._tokenizer.save_pretrained(str(root / "hf_tokenizer"))

    @classmethod
    def load(
        cls,
        path: str,
        embedder_config: EmbedderConfig | None = None,
    ) -> "CatBoostScorer":
        """Load scorer dumped with :pymeth:`dump`."""
        root = Path(path)
        simple_attrs = json.loads((root / "simple_attrs.json").read_text(encoding="utf-8"))

        cfg_dict = embedder_config.model_dump() if embedder_config else simple_attrs["classification_model_config"]
        cfg = EmbedderConfig.model_validate(cfg_dict)

        scorer = cls(
            classification_model_config=cfg,
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

        model_file = root / "model.cbm"
        if model_file.exists():
            scorer._model = CatBoostClassifier()  # noqa: SLF001
            scorer._model.load_model(str(model_file))  # noqa: SLF001

        dict_file = root / "dictionary" / "dictionary.tsv"
        if dict_file.exists():
            scorer._dictionary = Dictionary()  # noqa: SLF001
            scorer._dictionary.load(str(dict_file))  # noqa: SLF001
            scorer._dictionary_fitted = simple_attrs.get("_dictionary_fitted", True)  # noqa: SLF001

        tok_params_file = root / "tokenizer_params.json"
        if tok_params_file.exists():
            tok_params = json.loads(tok_params_file.read_text(encoding="utf-8"))
            scorer._tokenizer = Tokenizer(**tok_params)  # noqa: SLF001

        if scorer._use_embedder:  # noqa: SLF001
            emb_dir = root / "hf_model"
            if emb_dir.exists():
                tok_dir = root / "hf_tokenizer"
                scorer._tokenizer = AutoTokenizer.from_pretrained(str(tok_dir if tok_dir.exists() else emb_dir))  # noqa: SLF001
                model = AutoModel.from_pretrained(str(emb_dir)).to(
                    torch.device("cuda" if torch.cuda.is_available() else "cpu")
                )
                model.eval()

                raw_max = getattr(scorer._tokenizer, "model_max_length", None)  # noqa: SLF001
                max_len = (
                    DEFAULT_TOKEN_LENGTH
                    if not isinstance(raw_max, int) or raw_max <= 0 or raw_max > MAX_TOKEN_LENGTH
                    else raw_max
                )

                def encode(texts: list[str]) -> npt.NDArray[np.float32]:
                    with torch.no_grad():
                        batch = scorer._tokenizer(  # noqa: SLF001
                            texts,
                            padding=True,
                            truncation=True,
                            max_length=max_len,
                            return_tensors="pt",
                        ).to(model.device)
                        return model(**batch).last_hidden_state[:, 0, :].cpu().numpy().astype(np.float32)  # type: ignore[no-any-return]

                scorer._embedder = encode  # noqa: SLF001

        return scorer
