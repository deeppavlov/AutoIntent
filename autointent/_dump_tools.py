import importlib
import json
import logging
from pathlib import Path
from typing import Any, TypeAlias

import joblib
import numpy as np
import numpy.typing as npt
from peft import PeftModel
from pydantic import BaseModel
from sklearn.base import BaseEstimator
from torch import nn
from transformers import (  # type: ignore[attr-defined]
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

from autointent import Embedder, Ranker, VectorIndex
from autointent._wrappers import BaseTorchModule
from autointent.configs import CrossEncoderConfig, EmbedderConfig
from autointent.context.optimization_info import Artifact
from autointent.schemas import TagsList

ModuleSimpleAttributes = None | str | int | float | bool | list  # type: ignore[type-arg]

ModuleAttributes: TypeAlias = (
    ModuleSimpleAttributes | TagsList | np.ndarray | Embedder | VectorIndex | BaseEstimator | Ranker | nn.Module  # type: ignore[type-arg]
)

logger = logging.getLogger(__name__)


class Dumper:
    tags = "tags"
    simple_attrs = "simple_attrs.json"
    arrays = "arrays.npz"
    embedders = "embedders"
    indexes = "vector_indexes"
    estimators = "estimators"
    cross_encoders = "cross_encoders"
    pydantic_models: str = "pydantic"
    hf_models = "hf_models"
    hf_tokenizers = "hf_tokenizers"
    torch_models = "torch_models"
    ptuning_models = "ptuning_models"

    @staticmethod
    def make_subdirectories(path: Path, exists_ok: bool = False) -> None:
        """Make subdirectories for dumping.

        Args:
            path: Path to make subdirectories in
            exists_ok: If True, do not raise an error if the directory already exists
        """
        subdirectories = [
            path / Dumper.tags,
            path / Dumper.embedders,
            path / Dumper.indexes,
            path / Dumper.estimators,
            path / Dumper.cross_encoders,
            path / Dumper.pydantic_models,
            path / Dumper.hf_models,
            path / Dumper.hf_tokenizers,
            path / Dumper.torch_models,
            path / Dumper.ptuning_models,
        ]
        for subdir in subdirectories:
            subdir.mkdir(parents=True, exist_ok=exists_ok)

    @staticmethod
    def dump(obj: Any, path: Path, exists_ok: bool = False, exclude: list[type[Any]] | None = None) -> None:  # noqa: ANN401, C901, PLR0912, PLR0915
        """Dump modules attributes to filestystem.

        Args:
            obj: Object to dump
            path: Path to dump to
            exists_ok: If True, do not raise an error if the directory already exists
            exclude: List of types to exclude from dumping
        """
        attrs: dict[str, ModuleAttributes] = vars(obj)
        simple_attrs = {}
        arrays: dict[str, npt.NDArray[Any]] = {}

        Dumper.make_subdirectories(path, exists_ok)

        for key, val in attrs.items():
            if isinstance(val, Artifact) or (exclude and isinstance(val, tuple(exclude))):
                continue
            if isinstance(val, TagsList):
                val.dump(path / Dumper.tags / key)
            elif isinstance(val, ModuleSimpleAttributes):
                simple_attrs[key] = val
            elif isinstance(val, np.ndarray):
                arrays[key] = val
            elif isinstance(val, Embedder):
                val.dump(path / Dumper.embedders / key)
            elif isinstance(val, VectorIndex):
                val.dump(path / Dumper.indexes / key)
            elif isinstance(val, BaseEstimator):
                joblib.dump(val, path / Dumper.estimators / key)
            elif isinstance(val, Ranker):
                val.save(str(path / Dumper.cross_encoders / key))
            elif isinstance(val, BaseModel):
                try:
                    class_info = {"name": val.__class__.__name__, "module": val.__class__.__module__}
                    pydantic_path = path / Dumper.pydantic_models / key
                    pydantic_path.mkdir(parents=True, exist_ok=exists_ok)
                    with (pydantic_path / "class_info.json").open("w", encoding="utf-8") as file:
                        json.dump(class_info, file, ensure_ascii=False, indent=4)
                    with (pydantic_path / "model_dump.json").open("w", encoding="utf-8") as file:
                        json.dump(val.model_dump(), file, ensure_ascii=False, indent=4)
                except Exception as e:
                    msg = f"Error dumping pydantic model {key}: {e}"
                    logging.exception(msg)
            elif isinstance(val, PeftModel):
                # dumping peft models is a nightmare...
                # this might break with new versions of peft
                try:
                    if val._is_prompt_learning:  # noqa: SLF001
                        # strategy to save prompt learning models: save prompt encoder and bert classifier separately
                        model_path = path / Dumper.ptuning_models / key
                        model_path.mkdir(parents=True, exist_ok=True)
                        val.save_pretrained(str(model_path / "peft"))
                        val.base_model.save_pretrained(model_path / "base_model")  # type: ignore[attr-defined]
                    else:
                        # strategy to save lora models: merge adapters and save as usual hugging face model
                        model_path = path / Dumper.hf_models / key
                        model_path.mkdir(parents=True, exist_ok=True)
                        merged_model: PreTrainedModel = val.merge_and_unload()
                        merged_model.save_pretrained(model_path)  # type: ignore[attr-defined]
                except Exception as e:
                    msg = f"Error dumping PeftModel {key}: {e}"
                    logger.exception(msg)
            elif isinstance(val, PreTrainedModel):
                model_path = path / Dumper.hf_models / key
                model_path.mkdir(parents=True, exist_ok=True)
                try:
                    val.save_pretrained(model_path)  # type: ignore[attr-defined]
                except Exception as e:
                    msg = f"Error dumping HF model {key}: {e}"
                    logger.exception(msg)
            elif isinstance(val, BaseTorchModule):
                model_path = path / Dumper.torch_models / key
                model_path.mkdir(parents=True, exist_ok=True)
                try:
                    class_info = {
                        "module": val.__class__.__module__,
                        "name": val.__class__.__name__,
                    }
                    with (model_path / "class_info.json").open("w") as f:
                        json.dump(class_info, f)
                    val.dump(model_path)
                except Exception as e:
                    msg = f"Error dumping torch model {key}: {e}"
                    logger.exception(msg)
            elif isinstance(val, PreTrainedTokenizer | PreTrainedTokenizerFast):
                tokenizer_path = path / Dumper.hf_tokenizers / key
                tokenizer_path.mkdir(parents=True, exist_ok=True)
                try:
                    val.save_pretrained(tokenizer_path)  # type: ignore[union-attr]
                except Exception as e:
                    msg = f"Error dumping HF tokenizer {key}: {e}"
                    logger.exception(msg)
            else:
                msg = f"Attribute {key} of type {type(val)} cannot be dumped to file system."
                logger.error(msg)

        with (path / Dumper.simple_attrs).open("w", encoding="utf-8") as file:
            json.dump(simple_attrs, file, ensure_ascii=False, indent=4)

        np.savez(path / Dumper.arrays, allow_pickle=False, **arrays)

    @staticmethod
    def load(  # noqa: C901, PLR0912, PLR0915
        obj: Any,  # noqa: ANN401
        path: Path,
        embedder_config: EmbedderConfig | None = None,
        cross_encoder_config: CrossEncoderConfig | None = None,
    ) -> None:
        """Load attributes from file system."""
        tags: dict[str, Any] = {}
        simple_attrs: dict[str, Any] = {}
        arrays: dict[str, Any] = {}
        embedders: dict[str, Any] = {}
        indexes: dict[str, Any] = {}
        estimators: dict[str, Any] = {}
        cross_encoders: dict[str, Any] = {}
        pydantic_models: dict[str, Any] = {}
        hf_models: dict[str, Any] = {}
        hf_tokenizers: dict[str, Any] = {}
        torch_models: dict[str, Any] = {}

        for child in path.iterdir():
            if child.name == Dumper.tags:
                tags = {tags_dump.name: TagsList.load(tags_dump) for tags_dump in child.iterdir()}
            elif child.name == Dumper.simple_attrs:
                with child.open(encoding="utf-8") as file:
                    simple_attrs = json.load(file)
            elif child.name == Dumper.arrays:
                arrays = dict(np.load(child))
            elif child.name == Dumper.embedders:
                embedders = {
                    embedder_dump.name: Embedder.load(embedder_dump, override_config=embedder_config)
                    for embedder_dump in child.iterdir()
                }
            elif child.name == Dumper.indexes:
                indexes = {index_dump.name: VectorIndex.load(index_dump) for index_dump in child.iterdir()}
            elif child.name == Dumper.estimators:
                estimators = {estimator_dump.name: joblib.load(estimator_dump) for estimator_dump in child.iterdir()}
            elif child.name == Dumper.cross_encoders:
                cross_encoders = {
                    cross_encoder_dump.name: Ranker.load(cross_encoder_dump, override_config=cross_encoder_config)
                    for cross_encoder_dump in child.iterdir()
                }
            elif child.name == Dumper.pydantic_models:
                for model_dir in child.iterdir():
                    try:
                        with (model_dir / "model_dump.json").open("r", encoding="utf-8") as file:
                            content = json.load(file)

                        variable_name = model_dir.name

                        with (model_dir / "class_info.json").open("r", encoding="utf-8") as file:
                            class_info = json.load(file)

                        try:
                            model_type = importlib.import_module(class_info["module"])
                            model_type = getattr(model_type, class_info["name"])
                        except (ImportError, AttributeError) as e:
                            msg = f"Failed to import model type for {variable_name}: {e}"
                            logger.exception(msg)
                            continue

                        try:
                            pydantic_models[variable_name] = model_type.model_validate(content)
                        except Exception as e:
                            msg = f"Failed to reconstruct Pydantic model {variable_name}: {e}"
                            logger.exception(msg)
                            continue
                    except Exception as e:
                        msg = f"Error loading Pydantic model from {model_dir}: {e}"
                        logger.exception(msg)
                        continue
            elif child.name == Dumper.ptuning_models:
                for model_dir in child.iterdir():
                    try:
                        model = AutoModelForSequenceClassification.from_pretrained(model_dir / "base_model")  # type: ignore[no-untyped-call]
                        hf_models[model_dir.name] = PeftModel.from_pretrained(model, model_dir / "peft")
                    except Exception as e:  # noqa: PERF203
                        msg = f"Error loading PeftModel {model_dir.name}: {e}"
                        logger.exception(msg)
            elif child.name == Dumper.hf_models:
                for model_dir in child.iterdir():
                    try:
                        hf_models[model_dir.name] = AutoModelForSequenceClassification.from_pretrained(model_dir)  # type: ignore[no-untyped-call]
                    except Exception as e:  # noqa: PERF203
                        msg = f"Error loading HF model {model_dir.name}: {e}"
                        logger.exception(msg)
            elif child.name == Dumper.hf_tokenizers:
                for tokenizer_dir in child.iterdir():
                    try:
                        hf_tokenizers[tokenizer_dir.name] = AutoTokenizer.from_pretrained(tokenizer_dir)
                    except Exception as e:  # noqa: PERF203
                        msg = f"Error loading HF tokenizer {tokenizer_dir.name}: {e}"
                        logger.exception(msg)
            elif child.name == Dumper.torch_models:
                try:
                    for model_dir in child.iterdir():
                        with (model_dir / "class_info.json").open("r") as f:
                            class_info = json.load(f)
                        module = importlib.import_module(class_info["module"])
                        model_class: BaseTorchModule = getattr(module, class_info["name"])
                        model = model_class.load(model_dir)
                        torch_models[model_dir.name] = model
                except Exception as e:
                    msg = f"Error loading torch model {model_dir.name}: {e}"
                    logger.exception(msg)
            else:
                msg = f"Found unexpected child {child}"
                logger.error(msg)

        obj.__dict__.update(
            tags
            | simple_attrs
            | arrays
            | embedders
            | indexes
            | estimators
            | cross_encoders
            | pydantic_models
            | hf_models
            | hf_tokenizers
            | torch_models
        )
