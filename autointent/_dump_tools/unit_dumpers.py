import importlib
import json
import logging
from pathlib import Path
from typing import Any, TypeVar

import aiofiles
import joblib
import numpy as np
import numpy.typing as npt
from catboost import CatBoostClassifier
from peft import PeftModel
from pydantic import BaseModel
from sklearn.base import BaseEstimator
from transformers import (  # type: ignore[attr-defined]
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

from autointent import Embedder, Ranker, VectorIndex
from autointent._wrappers import BaseTorchModuleWithVocab
from autointent.schemas import TagsList

from .base import BaseObjectDumper, ModuleSimpleAttributes

T = TypeVar("T")
logger = logging.getLogger(__name__)


class TagsListDumper(BaseObjectDumper[TagsList]):
    dir_or_file_name = "tags"

    @staticmethod
    def dump(obj: TagsList, path: Path, exists_ok: bool) -> None:  # noqa: ARG004
        obj.dump(path)

    @staticmethod
    def load(path: Path, **kwargs: Any) -> TagsList:  # noqa: ANN401, ARG004
        return TagsList.load(path)

    @classmethod
    def check_isinstance(cls, obj: Any) -> bool:  # noqa: ANN401
        return isinstance(obj, TagsList)


class SimpleAttributesDumper(BaseObjectDumper[dict[str, ModuleSimpleAttributes]]):
    dir_or_file_name = "simple_attrs.json"

    @staticmethod
    def dump(obj: dict[str, ModuleSimpleAttributes], path: Path, exists_ok: bool) -> None:  # noqa: ARG004
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            json.dump(obj, file, ensure_ascii=False, indent=4)

    @staticmethod
    def load(path: Path, **kwargs: Any) -> dict[str, ModuleSimpleAttributes]:  # noqa: ANN401, ARG004
        with path.open(encoding="utf-8") as file:
            return json.load(file)  # type: ignore[no-any-return]

    @classmethod
    def check_isinstance(cls, obj: Any) -> bool:  # noqa: ANN401, ARG003
        # This dumper is used for collections of simple attributes, not individual objects
        # It should not match individual objects in the main loop
        return False


class ArraysDumper(BaseObjectDumper[dict[str, npt.NDArray[Any]]]):
    dir_or_file_name = "arrays.npz"

    @staticmethod
    def dump(obj: dict[str, npt.NDArray[Any]], path: Path, exists_ok: bool) -> None:  # noqa: ARG004
        np.savez(path, allow_pickle=False, **obj)

    @staticmethod
    def load(path: Path, **kwargs: Any) -> dict[str, npt.NDArray[Any]]:  # noqa: ANN401, ARG004
        return dict(np.load(path))

    @classmethod
    def check_isinstance(cls, obj: Any) -> bool:  # noqa: ANN401, ARG003
        # This dumper is used for collections of arrays, not individual objects
        # It should not match individual objects in the main loop
        return False


class EmbedderDumper(BaseObjectDumper[Embedder]):
    dir_or_file_name = "embedders"

    @staticmethod
    def dump(obj: Embedder, path: Path, exists_ok: bool) -> None:  # noqa: ARG004
        obj.dump(path)

    @staticmethod
    def load(path: Path, **kwargs: Any) -> Embedder:  # noqa: ANN401
        embedder_config = kwargs.get("embedder_config")
        return Embedder.load(path, override_config=embedder_config)

    @classmethod
    def check_isinstance(cls, obj: Any) -> bool:  # noqa: ANN401
        return isinstance(obj, Embedder)


class VectorIndexDumper(BaseObjectDumper[VectorIndex]):
    dir_or_file_name = "vector_indexes"

    @staticmethod
    def dump(obj: VectorIndex, path: Path, exists_ok: bool) -> None:  # noqa: ARG004
        obj.dump(path)

    @staticmethod
    def load(path: Path, **kwargs: Any) -> VectorIndex:  # noqa: ANN401, ARG004
        return VectorIndex.load(path)

    @classmethod
    def check_isinstance(cls, obj: Any) -> bool:  # noqa: ANN401
        return isinstance(obj, VectorIndex)


class EstimatorDumper(BaseObjectDumper[BaseEstimator]):
    dir_or_file_name = "estimators"

    @staticmethod
    def dump(obj: BaseEstimator, path: Path, exists_ok: bool) -> None:  # noqa: ARG004
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(obj, path)

    @staticmethod
    def load(path: Path, **kwargs: Any) -> BaseEstimator:  # noqa: ANN401, ARG004
        return joblib.load(path)

    @classmethod
    def check_isinstance(cls, obj: Any) -> bool:  # noqa: ANN401
        return isinstance(obj, BaseEstimator)


class RankerDumper(BaseObjectDumper[Ranker]):
    dir_or_file_name = "cross_encoders"

    @staticmethod
    def dump(obj: Ranker, path: Path, exists_ok: bool) -> None:  # noqa: ARG004
        obj.save(str(path))

    @staticmethod
    def load(path: Path, **kwargs: Any) -> Ranker:  # noqa: ANN401
        cross_encoder_config = kwargs.get("cross_encoder_config")
        return Ranker.load(path, override_config=cross_encoder_config)

    @classmethod
    def check_isinstance(cls, obj: Any) -> bool:  # noqa: ANN401
        return isinstance(obj, Ranker)


class PydanticModelDumper(BaseObjectDumper[BaseModel]):
    dir_or_file_name = "pydantic"

    @staticmethod
    def dump(obj: BaseModel, path: Path, exists_ok: bool) -> None:
        class_info = {"name": obj.__class__.__name__, "module": obj.__class__.__module__}
        path.mkdir(parents=True, exist_ok=exists_ok)
        with (path / "class_info.json").open("w", encoding="utf-8") as file:
            json.dump(class_info, file, ensure_ascii=False, indent=4)
        with (path / "model_dump.json").open("w", encoding="utf-8") as file:
            json.dump(obj.model_dump(), file, ensure_ascii=False, indent=4)

    @staticmethod
    async def dump_async(obj: BaseModel, path: Path, exists_ok: bool) -> None:
        class_info = {"name": obj.__class__.__name__, "module": obj.__class__.__module__}
        path.mkdir(parents=True, exist_ok=exists_ok)
        async with aiofiles.open(path / "class_info.json", mode="w", encoding="utf-8") as file:
            await file.write(json.dumps(class_info, ensure_ascii=False, indent=4))
        async with aiofiles.open(path / "model_dump.json", mode="w", encoding="utf-8") as file:
            await file.write(json.dumps(obj.model_dump(), ensure_ascii=False, indent=4))

    @staticmethod
    def load(path: Path, **kwargs: Any) -> BaseModel:  # noqa: ANN401, ARG004
        with (path / "model_dump.json").open("r", encoding="utf-8") as file:
            content = json.load(file)

        with (path / "class_info.json").open("r", encoding="utf-8") as file:
            class_info = json.load(file)

        model_type = importlib.import_module(class_info["module"])
        model_type = getattr(model_type, class_info["name"])
        return model_type.model_validate(content)  # type: ignore[no-any-return]

    @staticmethod
    async def load_async(path: Path, **kwargs: Any) -> BaseModel:  # noqa: ANN401, ARG004
        async with aiofiles.open(path / "model_dump.json", encoding="utf-8") as file:
            content_str = await file.read()
            content = json.loads(content_str)

        async with aiofiles.open(path / "class_info.json", encoding="utf-8") as file:
            class_info_str = await file.read()
            class_info = json.loads(class_info_str)

        model_type = importlib.import_module(class_info["module"])
        model_type = getattr(model_type, class_info["name"])
        return model_type.model_validate(content)  # type: ignore[no-any-return]

    @classmethod
    def check_isinstance(cls, obj: Any) -> bool:  # noqa: ANN401
        return isinstance(obj, BaseModel)


class PeftModelDumper(BaseObjectDumper[PeftModel]):
    dir_or_file_name = "peft_models"

    @staticmethod
    def dump(obj: PeftModel, path: Path, exists_ok: bool) -> None:
        path.mkdir(parents=True, exist_ok=exists_ok)
        if obj._is_prompt_learning:  # noqa: SLF001
            # strategy to save prompt learning models: save prompt encoder and bert classifier separately
            ptuning_path = path / "ptuning"
            ptuning_path.mkdir(parents=True, exist_ok=exists_ok)
            obj.save_pretrained(str(ptuning_path / "peft"))
            obj.base_model.save_pretrained(ptuning_path / "base_model")
        else:
            # strategy to save lora models: merge adapters and save as usual hugging face model
            lora_path = path / "lora"
            lora_path.mkdir(parents=True, exist_ok=exists_ok)
            merged_model: PreTrainedModel = obj.merge_and_unload()
            merged_model.save_pretrained(lora_path)

    @staticmethod
    def load(path: Path, **kwargs: Any) -> PeftModel:  # noqa: ANN401, ARG004
        if (path / "ptuning").exists():
            # prompt learning model
            ptuning_path = path / "ptuning"
            model = AutoModelForSequenceClassification.from_pretrained(ptuning_path / "base_model")
            return PeftModel.from_pretrained(model, ptuning_path / "peft")
        if (path / "lora").exists():
            # merged lora model
            lora_path = path / "lora"
            return AutoModelForSequenceClassification.from_pretrained(lora_path)  # type: ignore[no-any-return]
        msg = f"Invalid PeftModel directory structure at {path}. Expected 'ptuning' or 'lora' subdirectory."
        raise ValueError(msg)

    @classmethod
    def check_isinstance(cls, obj: Any) -> bool:  # noqa: ANN401
        return isinstance(obj, PeftModel)


class HFModelDumper(BaseObjectDumper[PreTrainedModel]):
    dir_or_file_name = "hf_models"

    @staticmethod
    def dump(obj: PreTrainedModel, path: Path, exists_ok: bool) -> None:
        path.mkdir(parents=True, exist_ok=exists_ok)
        obj.save_pretrained(path)

    @staticmethod
    def load(path: Path, **kwargs: Any) -> PreTrainedModel:  # noqa: ANN401, ARG004
        return AutoModelForSequenceClassification.from_pretrained(path)  # type: ignore[no-any-return]

    @classmethod
    def check_isinstance(cls, obj: Any) -> bool:  # noqa: ANN401
        return isinstance(obj, PreTrainedModel)


class HFTokenizerDumper(BaseObjectDumper[PreTrainedTokenizer | PreTrainedTokenizerFast]):
    dir_or_file_name = "hf_tokenizers"

    @staticmethod
    def dump(obj: PreTrainedTokenizer | PreTrainedTokenizerFast, path: Path, exists_ok: bool) -> None:
        path.mkdir(parents=True, exist_ok=exists_ok)
        obj.save_pretrained(path)

    @staticmethod
    def load(path: Path, **kwargs: Any) -> PreTrainedTokenizer | PreTrainedTokenizerFast:  # noqa: ANN401, ARG004
        return AutoTokenizer.from_pretrained(path)  # type: ignore[no-any-return,no-untyped-call]

    @classmethod
    def check_isinstance(cls, obj: Any) -> bool:  # noqa: ANN401
        return isinstance(obj, PreTrainedTokenizer | PreTrainedTokenizerFast)


class TorchModelDumper(BaseObjectDumper[BaseTorchModuleWithVocab]):
    dir_or_file_name = "torch_models"

    @staticmethod
    def dump(obj: BaseTorchModuleWithVocab, path: Path, exists_ok: bool) -> None:
        path.mkdir(parents=True, exist_ok=exists_ok)
        class_info = {
            "module": obj.__class__.__module__,
            "name": obj.__class__.__name__,
        }
        with (path / "class_info.json").open("w") as f:
            json.dump(class_info, f)
        obj.dump(path)

    @staticmethod
    def load(path: Path, **kwargs: Any) -> BaseTorchModuleWithVocab:  # noqa: ANN401, ARG004
        with (path / "class_info.json").open("r") as f:
            class_info = json.load(f)
        module = importlib.import_module(class_info["module"])
        model_class: BaseTorchModuleWithVocab = getattr(module, class_info["name"])
        return model_class.load(path)

    @classmethod
    def check_isinstance(cls, obj: Any) -> bool:  # noqa: ANN401
        return isinstance(obj, BaseTorchModuleWithVocab)


class CatBoostDumper(BaseObjectDumper[CatBoostClassifier]):
    dir_or_file_name = "catboost_models"

    @staticmethod
    def dump(obj: CatBoostClassifier, path: Path, exists_ok: bool) -> None:  # noqa: ARG004
        path.parent.mkdir(parents=True, exist_ok=True)
        obj.save_model(str(path), format="cbm")

    @staticmethod
    def load(path: Path, **kwargs: Any) -> CatBoostClassifier:  # noqa: ANN401, ARG004
        model = CatBoostClassifier()
        model.load_model(str(path))
        return model

    @classmethod
    def check_isinstance(cls, obj: Any) -> bool:  # noqa: ANN401
        return isinstance(obj, CatBoostClassifier)
