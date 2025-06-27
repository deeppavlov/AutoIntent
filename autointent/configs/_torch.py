from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import Self

from autointent._utils import detect_device


class FromDictMixin:
    @classmethod
    def from_search_config(cls, values: dict[str, Any] | BaseModel | None) -> Self:
        """Validate the model configuration.

        This classmethod is used to parse dictionaries that occur in search space configurations.

        Args:
            values: Model configuration values.

        Returns:
            Model configuration.
        """
        if values is None:
            return cls()
        if isinstance(values, BaseModel):
            return values  # type: ignore[return-value]
        return cls(**values)


class VocabConfig(BaseModel, FromDictMixin):
    model_config = ConfigDict(extra="forbid")
    padding_idx: int = 0
    max_seq_length: int = 50
    vocab: dict[str, int] | None = None
    max_vocab_size: int | None = None


class TorchTrainingConfig(BaseModel, FromDictMixin):
    model_config = ConfigDict(extra="forbid")
    num_train_epochs: int = 3
    batch_size: int = 8
    learning_rate: float = 5e-5
    seed: int = 42
    device: str = Field(default_factory=detect_device)
