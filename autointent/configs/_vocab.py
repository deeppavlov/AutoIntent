from typing import Any

from pydantic import BaseModel


class VocabConfig(BaseModel):
    padding_idx: int = 0
    max_seq_length: int = 50
    vocab: dict[str, int] | None = None
    max_vocab_size: int | None = None

    @classmethod
    def from_search_config(cls, values: dict[str, Any] | BaseModel | None) -> "VocabConfig":
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
