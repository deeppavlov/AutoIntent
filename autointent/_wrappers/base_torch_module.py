from abc import ABC, abstractmethod
from pathlib import Path

import torch
from torch import nn
from typing_extensions import Self


class BaseTorchModule(nn.Module, ABC):
    @abstractmethod
    def dump(self, path: Path) -> None:
        """Dump torch module to disk.

        This method encapsulates all the logic of dumping module's weights and
        hyperparameters required for initialization from disk and nice inference.

        Args:
            path: path in file system
        """

    @classmethod
    @abstractmethod
    def load(cls, path: Path, device: str | None = None) -> Self:
        """Load torch module from disk.

        This method loads all weights and hyperparameters required for
        initialization from disk and inference.

        Args:
            path: path in file system
            device: torch notation for CPU, CUDA, MPS, etc. By default, it is inferred automatically.
        """

    @property
    def device(self) -> torch.device:
        """Torch device object where this module resides."""
        return next(self.parameters()).device
