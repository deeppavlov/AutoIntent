"""TextCNN model for text classification."""

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn


class TextCNN(nn.Module):
    """TextCNN model implementation."""

    def __init__(
        self,
        vocab_size: int = 0,
        n_classes: int = 0,
        embed_dim: int = 128,
        kernel_sizes: list[int] = [3, 4, 5],  # noqa: B006
        num_filters: int = 100,
        dropout: float = 0.1,
        padding_idx: int = 0,
        pretrained_embs: torch.Tensor | None = None,
    ) -> None:
        super().__init__()

        self.vocab_size = vocab_size
        self.n_classes = n_classes
        self.embed_dim = embed_dim
        self.kernel_sizes = kernel_sizes
        self.num_filters = num_filters
        self.dropout_rate = dropout
        self.padding_idx = padding_idx

        if pretrained_embs is not None:
            _, embed_dim = pretrained_embs.shape
            self.embedding = nn.Embedding.from_pretrained(pretrained_embs, freeze=True) # type: ignore[no-untyped-call]
        else:
            self.embedding = nn.Embedding(
                num_embeddings=vocab_size,
                embedding_dim=embed_dim,
                padding_idx=padding_idx
            )

        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=embed_dim,
                out_channels=num_filters,
                kernel_size=k
            ) for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""
        embedded: torch.Tensor = self.embedding(x)
        embedded = embedded.permute(0, 2, 1)
        conved: list[torch.Tensor] = [F.relu(conv(embedded)).max(dim=2)[0] for conv in self.convs]
        concatenated: torch.Tensor = torch.cat(conved, dim=1)
        dropped: torch.Tensor = self.dropout(concatenated)
        return self.fc(dropped) # type: ignore[no-any-return]

    def load(self, model_path: str) -> None:
        """Load model from saved state.

        Args:
            model_path: Path to the saved model state dictionary.
        """
        state_dict = torch.load(model_path)
        self.load_state_dict(state_dict)

    def get_config(self) -> dict[str, int | list[int] | torch.Tensor | None]:
        return {
            "vocab_size": self.vocab_size.item(),
            "n_classes": self.n_classes.item(),
            "embed_dim": self.embed_dim.item(),
            "kernel_sizes": self.kernel_sizes.tolist(),
            "num_filters": self.num_filters.item(),
            "dropout": self.dropout_rate.item(),
            "padding_idx": self.padding_idx.item(),
            "pretrained_embs": self.pretrained_embs,
        }
