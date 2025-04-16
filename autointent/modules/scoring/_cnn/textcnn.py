"""TextCNN model for text classification."""

from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    """TextCNN model implementation."""

    def __init__(
        self,
        vocab_size: int,
        n_classes: int,
        embed_dim: int = 128,
        kernel_sizes: Tuple[int, ...] = (3, 4, 5),
        num_filters: int = 100,
        dropout: float = 0.1,
        padding_idx: int = 0,
        pretrained_embs: Optional[torch.Tensor] = None,
    ) -> None:
        """Initialize TextCNN model."""
        super().__init__()
        
        if pretrained_embs is not None:
            _, embed_dim = pretrained_embs.shape
            self.embedding = nn.Embedding.from_pretrained(pretrained_embs, freeze=True)
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
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        x = [F.relu(conv(x)).max(dim=2)[0] for conv in self.convs]
        x = torch.cat(x, dim=1)
        x = self.dropout(x)
        return self.fc(x)