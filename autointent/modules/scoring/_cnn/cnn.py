"""CNNScorer class for scoring."""

from typing import Any
import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
from collections import Counter
import re

from autointent import Context
from autointent._callbacks import REPORTERS_NAMES
from autointent.configs import EmbedderConfig
from autointent.custom_types import ListOfLabels
from autointent.modules.base import BaseScorer
from autointent.modules.scoring._cnn.textcnn import TextCNN


class CNNScorer(BaseScorer):
    """Convolutional Neural Network (CNN) scorer for intent classification."""

    name = "cnn"
    _n_classes: int
    _multilabel: bool
    supports_multilabel = True
    supports_multiclass = True

    def __init__(
        self,
        max_seq_length: int = 50,
        num_train_epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 5e-5,
        seed: int = 0,
        report_to: REPORTERS_NAMES | None = None,
        **cnn_kwargs: dict[str, Any],
    ) -> None:
        self.max_seq_length = max_seq_length
        self.num_train_epochs = num_train_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.seed = seed
        self.report_to = report_to
        self.cnn_config = cnn_kwargs
        
        # Will be initialized during fit()
        self._model = None
        self._vocab = None
        self._padding_idx = 0
        self._unk_token = "<UNK>"
        self._pad_token = "<PAD>"

    @classmethod
    def from_context(
        cls,
        context: Context,
        num_train_epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 5e-5,
        seed: int = 0,
        **cnn_kwargs: dict[str, Any],
    ) -> "CNNScorer":
        return cls(
            num_train_epochs=num_train_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=seed,
            report_to=context.logging_config.report_to,
            **cnn_kwargs,
        )

    def fit(self, utterances: list[str], labels: ListOfLabels, clear_cache: bool = False) -> None:
        if clear_cache:
            self.clear_cache()
        
        self._validate_task(labels)
        self._multilabel = isinstance(labels[0], (list, np.ndarray))
        
        # Build vocabulary and tokenize
        self._build_vocab(utterances)
        
        # Convert text to padded indices
        X = self._text_to_indices(utterances)
        X = torch.tensor(X, dtype=torch.long)
        y = torch.tensor(labels, dtype=torch.long)
        
        # Initialize model
        self._model = TextCNN(
            vocab_size=len(self._vocab),
            n_classes=self._n_classes,
            embed_dim=self.cnn_config.get('embed_dim', 128),
            kernel_sizes=self.cnn_config.get('kernel_sizes', (3, 4, 5)),
            num_filters=self.cnn_config.get('num_filters', 100),
            dropout=self.cnn_config.get('dropout', 0.1),
            padding_idx=self._padding_idx,
            pretrained_embs=self.cnn_config.get('pretrained_embs', None)
        )
        
        # Training
        self._train_model(X, y)

    def predict(self, utterances: list[str]) -> npt.NDArray[Any]:
        if self._model is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        
        X = self._text_to_indices(utterances)
        X = torch.tensor(X, dtype=torch.long)
        
        self._model.eval()
        all_probs = []
        
        with torch.no_grad():
            for i in range(0, len(X), self.batch_size):
                batch_X = X[i:i+self.batch_size]
                outputs = self._model(batch_X)
                if self._multilabel:
                    probs = torch.sigmoid(outputs).cpu().numpy()
                else:
                    probs = torch.softmax(outputs, dim=1).cpu().numpy()
                all_probs.append(probs)
        
        return np.concatenate(all_probs, axis=0) if all_probs else np.array([])

    def _build_vocab(self, utterances: list[str]) -> None:
        """Build vocabulary from training utterances."""
        word_counts = Counter()
        for utterance in utterances:
            words = re.findall(r'\w+', utterance.lower())
            word_counts.update(words)
        
        # Create vocabulary with special tokens
        self._vocab = {
            self._pad_token: 0,
            self._unk_token: 1
        }
        
        # Add words to vocabulary
        for word, _ in word_counts.most_common():
            if word not in self._vocab:
                self._vocab[word] = len(self._vocab)
        
        self._unk_idx = 1
        self._padding_idx = 0

    def _text_to_indices(self, utterances: list[str]) -> list[list[int]]:
        """Convert utterances to padded sequences of word indices."""
        sequences = []
        for utterance in utterances:
            words = re.findall(r'\w+', utterance.lower())
            # Convert words to indices, using UNK for unknown words
            seq = [self._vocab.get(word, self._unk_idx) for word in words]
            # Truncate if too long
            seq = seq[:self.max_seq_length]
            # Pad if too short
            seq = seq + [self._padding_idx] * (self.max_seq_length - len(seq))
            sequences.append(seq)
        return sequences

    def clear_cache(self) -> None:
        self._model = None
        torch.cuda.empty_cache()

    def _train_model(self, X: torch.Tensor, y: torch.Tensor) -> None:
        dataset = torch.utils.data.TensorDataset(X, y)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
        
        criterion = nn.CrossEntropyLoss() if not self._multilabel else nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.learning_rate)
        
        self._model.train()
        for epoch in range(self.num_train_epochs):
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self._model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
        
        self._model.eval()