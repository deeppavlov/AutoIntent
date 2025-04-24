import torch
import torch.nn as nn
import numpy as np
import numpy.typing as npt
from typing import Any, Optional, Dict, List, Union

from autointent import Context
from autointent._callbacks import REPORTERS_NAMES
from autointent.configs import Config
from autointent.custom_types import ListOfLabels
from autointent.modules.base import BaseScorer
from autointent.context.optimization_info import ScorerArtifact

class RNNConfig(Config):
    """Configuration for RNN models."""
    model_name: str = "rnn"
    embed_dim: int = 128
    hidden_dim: int = 512
    n_layers: int = 2
    dropout: float = 0.1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_seq_length: int = 128
    padding_idx: int = 0
    pretrained_embs: Optional[torch.Tensor] = None

class RNNScorer(BaseScorer):
    """Scorer based on RNN model for text classification."""
    name = "rnn"
    supports_multiclass = True
    supports_multilabel = True
    
    def __init__(
        self,
        rnn_config: Optional[Union[RNNConfig, str, Dict[str, Any]]] = None,
        num_train_epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 5e-5,
        seed: int = 0,
        report_to: Optional[REPORTERS_NAMES] = None,
    ) -> None:
        """Initialize the RNN scorer."""
        self.rnn_config = RNNConfig.from_search_config(rnn_config) if rnn_config else RNNConfig()
        self.num_train_epochs = num_train_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.seed = seed
        self.report_to = report_to
        self._artifact = None
        
    @classmethod
    def from_context(
        cls,
        context: Context,
        rnn_config: Optional[Union[RNNConfig, str, Dict[str, Any]]] = None,
        num_train_epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 5e-5,
        seed: int = 0,
    ) -> "RNNScorer":
        """Create a RNNScorer from context."""
        report_to = context.logging_config.report_to

        return cls(
            rnn_config=rnn_config,
            num_train_epochs=num_train_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=seed,
            report_to=report_to,
        )
    
    def get_embedder_config(self) -> Dict[str, Any]:
        """Get the configuration of the embedder."""
        return self.rnn_config.model_dump()

    def _validate_task(self, labels: ListOfLabels) -> None:
        """Validate the task type and set appropriate attributes."""
        if isinstance(labels[0], list):
            self._multilabel = True
            self._n_classes = len(labels[0])
        else:
            self._multilabel = False
            self._n_classes = max(labels) + 1

    def __initialize_model(self, vocab_size: int) -> None:
        """Initialize the RNN model."""
        self._model = SupervisedRNNClassifier(
            vocab_size=vocab_size,
            n_classes=self._n_classes,
            embed_dim=self.rnn_config.embed_dim,
            hidden_dim=self.rnn_config.hidden_dim,
            n_layers=self.rnn_config.n_layers,
            padding_idx=self.rnn_config.padding_idx,
            dropout=self.rnn_config.dropout,
            pretrained_embs=self.rnn_config.pretrained_embs
        )
        self._model.to(self.rnn_config.device)

    def fit(
        self,
        utterances: List[str],
        labels: ListOfLabels,
    ) -> None:
        """Fit the model to the given data."""
        if hasattr(self, "_model"):
            self.clear_cache()
        self._validate_task(labels)
        
        # Create vocabulary
        self._create_vocab(utterances)
        
        # Initialize model
        self.__initialize_model(len(self._vocab))
        
        # Convert utterances to sequences
        X = self._texts_to_sequences(utterances)
        
        # Convert labels to tensors
        if self._multilabel:
            y = torch.tensor(labels, dtype=torch.float)
        else:
            y = torch.tensor(labels, dtype=torch.long)
        
        # Train the model
        self._train_model(X, y)
        
    def _create_vocab(self, utterances: List[str]) -> None:
        """Create vocabulary from utterances."""
        # Create a simple vocabulary based on all words in the dataset
        unique_words = set()
        for text in utterances:
            for word in text.lower().split():
                unique_words.add(word)
        
        self._vocab = {"<PAD>": 0, "<UNK>": 1}
        for i, word in enumerate(unique_words):
            self._vocab[word] = i + 2
            
    def _texts_to_sequences(self, texts: List[str]) -> torch.Tensor:
        """Convert texts to sequences using the vocabulary."""
        # Convert texts to sequences using the vocabulary
        sequences = []
        for text in texts:
            sequence = []
            for word in text.lower().split():
                sequence.append(self._vocab.get(word, self._vocab["<UNK>"]))
            sequences.append(sequence)
        
        # Pad sequences
        max_len = min(max(len(seq) for seq in sequences), self.rnn_config.max_seq_length)
        padded_sequences = []
        for seq in sequences:
            if len(seq) > max_len:
                padded_seq = seq[:max_len]
            else:
                padded_seq = seq + [self._vocab["<PAD>"]] * (max_len - len(seq))
            padded_sequences.append(padded_seq)
        
        return torch.tensor(padded_sequences, dtype=torch.long)
    
    def _train_model(self, X: torch.Tensor, y: torch.Tensor) -> None:
        """Train the model."""
        self._model.train()
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.learning_rate)
        
        if self._multilabel:
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.CrossEntropyLoss()
        
        X = X.to(self.rnn_config.device)
        y = y.to(self.rnn_config.device)
        
        dataset = torch.utils.data.TensorDataset(X, y)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )
        
        torch.manual_seed(self.seed)
        
        for epoch in range(self.num_train_epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs, _ = self._model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
        
        self._model.eval()
    
    def predict(self, utterances: List[str]) -> npt.NDArray[Any]:
        """Predict probabilities for utterances."""
        if not hasattr(self, "_model") or not hasattr(self, "_vocab"):
            msg = "Model is not trained. Call fit() first."
            raise RuntimeError(msg)
        
        X = self._texts_to_sequences(utterances)
        X = X.to(self.rnn_config.device)
        
        self._model.eval()
        all_predictions = []
        
        with torch.no_grad():
            for i in range(0, len(X), self.batch_size):
                batch_X = X[i:i+self.batch_size]
                outputs, _ = self._model(batch_X)
                
                if self._multilabel:
                    batch_predictions = torch.sigmoid(outputs).cpu().numpy()
                else:
                    batch_predictions = torch.softmax(outputs, dim=1).cpu().numpy()
                
                all_predictions.append(batch_predictions)
        
        return np.vstack(all_predictions) if all_predictions else np.array([])
    
    def clear_cache(self) -> None:
        """Clear model cache."""
        if hasattr(self, "_model"):
            del self._model


class SupervisedRNNClassifier(nn.Module):
    def __init__(self, 
                 vocab_size, 
                 n_classes, 
                 embed_dim=128, 
                 hidden_dim=512, 
                 n_layers=2, 
                 padding_idx=0,
                 dropout=0.1,
                 pretrained_embs=None
        ):
        super().__init__()
        if pretrained_embs is not None:
            _, embed_dim = pretrained_embs.shape
            self.embedding = nn.Embedding.from_pretrained(pretrained_embs, freeze=True)
        else:
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.rnn = nn.LSTM(embed_dim, hidden_dim, num_layers=n_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, n_classes)
    
    def forward(self, text):
        embedded = self.embedding(text)
        outputs, (hidden, _) = self.rnn(embedded)
        return self.fc(outputs[:,-1]), hidden[-1]